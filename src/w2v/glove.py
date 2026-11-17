"""GloVe (Pennington, Socher & Manning 2014): co-occurrence counting and weighted least squares.

The model fits, for every non-zero co-occurrence count :math:`X_{ij}`,

.. math::

    J = \\sum_{i,j:\\,X_{ij}>0} f(X_{ij})\\,\\big(w_i^\\top \\tilde w_j + b_i + \\tilde b_j - \\log X_{ij}\\big)^2,
    \\qquad f(x) = \\min\\big(1, (x/x_{\\max})^{\\alpha}\\big),

with AdaGrad on mini-batches of non-zero entries. Gradients are written out by
hand (see ``glove_loss_and_grads``) and checked against autograd in the tests.
The final embedding is :math:`w_i + \\tilde w_i`, as in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .corpus import prepare_text8
from .data import DATA_DIR
from .embeddings import Embeddings
from .evaluate import evaluate_analogies


# --------------------------------------------------------------------------- co-occurrence
def _merge(keys_a: torch.Tensor, vals_a: torch.Tensor, keys_b: torch.Tensor, vals_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum two sparse (key, value) lists into one with unique sorted keys."""
    keys = torch.cat([keys_a, keys_b])
    vals = torch.cat([vals_a, vals_b])
    uk, inv = torch.unique(keys, sorted=True, return_inverse=True)
    out = torch.zeros(uk.shape[0], dtype=vals.dtype, device=vals.device)
    out.index_add_(0, inv, vals)
    return uk, out


@torch.no_grad()
def cooccurrence(
    ids: torch.Tensor,
    vocab_size: int,
    window: int = 10,
    distance_weighting: bool = True,
    chunk: int = 8_000_000,
    partitions: int = 1,
    out_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric word–context co-occurrence matrix in COO form.

    A pair of tokens ``d`` positions apart (``1 <= d <= window``) adds ``1/d``
    (or 1 without distance weighting) to both :math:`X_{ij}` and :math:`X_{ji}`,
    matching the symmetric-window setting of the GloVe paper.

    Counting is done with sort-based ``unique`` + ``index_add_`` on flattened keys
    :math:`iV + j`. To bound peak memory the rows are split into ``partitions``
    groups (``i mod partitions``) that are counted independently, and the corpus
    is streamed in ``chunk``-token slices.

    Returns ``(rows, cols, values)`` sorted by ``(row, col)`` within each partition.
    """
    ids = ids.long()
    dev = ids.device
    out_device = out_device or dev
    V, n = vocab_size, ids.shape[0]
    out_r, out_c, out_v = [], [], []
    for p in range(partitions):
        acc_k = torch.empty(0, dtype=torch.long, device=dev)
        acc_v = torch.empty(0, dtype=torch.float32, device=dev)
        for d in range(1, window + 1):
            w = 1.0 / d if distance_weighting else 1.0
            for s in range(0, n - d, chunk):
                e = min(s + chunk, n - d)
                left, right = ids[s:e], ids[s + d : e + d]
                k = torch.cat([left * V + right, right * V + left])  # both directions -> symmetric X
                if partitions > 1:
                    k = k[(k // V) % partitions == p]
                uk, inv = torch.unique(k, sorted=True, return_inverse=True)
                cnt = torch.zeros(uk.shape[0], dtype=torch.float32, device=dev)
                cnt.index_add_(0, inv, torch.full(k.shape, w, dtype=torch.float32, device=dev))
                del k, inv
                acc_k, acc_v = _merge(acc_k, acc_v, uk, cnt)
                del uk, cnt
        out_r.append((acc_k // V).int().to(out_device))
        out_c.append((acc_k % V).int().to(out_device))
        out_v.append(acc_v.to(out_device))
        del acc_k, acc_v
    return torch.cat(out_r), torch.cat(out_c), torch.cat(out_v)


# --------------------------------------------------------------------------- model
@dataclass
class GloVeGrads:
    loss: torch.Tensor
    grad_w: torch.Tensor
    grad_c: torch.Tensor
    grad_bw: torch.Tensor
    grad_bc: torch.Tensor


def weighting(x: torch.Tensor, x_max: float, alpha: float) -> torch.Tensor:
    """GloVe weighting function f(x) = min(1, (x/x_max)^alpha)."""
    return torch.clamp(x / x_max, max=1.0) ** alpha


def glove_loss_and_grads(
    w: torch.Tensor, c: torch.Tensor, bw: torch.Tensor, bc: torch.Tensor, x: torch.Tensor, x_max: float, alpha: float
) -> GloVeGrads:
    """Loss ``0.5 * sum f(x) (w.c + bw + bc - log x)^2`` and its closed-form gradients.

    Shapes: ``w, c`` are ``(B, D)``; ``bw, bc, x`` are ``(B,)``.
    """
    diff = (w * c).sum(-1) + bw + bc - torch.log(x)
    fx = weighting(x, x_max, alpha)
    fdiff = fx * diff
    loss = 0.5 * (fdiff * diff).sum()
    return GloVeGrads(loss, fdiff.unsqueeze(-1) * c, fdiff.unsqueeze(-1) * w, fdiff, fdiff)


_SCRATCH: dict[tuple[int, ...], torch.Tensor] = {}


def _scratch(param: torch.Tensor) -> torch.Tensor:
    """Persistent all-zero buffer shaped like ``param`` (cleared by the caller after use)."""
    key = (param.data_ptr(), *param.shape)
    if key not in _SCRATCH:
        _SCRATCH[key] = torch.zeros_like(param)
    return _SCRATCH[key]


def _counts(param: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    cnt = _scratch(param.new_empty(param.shape[0]).zero_()) if False else None
    key = ("cnt", param.data_ptr(), param.shape[0])
    if key not in _SCRATCH:
        _SCRATCH[key] = torch.zeros(param.shape[0], dtype=param.dtype, device=param.device)
    cnt = _SCRATCH[key]
    cnt.index_add_(0, idx, torch.ones_like(idx, dtype=param.dtype))
    return cnt


class GloVe:
    """Word/context vectors and biases trained with per-parameter AdaGrad."""

    def __init__(self, vocab_size: int, dim: int, device: torch.device | str = "cpu", seed: int = 0, average_duplicates: bool = True):
        g = torch.Generator().manual_seed(seed)
        self.average_duplicates = average_duplicates
        init = lambda *s: ((torch.rand(*s, generator=g) - 0.5) / dim).to(device)  # noqa: E731
        self.W, self.C = init(vocab_size, dim), init(vocab_size, dim)
        self.bW, self.bC = init(vocab_size), init(vocab_size)
        # AdaGrad accumulators start at 1, as in the reference glove.c
        self.gW = torch.ones_like(self.W)
        self.gC = torch.ones_like(self.C)
        self.gbW = torch.ones_like(self.bW)
        self.gbC = torch.ones_like(self.bC)
        # persistent all-zero scratch space for the sort-free sparse AdaGrad (see ``_adagrad``)
        self._buf = {id(t): torch.zeros_like(t) for t in (self.W, self.C, self.bW, self.bC)}
        self._cnt = {id(t): torch.zeros(vocab_size, device=t.device) for t in (self.W, self.C, self.bW, self.bC)}
        self.dim = dim

    def _adagrad(self, param: torch.Tensor, acc: torch.Tensor, idx: torch.Tensor, grad: torch.Tensor, lr: float) -> None:
        """Sparse AdaGrad step in the form used by the reference glove.c.

        Duplicate rows within the batch are summed first. Like glove.c, the
        accumulator stores squared *scaled* gradients ``(lr * g)^2`` and the step
        uses the accumulator from before the update:
        ``param -= lr * g / sqrt(acc); acc += (lr * g)^2``.

        Sorting for ``torch.unique`` dominated the cost of the first version, so
        instead the scaled gradients are scatter-added into a persistent dense
        buffer; every occurrence of a row then applies ``1/count`` of that row's
        update, which sums to exactly one update per distinct row. The touched
        buffer rows are zeroed afterwards. Everything is O(batch * dim).

        glove.c applies one entry at a time, so a frequent word receives many small steps
        per epoch. In a large batch the same word appears hundreds of times and summing
        those gradients gives one huge first step (it diverged for ``lr >= 0.1``), so by
        default (``average_duplicates=True``) the duplicate gradients are *averaged*.
        """
        buf, cnt = self._buf[id(param)], self._cnt[id(param)]
        buf.index_add_(0, idx, grad, alpha=lr)
        cnt.index_add_(0, idx, torch.ones(idx.shape[0], device=idx.device))
        shape = (-1, *([1] * (grad.dim() - 1)))
        inv = (1.0 / cnt[idx]).view(shape)
        g = buf[idx] * inv if self.average_duplicates else buf[idx]
        param.index_add_(0, idx, -(g / torch.sqrt(acc[idx])) * inv)
        acc.index_add_(0, idx, g * g * inv)
        buf[idx] = 0.0
        cnt[idx] = 0.0

    @torch.no_grad()
    def step(self, i: torch.Tensor, j: torch.Tensor, x: torch.Tensor, lr: float, x_max: float, alpha: float, clip: float = 100.0) -> torch.Tensor:
        g = glove_loss_and_grads(self.W[i], self.C[j], self.bW[i], self.bC[j], x, x_max, alpha)
        # glove.c clips the lr-scaled gradient; we clip the raw gradient at clip / lr equivalently
        clip = clip / lr
        gw, gc = g.grad_w.clamp(-clip, clip), g.grad_c.clamp(-clip, clip)
        gbw, gbc = g.grad_bw.clamp(-clip, clip), g.grad_bc.clamp(-clip, clip)
        self._adagrad(self.W, self.gW, i, gw, lr)
        self._adagrad(self.C, self.gC, j, gc, lr)
        self._adagrad(self.bW, self.gbW, i, gbw, lr)
        self._adagrad(self.bC, self.gbC, j, gbc, lr)
        return g.loss

    def vectors(self) -> np.ndarray:
        return (self.W + self.C).cpu().numpy()


# --------------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GloVe on text8")
    p.add_argument("--dim", type=int, default=300)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--x-max", type=float, default=100.0)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--partitions", type=int, default=4, help="row partitions for co-occurrence counting (memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--average-duplicates", action=argparse.BooleanOptionalAction, default=True,
                   help="average (default) or sum the gradients of rows repeated within a batch")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("runs/glove"))
    return p.parse_args(argv)


def train(args: argparse.Namespace) -> Embeddings:
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed)  # the (CPU-resident) triplets are shuffled on the CPU
    vocab, ids_np = prepare_text8(args.min_count)
    t0 = time.time()
    cache = DATA_DIR / f"text8_min{args.min_count}_cooc_w{args.window}.npz"
    if cache.exists():
        z = np.load(cache)
        rows, cols, vals = (torch.from_numpy(z[k]) for k in ("rows", "cols", "vals"))
    else:  # counting on the CPU keeps GPU memory free for training; the result is cached on disk
        rows, cols, vals = cooccurrence(torch.from_numpy(ids_np), len(vocab), args.window, partitions=args.partitions)
        np.savez(cache, rows=rows.numpy(), cols=cols.numpy(), vals=vals.numpy())
    t_cooc = time.time() - t0
    nnz = rows.shape[0]
    peak = lambda: torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0  # noqa: E731
    print(f"co-occurrence: {nnz:,} non-zeros ({nnz / len(vocab) ** 2:.2%} dense) in {t_cooc:.0f}s  peak GPU {peak():.0f} MB", flush=True)
    model = GloVe(len(vocab), args.dim, device=dev, seed=args.seed, average_duplicates=args.average_duplicates)
    hist: dict[str, list] = {"epoch": [], "loss": [], "epoch_eval": []}
    for ep in range(args.epochs):
        perm = torch.randperm(nnz, generator=gen)
        r_ep, c_ep, v_ep = rows[perm], cols[perm], vals[perm]  # one shuffled copy per epoch (CPU)
        tot = torch.zeros((), device=dev)
        for s in range(0, nnz, args.batch):
            sl = slice(s, s + args.batch)
            tot += model.step(r_ep[sl].to(dev).long(), c_ep[sl].to(dev).long(), v_ep[sl].to(dev), args.lr, args.x_max, args.alpha)
        del r_ep, c_ep, v_ep
        hist["epoch"].append(ep + 1)
        hist["loss"].append(float(tot) / nnz)
        msg = f"epoch {ep + 1}/{args.epochs}  loss/nnz {hist['loss'][-1]:.4f}  [{time.time() - t0:.0f}s, peak {peak():.0f} MB]"
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            acc = evaluate_analogies(Embeddings(vocab.itos, model.vectors()))
            hist["epoch_eval"].append({"epoch": ep + 1, "elapsed_s": time.time() - t0, **acc.as_dict()})
            msg += f"  analogy sem {acc.semantic:.3f} syn {acc.syntactic:.3f} all {acc.overall:.3f}"
        print(msg, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    emb = Embeddings(vocab.itos, model.vectors())
    emb.save(args.out / "vectors.npz")
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update(train_seconds=time.time() - t0, cooc_seconds=t_cooc, nnz=nnz)
    cfg["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0
    (args.out / "history.json").write_text(json.dumps({"config": cfg, **hist}, indent=1))
    print(f"saved {args.out} ({cfg['train_seconds']:.0f}s, peak GPU {cfg['peak_gpu_mb']:.0f} MB)")
    return emb


def main(argv: list[str] | None = None) -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    train(parse_args(argv))


if __name__ == "__main__":
    main()
