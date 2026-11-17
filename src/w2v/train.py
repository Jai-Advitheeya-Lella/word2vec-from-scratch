"""Train skip-gram / CBOW with negative sampling on text8.

Example::

    python -m w2v.train --model skipgram --epochs 5 --out runs/sgns
    python -m w2v.train --model cbow --epochs 5 --out runs/cbow
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .corpus import prepare_text8
from .embeddings import Embeddings
from .evaluate import evaluate_analogies
from .sampling import NegativeSampler, cbow_contexts, skipgram_pairs, subsample
from .sgns import Word2Vec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["skipgram", "cbow"], default="skipgram")
    p.add_argument("--dim", type=int, default=200)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--negative", type=int, default=5)
    p.add_argument("--sample", type=float, default=1e-4, help="sub-sampling threshold t")
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.025, help="initial SGD learning rate (linearly decayed)")
    p.add_argument("--min-lr-frac", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=1024, help="examples per SGD step")
    p.add_argument("--chunk", type=int, default=2_000_000, help="corpus tokens per shuffled chunk")
    p.add_argument("--max-tokens", type=int, default=None, help="truncate the corpus (debugging)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out", type=Path, default=Path("runs/sgns"))
    return p.parse_args(argv)


def train(args: argparse.Namespace) -> Embeddings:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    vocab, ids_np = prepare_text8(args.min_count, max_tokens=args.max_tokens)
    print(f"corpus: {len(ids_np):,} tokens, vocab {len(vocab):,}")
    ids = torch.from_numpy(ids_np.astype(np.int64)).to(dev)
    keep = torch.from_numpy(vocab.keep_probs(args.sample)).to(dev)
    sampler = NegativeSampler(torch.from_numpy(vocab.noise_dist(0.75)), device=dev)
    model = Word2Vec(len(vocab), args.dim, device=dev, seed=args.seed)

    n_chunks = (len(ids) + args.chunk - 1) // args.chunk
    total_chunks = n_chunks * args.epochs
    history: dict[str, list] = {"step": [], "loss": [], "lr": [], "progress": [], "epoch_eval": []}
    step, done_chunks = 0, 0
    run_loss = torch.zeros((), device=dev)
    run_n = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        n_examples = 0
        for ci in range(n_chunks):
            chunk = subsample(ids[ci * args.chunk : (ci + 1) * args.chunk], keep, gen)
            if args.model == "skipgram":
                inputs, targets = skipgram_pairs(chunk, args.window, gen)
            else:
                targets, inputs = cbow_contexts(chunk, args.window, gen)
            n = targets.shape[0]
            n_examples += n
            for s in range(0, n, args.batch):
                frac = (done_chunks + s / n) / total_chunks
                lr = args.lr * max(args.min_lr_frac, 1.0 - frac)
                tb = targets[s : s + args.batch]
                ib = inputs[s : s + args.batch]
                neg = sampler((tb.shape[0], args.negative), gen)
                if args.model == "skipgram":
                    loss = model.skipgram_step(ib, tb, neg, lr)
                else:
                    loss = model.cbow_step(tb, ib, neg, lr)
                run_loss += loss
                run_n += tb.shape[0]
                step += 1
                if step % args.log_every == 0:
                    history["step"].append(step)
                    history["loss"].append(float(run_loss) / run_n)
                    history["lr"].append(lr)
                    history["progress"].append(frac)
                    run_loss.zero_()
                    run_n = 0
            done_chunks += 1
        emb = Embeddings(vocab.itos, model.W_in.cpu().numpy())
        acc = evaluate_analogies(emb, restrict=30_000)
        el = time.time() - t0
        history["epoch_eval"].append(
            {"epoch": epoch + 1, "examples": n_examples, "elapsed_s": el, **acc.as_dict()}
        )
        last = history["loss"][-1] if history["loss"] else float("nan")
        print(
            f"epoch {epoch + 1}/{args.epochs}  examples {n_examples:,}  loss/ex {last:.4f}  "
            f"analogy sem {acc.semantic:.3f} syn {acc.syntactic:.3f} all {acc.overall:.3f}  [{el:.0f}s]",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    emb = Embeddings(vocab.itos, model.W_in.cpu().numpy())
    emb.save(args.out / "vectors.npz")
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    meta["train_seconds"] = time.time() - t0
    meta["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0
    (args.out / "history.json").write_text(json.dumps({"config": meta, **history}, indent=1))
    print(f"saved {args.out}  ({meta['train_seconds']:.0f}s, peak GPU {meta['peak_gpu_mb']:.0f} MB)")
    return emb


def main(argv: list[str] | None = None) -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    train(parse_args(argv))


if __name__ == "__main__":
    main()
