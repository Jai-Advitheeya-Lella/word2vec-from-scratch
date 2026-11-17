from collections import defaultdict

import pytest
import torch

from w2v.glove import GloVe, cooccurrence, glove_loss_and_grads


def _brute_force_cooc(seq, window):
    X = defaultdict(float)
    for i, a in enumerate(seq):
        for d in range(1, window + 1):
            if i + d < len(seq):
                b = seq[i + d]
                X[(a, b)] += 1.0 / d
                X[(b, a)] += 1.0 / d
    return X


def test_cooccurrence_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    V = 7
    seq = torch.randint(0, V, (300,), generator=g)
    # small chunk size exercises the incremental merge path
    ref = _brute_force_cooc(seq.tolist(), 4)
    for parts in (1, 3):
        r, c, x = cooccurrence(seq, V, window=4, chunk=37, partitions=parts)
        got = {(int(a), int(b)): float(v) for a, b, v in zip(r, c, x)}
        assert len(got) == r.numel()  # no duplicate keys
        assert set(got) == set(ref)
        for k in ref:
            assert abs(got[k] - ref[k]) < 1e-4, k
        for (a, b), v in got.items():  # symmetric
            assert abs(got[(b, a)] - v) < 1e-5


def test_glove_gradients_match_autograd():
    torch.manual_seed(0)
    B, D = 9, 4
    w, c = torch.randn(B, D, dtype=torch.float64), torch.randn(B, D, dtype=torch.float64)
    bw, bc = torch.randn(B, dtype=torch.float64), torch.randn(B, dtype=torch.float64)
    x = torch.rand(B, dtype=torch.float64) * 200 + 0.1  # straddles x_max
    params = [t.clone().requires_grad_(True) for t in (w, c, bw, bc)]
    diff = (params[0] * params[1]).sum(-1) + params[2] + params[3] - torch.log(x)
    f = torch.clamp(x / 100.0, max=1.0) ** 0.75
    loss = 0.5 * (f * diff**2).sum()
    loss.backward()
    g = glove_loss_and_grads(w, c, bw, bc, x, 100.0, 0.75)
    torch.testing.assert_close(g.loss, loss.detach())
    for ours, p in zip((g.grad_w, g.grad_c, g.grad_bw, g.grad_bc), params):
        torch.testing.assert_close(ours, p.grad)


def test_glove_training_reduces_loss():
    g = torch.Generator().manual_seed(0)
    V = 30
    seq = torch.randint(0, V, (3000,), generator=g)
    r, c, x = cooccurrence(seq, V, window=3)
    m = GloVe(V, 8, seed=0)
    losses = []
    for _ in range(30):
        perm = torch.randperm(r.numel(), generator=g)
        tot = 0.0
        for s in range(0, r.numel(), 128):
            b = perm[s : s + 128]
            tot += float(m.step(r[b].long(), c[b].long(), x[b], lr=0.05, x_max=10.0, alpha=0.75))
        losses.append(tot)
    assert losses[-1] < 0.5 * losses[0]


@pytest.mark.parametrize("average", [False, True])
def test_sparse_adagrad_matches_reference_with_duplicates(average):
    """The sort-free scatter AdaGrad equals a naive per-distinct-row AdaGrad (duplicate gradients summed or averaged)."""
    torch.manual_seed(0)
    V, D, B, lr = 6, 3, 20, 0.1
    m = GloVe(V, D, seed=0, average_duplicates=average)
    ref_p, ref_a = m.W.clone(), m.gW.clone()
    for _ in range(3):
        idx = torch.randint(0, V, (B,))
        grad = torch.randn(B, D)
        m._adagrad(m.W, m.gW, idx, grad, lr)
        for r in idx.unique():
            rows = grad[idx == r]
            g = lr * (rows.mean(0) if average else rows.sum(0))
            ref_p[r] -= g / ref_a[r].sqrt()
            ref_a[r] += g * g
    torch.testing.assert_close(m.W, ref_p)
    torch.testing.assert_close(m.gW, ref_a)
    assert float(m._buf[id(m.W)].abs().sum()) == 0.0  # scratch buffer left clean
