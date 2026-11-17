import torch

from w2v.sampling import NegativeSampler, cbow_contexts, skipgram_pairs, subsample


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_window_one_gives_all_adjacent_pairs():
    ids = torch.arange(6)
    c, o = skipgram_pairs(ids, window=1, generator=_gen(), shuffle=False)
    got = sorted(zip(c.tolist(), o.tolist()))
    want = sorted([(i, i + 1) for i in range(5)] + [(i + 1, i) for i in range(5)])
    assert got == want


def test_skipgram_pairs_respect_dynamic_window():
    n, W = 2000, 5
    ids = torch.arange(n)  # ids == positions, so |c - o| is the distance
    c, o = skipgram_pairs(ids, window=W, generator=_gen())
    d = (c - o).abs()
    assert d.min() >= 1 and d.max() <= W
    # E[#pairs] = n * E[2 b] = n * (W + 1) with b ~ U{1..W} (minus boundary effects)
    assert abs(c.numel() / (n * (W + 1)) - 1) < 0.05
    # dynamic window: nearer contexts are sampled more often (count(d) ∝ W - d + 1)
    counts = torch.bincount(d, minlength=W + 1)[1:].float()
    ratio = counts / counts[0]
    expected = torch.tensor([(W - k + 1) / W for k in range(1, W + 1)])
    assert torch.allclose(ratio, expected, atol=0.06)


def test_cbow_contexts_are_consistent_with_positions():
    n, W = 500, 3
    ids = torch.arange(n)
    centres, ctx = cbow_contexts(ids, window=W, generator=_gen())
    assert ctx.shape == (centres.numel(), 2 * W)
    offs = torch.tensor(list(range(-W, 0)) + list(range(1, W + 1)))
    valid = ctx >= 0
    assert torch.equal((ctx - centres[:, None])[valid], offs.expand_as(ctx)[valid])
    # symmetric: if offset +d is present, so is -d whenever it is inside the corpus
    for k in range(W):
        left, right = valid[:, W - 1 - k], valid[:, W + k]
        inner = (centres > W) & (centres < n - W - 1)
        assert torch.equal(left[inner], right[inner])


def test_negative_sampler_matches_distribution():
    p = torch.tensor([0.5, 0.25, 0.125, 0.125])
    s = NegativeSampler(p)
    draws = s((200_000,), generator=_gen())
    freq = torch.bincount(draws, minlength=4).double() / draws.numel()
    assert torch.allclose(freq, p.double(), atol=5e-3)


def test_subsample_keep_rate():
    ids = torch.zeros(100_000, dtype=torch.long)
    keep = torch.tensor([0.3])
    kept = subsample(ids, keep, generator=_gen())
    assert abs(kept.numel() / ids.numel() - 0.3) < 0.01
