"""Vectorised training-example generation for word2vec (runs on CPU or GPU).

One *epoch* of examples is produced in a handful of tensor ops:

1. frequent-word sub-sampling (a Bernoulli mask over the corpus),
2. a *dynamic window* ``b_i ~ U{1..window}`` per centre position (word2vec.c
   samples a reduced window per word, which down-weights distant contexts),
3. (centre, context) pairs for skip-gram, or a padded context matrix for CBOW,
4. a global shuffle.

Negative samples are drawn from the unigram^0.75 distribution by inverse-CDF
sampling (``searchsorted`` on the cumulative distribution), which is exact,
unlike the fixed-size unigram table of word2vec.c.
"""

from __future__ import annotations

import torch


def subsample(ids: torch.Tensor, keep_prob: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Drop each token independently with probability ``1 - keep_prob[token]``."""
    u = torch.rand(ids.shape[0], device=ids.device, generator=generator)
    return ids[u < keep_prob[ids.long()]]


def dynamic_windows(n: int, window: int, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
    """Effective window size for every centre position, uniform on {1, ..., window}."""
    return torch.randint(1, window + 1, (n,), device=device, generator=generator)


def skipgram_pairs(
    ids: torch.Tensor, window: int, generator: torch.Generator | None = None, shuffle: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """All (centre, context) pairs of one epoch with a dynamic window.

    For centre ``i`` with effective window ``b_i`` the contexts are
    ``ids[i-b_i .. i+b_i]`` minus ``ids[i]``.
    """
    n = ids.shape[0]
    b = dynamic_windows(n, window, ids.device, generator)
    centres, contexts = [], []
    for d in range(1, window + 1):
        if d >= n:
            break
        right = b[:-d] >= d  # centre i, context i + d
        centres.append(ids[:-d][right])
        contexts.append(ids[d:][right])
        left = b[d:] >= d  # centre i, context i - d
        centres.append(ids[d:][left])
        contexts.append(ids[:-d][left])
    c = torch.cat(centres)
    o = torch.cat(contexts)
    if shuffle:
        perm = torch.randperm(c.shape[0], device=ids.device, generator=generator)
        c, o = c[perm], o[perm]
    return c, o


def cbow_contexts(
    ids: torch.Tensor, window: int, generator: torch.Generator | None = None, shuffle: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centre words and a ``(n, 2*window)`` context matrix padded with ``-1``.

    Column layout is ``[-window, ..., -1, +1, ..., +window]``.
    """
    n = ids.shape[0]
    b = dynamic_windows(n, window, ids.device, generator)
    ctx = torch.full((n, 2 * window), -1, dtype=ids.dtype, device=ids.device)
    pos = torch.arange(n, device=ids.device)
    offsets = list(range(-window, 0)) + list(range(1, window + 1))
    for col, d in enumerate(offsets):
        j = pos + d
        ok = (j >= 0) & (j < n) & (b >= abs(d))
        ctx[ok, col] = ids[j[ok]]
    has_ctx = (ctx >= 0).any(dim=1)
    centres, ctx = ids[has_ctx], ctx[has_ctx]
    if shuffle:
        perm = torch.randperm(centres.shape[0], device=ids.device, generator=generator)
        centres, ctx = centres[perm], ctx[perm]
    return centres, ctx


class NegativeSampler:
    """Draw word ids i.i.d. from a fixed categorical distribution via inverse CDF."""

    def __init__(self, probs: torch.Tensor, device: torch.device | str = "cpu"):
        p = torch.as_tensor(probs, dtype=torch.float64)
        cdf = torch.cumsum(p / p.sum(), 0)
        cdf[-1] = 1.0
        self.cdf = cdf.to(device=device, dtype=torch.float64)
        self.vocab_size = p.shape[0]

    def __call__(self, shape: tuple[int, ...], generator: torch.Generator | None = None) -> torch.Tensor:
        u = torch.rand(shape, device=self.cdf.device, dtype=torch.float64, generator=generator)
        return torch.searchsorted(self.cdf, u).clamp_(max=self.vocab_size - 1)
