"""Skip-gram and CBOW with negative sampling, with hand-derived gradients.

For an input vector :math:`v` (the centre word for skip-gram, the mean of the
context vectors for CBOW), a positive output vector :math:`u_o` and :math:`K`
negative output vectors :math:`u_k \\sim P_n`, the per-example loss is

.. math::

    \\ell = -\\log\\sigma(u_o^\\top v) - \\sum_{k=1}^{K} \\log\\sigma(-u_k^\\top v).

Its gradients are

.. math::

    \\partial\\ell/\\partial v = (\\sigma(u_o^\\top v) - 1)\\,u_o + \\sum_k \\sigma(u_k^\\top v)\\,u_k,\\quad
    \\partial\\ell/\\partial u_o = (\\sigma(u_o^\\top v) - 1)\\,v,\\quad
    \\partial\\ell/\\partial u_k = \\sigma(u_k^\\top v)\\,v.

We compute these in closed form (no autograd) and apply plain SGD with
``index_add_`` scatter updates — the minibatch analogue of the asynchronous
per-pair SGD of word2vec.c. ``tests/test_sgns.py`` checks the closed-form
gradients against ``torch.autograd``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class NSGrads:
    loss: torch.Tensor  # scalar, summed over the batch
    grad_v: torch.Tensor  # (B, D)
    grad_pos: torch.Tensor  # (B, D)
    grad_neg: torch.Tensor  # (B, K, D)


def ns_loss_and_grads(
    v: torch.Tensor, u_pos: torch.Tensor, u_neg: torch.Tensor, neg_mask: torch.Tensor | None = None
) -> NSGrads:
    """Negative-sampling loss and its closed-form gradients.

    Args:
        v: input vectors, shape ``(B, D)``.
        u_pos: output vectors of the true targets, ``(B, D)``.
        u_neg: output vectors of the negatives, ``(B, K, D)``.
        neg_mask: optional ``(B, K)`` 0/1 mask; masked negatives (e.g. a sampled
            negative that equals the true target) contribute nothing.
    """
    s_pos = (v * u_pos).sum(-1)  # (B,)
    s_neg = torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)  # (B, K)
    if neg_mask is None:
        neg_mask = torch.ones_like(s_neg)
    loss = -F.logsigmoid(s_pos).sum() - (F.logsigmoid(-s_neg) * neg_mask).sum()
    g_pos = torch.sigmoid(s_pos) - 1.0  # dl/ds_pos
    g_neg = torch.sigmoid(s_neg) * neg_mask  # dl/ds_neg
    grad_v = g_pos.unsqueeze(-1) * u_pos + torch.bmm(g_neg.unsqueeze(1), u_neg).squeeze(1)
    grad_pos = g_pos.unsqueeze(-1) * v
    grad_neg = g_neg.unsqueeze(-1) * v.unsqueeze(1)
    return NSGrads(loss, grad_v, grad_pos, grad_neg)


class Word2Vec:
    """Two embedding tables (``W_in`` = word vectors, ``W_out`` = context vectors).

    Initialisation follows word2vec.c: ``W_in ~ U(-0.5/D, 0.5/D)``, ``W_out = 0``.
    """

    def __init__(self, vocab_size: int, dim: int, device: torch.device | str = "cpu", seed: int = 0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.W_in = ((torch.rand(vocab_size, dim, generator=g) - 0.5) / dim).to(device)
        self.W_out = torch.zeros(vocab_size, dim, device=device)
        self.dim = dim

    @torch.no_grad()
    def skipgram_step(self, centres: torch.Tensor, contexts: torch.Tensor, negatives: torch.Tensor, lr: float) -> torch.Tensor:
        """One SGD step on a batch of (centre, context) pairs. Returns the summed loss (0-d tensor)."""
        v = self.W_in[centres]
        u_pos = self.W_out[contexts]
        u_neg = self.W_out[negatives]
        mask = (negatives != contexts.unsqueeze(1)).to(v.dtype)
        g = ns_loss_and_grads(v, u_pos, u_neg, mask)
        self.W_in.index_add_(0, centres, g.grad_v, alpha=-lr)
        self.W_out.index_add_(0, contexts, g.grad_pos, alpha=-lr)
        self.W_out.index_add_(0, negatives.reshape(-1), g.grad_neg.reshape(-1, self.dim), alpha=-lr)
        return g.loss

    @torch.no_grad()
    def cbow_step(self, centres: torch.Tensor, ctx: torch.Tensor, negatives: torch.Tensor, lr: float) -> torch.Tensor:
        """One SGD step for CBOW. ``ctx`` is ``(B, 2W)`` padded with -1.

        The hidden vector is the mean of the context vectors. As in word2vec.c
        (``cbow_mean=1``), the gradient w.r.t. the hidden vector is added to
        *each* context word without the 1/|C| factor of the exact gradient.
        """
        m = (ctx >= 0).to(self.W_in.dtype)  # (B, 2W)
        safe = ctx.clamp(min=0)
        cnt = m.sum(1, keepdim=True)
        h = (self.W_in[safe] * m.unsqueeze(-1)).sum(1) / cnt
        u_pos = self.W_out[centres]
        u_neg = self.W_out[negatives]
        mask = (negatives != centres.unsqueeze(1)).to(h.dtype)
        g = ns_loss_and_grads(h, u_pos, u_neg, mask)
        upd = (g.grad_v.unsqueeze(1) * m.unsqueeze(-1)).reshape(-1, self.dim)
        self.W_in.index_add_(0, safe.reshape(-1), upd, alpha=-lr)
        self.W_out.index_add_(0, centres, g.grad_pos, alpha=-lr)
        self.W_out.index_add_(0, negatives.reshape(-1), g.grad_neg.reshape(-1, self.dim), alpha=-lr)
        return g.loss
