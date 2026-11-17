import torch
import torch.nn.functional as F

from w2v.sampling import NegativeSampler, skipgram_pairs
from w2v.sgns import Word2Vec, ns_loss_and_grads


def _autograd_ns(v, u_pos, u_neg, mask):
    v, u_pos, u_neg = (t.clone().requires_grad_(True) for t in (v, u_pos, u_neg))
    s_pos = (v * u_pos).sum(-1)
    s_neg = torch.einsum("bkd,bd->bk", u_neg, v)
    loss = -F.logsigmoid(s_pos).sum() - (F.logsigmoid(-s_neg) * mask).sum()
    loss.backward()
    return loss.detach(), v.grad, u_pos.grad, u_neg.grad


def test_closed_form_gradients_match_autograd():
    torch.manual_seed(0)
    B, K, D = 7, 4, 5
    v, u_pos, u_neg = torch.randn(B, D, dtype=torch.float64), torch.randn(B, D, dtype=torch.float64), torch.randn(B, K, D, dtype=torch.float64)
    mask = (torch.rand(B, K) > 0.2).double()
    g = ns_loss_and_grads(v, u_pos, u_neg, mask)
    loss, gv, gp, gn = _autograd_ns(v, u_pos, u_neg, mask)
    torch.testing.assert_close(g.loss, loss)
    torch.testing.assert_close(g.grad_v, gv)
    torch.testing.assert_close(g.grad_pos, gp)
    torch.testing.assert_close(g.grad_neg, gn)


def test_gradcheck_finite_differences():
    torch.manual_seed(1)
    v = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    u_pos = torch.randn(3, 4, dtype=torch.float64)
    u_neg = torch.randn(3, 2, 4, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: ns_loss_and_grads(x, u_pos, u_neg).loss, (v,))


def test_skipgram_step_equals_autograd_sgd_with_duplicates():
    """The scatter (index_add_) update equals SGD on the summed loss, even with repeated rows."""
    torch.manual_seed(2)
    V, D, lr = 6, 3, 0.1
    m = Word2Vec(V, D, seed=0)
    m.W_out = torch.randn(V, D) * 0.1
    c = torch.tensor([0, 1, 0, 2])
    o = torch.tensor([1, 2, 3, 0])
    neg = torch.tensor([[4, 5], [5, 5], [3, 4], [1, 0]])
    W_in = m.W_in.clone().requires_grad_(True)
    W_out = m.W_out.clone().requires_grad_(True)
    mask = (neg != o[:, None]).float()
    s_pos = (W_in[c] * W_out[o]).sum(-1)
    s_neg = torch.einsum("bkd,bd->bk", W_out[neg], W_in[c])
    loss = -F.logsigmoid(s_pos).sum() - (F.logsigmoid(-s_neg) * mask).sum()
    loss.backward()
    m.skipgram_step(c, o, neg, lr)
    torch.testing.assert_close(m.W_in, (W_in - lr * W_in.grad).detach())
    torch.testing.assert_close(m.W_out, (W_out - lr * W_out.grad).detach())


def test_cbow_step_applies_hidden_gradient_to_each_context_word():
    torch.manual_seed(3)
    V, D, lr = 8, 4, 0.05
    m = Word2Vec(V, D, seed=0)
    m.W_out = torch.randn(V, D) * 0.1
    centres = torch.tensor([0, 1])
    ctx = torch.tensor([[2, 3, -1], [4, -1, -1]])
    neg = torch.tensor([[5, 6], [7, 5]])
    h = torch.stack([m.W_in[[2, 3]].mean(0), m.W_in[4]])
    g = ns_loss_and_grads(h, m.W_out[centres], m.W_out[neg], (neg != centres[:, None]).float())
    before = m.W_in.clone()
    m.cbow_step(centres, ctx, neg, lr)
    torch.testing.assert_close(m.W_in[2], before[2] - lr * g.grad_v[0])
    torch.testing.assert_close(m.W_in[3], before[3] - lr * g.grad_v[0])
    torch.testing.assert_close(m.W_in[4], before[4] - lr * g.grad_v[1])
    torch.testing.assert_close(m.W_in[0], before[0])  # padding slot (-1) must not touch row 0


def test_tiny_corpus_learns_topic_clusters():
    """Words that only co-occur within their own group end up closer to each other."""
    g = torch.Generator().manual_seed(0)
    groups = [torch.arange(0, 10), torch.arange(10, 20)]
    seq = torch.cat([groups[i % 2][torch.randint(0, 10, (50,), generator=g)] for i in range(200)])
    m = Word2Vec(20, 16, seed=0)
    sampler = NegativeSampler(torch.ones(20))
    for _ in range(5):
        c, o = skipgram_pairs(seq, 2, generator=g)
        for s in range(0, c.numel(), 256):
            m.skipgram_step(c[s : s + 256], o[s : s + 256], sampler((min(256, c.numel() - s), 5), g), 0.05)
    E = F.normalize(m.W_in, dim=1)
    S = E @ E.T
    within = (S[:10, :10].sum() - 10) / 90 + (S[10:, 10:].sum() - 10) / 90
    across = S[:10, 10:].mean() * 2
    assert within > across + 0.5
