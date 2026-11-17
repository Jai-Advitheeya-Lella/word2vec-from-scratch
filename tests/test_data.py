import numpy as np
import pytest

from w2v.data import Vocab, iter_tokens, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("The cat's hat, 1984!") == ["the", "cat's", "hat", "1984"]
    assert list(iter_tokens("a  b\nc")) == tokenize("a  b\nc") == ["a", "b", "c"]


def test_vocab_is_frequency_sorted_with_min_count():
    toks = "b a a c c c d".split()
    v = Vocab.build(toks, min_count=2)
    assert v.itos == ["c", "a"]
    assert v.counts.tolist() == [3, 2]
    assert v.encode(toks).tolist() == [1, 1, 0, 0, 0]  # OOV tokens dropped


def test_noise_distribution_is_unigram_to_three_quarters():
    v = Vocab(["x", "y", "z"], np.array([16, 81, 1]))
    p = v.noise_dist(0.75)
    expected = np.array([8.0, 27.0, 1.0]) / 36.0
    np.testing.assert_allclose(p, expected)


def test_keep_probs_match_gensim():
    """Our sub-sampling keep probabilities equal gensim's ``sample_int / 2**32``."""
    gensim = pytest.importorskip("gensim")
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(50)]
    zipf = 1.0 / np.arange(1, 51)
    corpus = rng.choice(words, size=20_000, p=zipf / zipf.sum()).tolist()
    sents = [corpus[i : i + 100] for i in range(0, len(corpus), 100)]
    sample = 1e-2
    m = gensim.models.Word2Vec(vector_size=4, min_count=1, sample=sample, workers=1)
    m.build_vocab(sents)
    v = Vocab.build(corpus, min_count=1)
    ours = v.keep_probs(sample)
    for w, p in zip(v.itos, ours):
        ref = m.wv.get_vecattr(w, "sample_int") / 2**32
        assert abs(ref - p) < 1e-6, (w, ref, p)
