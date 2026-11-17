import numpy as np
import pytest

from w2v.bias import direct_bias, gender_direction, neutralize
from w2v.embeddings import Embeddings
from w2v.evaluate import SEMANTIC_SECTIONS, analogy_path, evaluate_analogies, evaluate_similarity, load_analogies, wordsim_path

gensim = pytest.importorskip("gensim")


def _random_embeddings(dim=6, seed=0):
    words = sorted({w for _, qs in load_analogies(analogy_path()) for q in qs for w in q})
    rng = np.random.default_rng(seed)
    return Embeddings(words, rng.standard_normal((len(words), dim)).astype(np.float32))


def _to_gensim(emb):
    kv = gensim.models.KeyedVectors(emb.dim)
    kv.add_vectors(emb.words, emb.vectors)
    return kv


def test_analogy_accuracy_matches_gensim_exactly():
    emb = _random_embeddings()
    restrict = 600  # exercises the restricted-vocabulary / skipped-question logic
    ours = evaluate_analogies(emb, restrict=restrict, device="cpu")
    _, sections = _to_gensim(emb).evaluate_word_analogies(analogy_path(), restrict_vocab=restrict, case_insensitive=True)
    ref = {s["section"]: (len(s["correct"]), len(s["correct"]) + len(s["incorrect"])) for s in sections}
    assert sum(ours.correct.values()) > 0
    for name in ours.total:
        if ours.total[name]:
            assert (ours.correct[name], ours.total[name]) == ref[name], name
    sem = [s for s in ref if s in SEMANTIC_SECTIONS]
    assert abs(ours.semantic - sum(ref[s][0] for s in sem) / sum(ref[s][1] for s in sem)) < 1e-12


def test_planted_analogies_are_solved():
    # vectors built so that b - a + c == d exactly for capital/country pairs
    rng = np.random.default_rng(0)
    countries = ["france", "germany", "italy", "spain", "japan"]
    capitals = ["paris", "berlin", "rome", "madrid", "tokyo"]
    offset = rng.standard_normal(16)
    base = {c: rng.standard_normal(16) * 3 for c in countries}
    vecs = {c: base[c] for c in countries} | {cap: base[c] + offset for c, cap in zip(countries, capitals)}
    emb = Embeddings(list(vecs), np.stack(list(vecs.values())).astype(np.float32))
    q = [(a, b, c, d) for a, b in zip(countries, capitals) for c, d in zip(countries, capitals) if a != c]
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(": capital-world\n" + "\n".join(" ".join(x) for x in q))
    # unit-normalisation perturbs exactness, so require most to be right
    res = evaluate_analogies(emb, path=f.name, device="cpu")
    assert res.overall > 0.9 and res.coverage == 1.0


def test_similarity_matches_gensim():
    emb = _random_embeddings(dim=8, seed=1)
    words = set(emb.words)
    rng = np.random.default_rng(1)
    from w2v.evaluate import load_word_pairs

    extra = sorted({w for a, b, _ in load_word_pairs(wordsim_path()) for w in (a, b)} - words)
    emb = Embeddings(emb.words + extra, np.vstack([emb.vectors, rng.standard_normal((len(extra), 8))]).astype(np.float32))
    rho, cov = evaluate_similarity(emb, wordsim_path())
    _, (ref_rho, _), oov = _to_gensim(emb).evaluate_word_pairs(wordsim_path(), case_insensitive=True)
    assert abs(rho - ref_rho) < 1e-6
    assert abs(cov - (1 - oov / 100)) < 1e-9


def test_gender_direction_recovers_planted_axis():
    rng = np.random.default_rng(0)
    g_true = rng.standard_normal(20)
    g_true /= np.linalg.norm(g_true)
    pairs = [("she", "he"), ("her", "his"), ("woman", "man"), ("girl", "boy")]
    words, vecs = [], []
    for f, m in pairs:
        base = rng.standard_normal(20)
        words += [f, m]
        vecs += [base + 1.5 * g_true, base - 1.5 * g_true]
    words += ["nurse", "engineer"]
    vecs += [rng.standard_normal(20) + 0.8 * g_true, rng.standard_normal(20) - 0.8 * g_true]
    emb = Embeddings(words, np.stack(vecs).astype(np.float32))
    g, spectrum = gender_direction(emb, pairs)
    assert abs(g @ g_true) > 0.95 and g @ g_true > 0
    assert spectrum[0] > 0.5
    assert direct_bias(emb, ["nurse", "engineer"], g) > 0.1
    assert direct_bias(neutralize(emb, ["nurse", "engineer"], g), ["nurse", "engineer"], g) < 1e-6


def test_only_words_restricts_the_question_set():
    emb = _random_embeddings()
    full = evaluate_analogies(emb, device="cpu")
    allowed = set(emb.words[:400])
    sub = evaluate_analogies(emb, device="cpu", only_words=allowed)
    assert 0 < sum(sub.total.values()) < sum(full.total.values())
    ref = sum(
        1 for _, qs in load_analogies(analogy_path()) for q in qs if set(q) <= allowed
    )
    assert sum(sub.total.values()) == ref
