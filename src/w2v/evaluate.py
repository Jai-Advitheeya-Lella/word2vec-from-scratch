"""Intrinsic evaluation: Google word analogies and word-similarity benchmarks.

Analogy protocol (Mikolov et al. 2013a, as implemented in word2vec.c's
``compute-accuracy`` and gensim): unit-normalise all vectors, keep only the
``restrict`` most frequent words as candidates (30k by convention), skip any
question that mentions a word outside that set, and answer ``a:b :: c:?`` with

* **3CosAdd**: :math:`\\arg\\max_{d} \\cos(d, b - a + c)`, and
* **3CosMul** (Levy & Goldberg 2014): :math:`\\arg\\max_d \\frac{\\cos'(d,b)\\cos'(d,c)}{\\cos'(d,a)+\\epsilon}`
  with :math:`\\cos' = (\\cos + 1)/2`,

excluding ``a``, ``b`` and ``c`` from the candidates.

Similarity protocol: Spearman's :math:`\\rho` between cosine similarity and
human ratings over the pairs whose two words are in the vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .embeddings import Embeddings

SEMANTIC_SECTIONS = {"capital-common-countries", "capital-world", "currency", "city-in-state", "family"}


def _gensim_test_file(name: str) -> Path:
    from gensim.test.utils import datapath  # gensim ships the canonical copies

    return Path(datapath(name))


def analogy_path() -> Path:
    return _gensim_test_file("questions-words.txt")


def wordsim_path() -> Path:
    return _gensim_test_file("wordsim353.tsv")


def simlex_path() -> Path:
    return _gensim_test_file("simlex999.txt")


def load_analogies(path: str | Path) -> list[tuple[str, list[tuple[str, str, str, str]]]]:
    """Parse ``questions-words.txt`` into ``[(section, [(a, b, c, d), ...]), ...]`` (lower-cased)."""
    sections: list[tuple[str, list[tuple[str, str, str, str]]]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            sections.append((line[1:].strip(), []))
            continue
        a, b, c, d = line.lower().split()
        sections[-1][1].append((a, b, c, d))
    return sections


def load_word_pairs(path: str | Path) -> list[tuple[str, str, float]]:
    """Parse a tab-separated ``word1 word2 score`` file, skipping ``#`` comments."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        a, b, s = line.split("\t")[:3]
        out.append((a.lower(), b.lower(), float(s)))
    return out


@dataclass
class AnalogyResult:
    method: str
    correct: dict[str, int] = field(default_factory=dict)
    total: dict[str, int] = field(default_factory=dict)
    n_questions: int = 0

    def acc(self, sections: set[str] | None = None) -> float:
        keys = [s for s in self.total if sections is None or s in sections]
        t = sum(self.total[s] for s in keys)
        return sum(self.correct[s] for s in keys) / t if t else float("nan")

    @property
    def semantic(self) -> float:
        return self.acc(SEMANTIC_SECTIONS)

    @property
    def syntactic(self) -> float:
        return self.acc({s for s in self.total if s not in SEMANTIC_SECTIONS})

    @property
    def overall(self) -> float:
        return self.acc()

    @property
    def coverage(self) -> float:
        return sum(self.total.values()) / max(self.n_questions, 1)

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "syntactic": self.syntactic,
            "overall": self.overall,
            "coverage": self.coverage,
        }


@torch.no_grad()
def evaluate_analogies(
    emb: Embeddings,
    path: str | Path | None = None,
    restrict: int = 30_000,
    method: str = "add",
    device: str | torch.device | None = None,
    batch: int = 512,
    only_words: set[str] | None = None,
) -> AnalogyResult:
    """Accuracy on the Google analogy set with 3CosAdd (``method="add"``) or 3CosMul (``"mul"``).

    ``only_words`` additionally drops every question that mentions a word outside the given set;
    this evaluates models with different vocabularies on a common question subset.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sections = load_analogies(path or analogy_path())
    words = emb.words[:restrict]
    idx = {w: i for i, w in enumerate(words)}
    E = torch.from_numpy(emb.unit[:restrict]).to(device)
    res = AnalogyResult(method=method, n_questions=sum(len(q) for _, q in sections))
    for name, qs in sections:
        rows = [
            (idx[a], idx[b], idx[c], idx[d])
            for a, b, c, d in qs
            if a in idx and b in idx and c in idx and d in idx
            and (only_words is None or {a, b, c, d} <= only_words)
        ]
        res.total[name] = len(rows)
        res.correct[name] = 0
        if not rows:
            continue
        q = torch.tensor(rows, device=device)
        for s in range(0, len(rows), batch):
            qb = q[s : s + batch]
            a, b, c, d = qb.unbind(1)
            if method == "add":
                scores = (E[b] - E[a] + E[c]) @ E.T
            elif method == "mul":
                ca, cb, cc = ((E[x] @ E.T + 1) / 2 for x in (a, b, c))
                scores = cb * cc / (ca + 1e-3)
            else:
                raise ValueError(method)
            r = torch.arange(len(qb), device=device)
            for x in (a, b, c):
                scores[r, x] = -float("inf")
            res.correct[name] += int((scores.argmax(1) == d).sum())
    return res


def evaluate_similarity(emb: Embeddings, path: str | Path) -> tuple[float, float]:
    """Spearman's rho and the fraction of pairs covered by the vocabulary."""
    pairs = load_word_pairs(path)
    got, gold = [], []
    for a, b, s in pairs:
        if a in emb and b in emb:
            got.append(emb.similarity(a, b))
            gold.append(s)
    rho = spearmanr(got, gold).statistic if len(got) > 2 else float("nan")
    return float(rho), len(got) / len(pairs)


def full_report(emb: Embeddings, restrict: int = 30_000, only_words: set[str] | None = None) -> dict[str, float]:
    """All intrinsic metrics in one flat dict (``only_words``: common question subset, see above)."""
    add = evaluate_analogies(emb, restrict=restrict, method="add", only_words=only_words)
    mul = evaluate_analogies(emb, restrict=restrict, method="mul", only_words=only_words)
    ws, ws_cov = evaluate_similarity(emb, wordsim_path())
    sl, sl_cov = evaluate_similarity(emb, simlex_path())
    return {
        "analogy_sem": add.semantic,
        "analogy_syn": add.syntactic,
        "analogy_all": add.overall,
        "analogy_all_3cosmul": mul.overall,
        "analogy_coverage": add.coverage,
        "wordsim353": ws,
        "wordsim353_coverage": ws_cov,
        "simlex999": sl,
        "simlex999_coverage": sl_cov,
    }
