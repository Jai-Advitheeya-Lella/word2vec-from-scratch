"""A minimal keyed-vectors container: save/load, normalisation, neighbours, analogies."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class Embeddings:
    """Word vectors indexed by a frequency-sorted word list."""

    def __init__(self, words: Sequence[str], vectors: np.ndarray):
        if len(words) != vectors.shape[0]:
            raise ValueError("words and vectors disagree in length")
        self.words = list(words)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.index = {w: i for i, w in enumerate(self.words)}
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.unit = self.vectors / np.maximum(norms, 1e-12)

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, w: str) -> bool:
        return w in self.index

    def __getitem__(self, w: str) -> np.ndarray:
        return self.vectors[self.index[w]]

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    # ---------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, vectors=self.vectors, words=np.array(self.words, dtype=object))

    @classmethod
    def load(cls, path: str | Path) -> "Embeddings":
        d = np.load(path, allow_pickle=True)
        return cls(list(d["words"]), d["vectors"])

    @classmethod
    def from_gensim(cls, kv) -> "Embeddings":  # noqa: ANN001 - gensim KeyedVectors
        return cls(list(kv.index_to_key), kv.vectors)

    # ---------------------------------------------------------- queries
    def most_similar(
        self, positive: Sequence[str], negative: Sequence[str] = (), topn: int = 10, restrict: int | None = None
    ) -> list[tuple[str, float]]:
        """Nearest neighbours of ``sum(positive) - sum(negative)`` (unit vectors), excluding the query words."""
        m = self.unit if restrict is None else self.unit[:restrict]
        q = np.zeros(self.dim, dtype=np.float32)
        for w in positive:
            q += self.unit[self.index[w]]
        for w in negative:
            q -= self.unit[self.index[w]]
        q /= max(np.linalg.norm(q), 1e-12)
        sims = m @ q
        exclude = {self.index[w] for w in (*positive, *negative)}
        out: list[tuple[str, float]] = []
        for i in np.argsort(-sims):
            if int(i) in exclude:
                continue
            out.append((self.words[i], float(sims[i])))
            if len(out) == topn:
                break
        return out

    def similarity(self, a: str, b: str) -> float:
        return float(self.unit[self.index[a]] @ self.unit[self.index[b]])
