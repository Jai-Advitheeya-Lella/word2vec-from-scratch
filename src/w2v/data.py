"""Corpus download, tokenization and vocabulary construction.

Everything here is plain Python / NumPy: a regex tokenizer, a frequency-sorted
vocabulary with a ``min_count`` cut-off, the word2vec sub-sampling keep
probabilities and the unigram^0.75 noise distribution used for negative
sampling.
"""

from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case and split ``text`` into alphabetic / numeric tokens.

    text8 is already lower-cased and whitespace separated, so on text8 this is
    equivalent to ``text.split()``; on raw text (e.g. WikiText) it strips
    punctuation and markup.
    """
    return _TOKEN_RE.findall(text.lower())


def iter_tokens(text: str) -> Iterator[str]:
    """Streaming version of :func:`tokenize` (avoids materialising a list of tokens)."""
    for m in _TOKEN_RE.finditer(text.lower()):
        yield m.group()


def download_text8(data_dir: Path = DATA_DIR) -> Path:
    """Download and unzip text8 (first 1e8 bytes of a cleaned 2006 Wikipedia dump)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "text8"
    if path.exists():
        return path
    zpath = data_dir / "text8.zip"
    if not zpath.exists():
        print(f"downloading {TEXT8_URL} ...")
        urllib.request.urlretrieve(TEXT8_URL, zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extract("text8", data_dir)
    return path


def load_text8_tokens(data_dir: Path = DATA_DIR, max_tokens: int | None = None) -> list[str]:
    """Return text8 as a list of tokens (17,005,207 tokens in total)."""
    text = download_text8(data_dir).read_text()
    tokens = tokenize(text)
    return tokens[:max_tokens] if max_tokens else tokens


@dataclass
class Vocab:
    """Frequency-sorted vocabulary.

    Index 0 is the most frequent word. Ties are broken alphabetically so the
    ordering is deterministic.
    """

    itos: list[str]
    counts: np.ndarray  # int64, aligned with itos
    stoi: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = np.asarray(self.counts, dtype=np.int64)

    @classmethod
    def build(cls, tokens: Iterable[str], min_count: int = 5, max_size: int | None = None) -> "Vocab":
        counter = Counter(tokens)
        items = sorted(((w, c) for w, c in counter.items() if c >= min_count), key=lambda x: (-x[1], x[0]))
        if max_size is not None:
            items = items[:max_size]
        return cls([w for w, _ in items], np.array([c for _, c in items], dtype=np.int64))

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, word: str) -> bool:
        return word in self.stoi

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def encode(self, tokens: Iterable[str]) -> np.ndarray:
        """Map tokens to ids, dropping out-of-vocabulary tokens (as word2vec.c does)."""
        stoi = self.stoi
        return np.fromiter((stoi[t] for t in tokens if t in stoi), dtype=np.int32)

    def keep_probs(self, sample: float) -> np.ndarray:
        """Probability of *keeping* each word under frequent-word sub-sampling.

        Mikolov et al. (2013b) discard word ``w`` with probability
        ``1 - sqrt(t / f(w))``. The released word2vec.c code (and gensim) uses
        the slightly smoother ``(sqrt(f / t) + 1) * t / f``; we follow the code
        so results are comparable with gensim. ``f`` is the relative frequency.
        """
        if sample <= 0:
            return np.ones(len(self), dtype=np.float32)
        f = self.counts / self.total
        p = (np.sqrt(f / sample) + 1.0) * sample / f
        return np.minimum(p, 1.0).astype(np.float32)

    def noise_dist(self, power: float = 0.75) -> np.ndarray:
        """Unigram distribution raised to ``power`` and renormalised: P_n(w) ∝ U(w)^{3/4}."""
        w = self.counts.astype(np.float64) ** power
        return (w / w.sum()).astype(np.float64)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"itos": self.itos, "counts": self.counts.tolist()}))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        d = json.loads(path.read_text())
        return cls(d["itos"], np.array(d["counts"], dtype=np.int64))


def chunk_ids(ids: np.ndarray, chunk: int) -> Iterator[np.ndarray]:
    """Yield consecutive chunks of an id array."""
    for s in range(0, len(ids), chunk):
        yield ids[s : s + chunk]
