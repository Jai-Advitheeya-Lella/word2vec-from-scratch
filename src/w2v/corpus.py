"""Cached corpus preparation shared by all trainers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import itertools

from .data import DATA_DIR, Vocab, download_text8, iter_tokens


def prepare_text8(min_count: int = 5, data_dir: Path = DATA_DIR, max_tokens: int | None = None) -> tuple[Vocab, np.ndarray]:
    """Tokenise text8, build the vocabulary and encode the corpus (cached in ``data/``)."""
    tag = f"text8_min{min_count}" + (f"_{max_tokens}" if max_tokens else "")
    vpath, ipath = data_dir / f"{tag}.vocab.json", data_dir / f"{tag}.ids.npy"
    if vpath.exists() and ipath.exists():
        return Vocab.load(vpath), np.load(ipath)
    text = download_text8(data_dir).read_text()
    # two streaming passes keep peak memory low (no 17M-element list of strings)
    vocab = Vocab.build(itertools.islice(iter_tokens(text), max_tokens), min_count=min_count)
    ids = vocab.encode(itertools.islice(iter_tokens(text), max_tokens))
    del text
    vocab.save(vpath)
    np.save(ipath, ids)
    return vocab, ids
