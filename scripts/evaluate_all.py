"""Evaluate every trained embedding set (and the pretrained GloVe reference) on the intrinsic benchmarks.

    python scripts/evaluate_all.py --out results/metrics.json \
        sgns=runs/sgns cbow=runs/cbow glove=runs/glove gensim_sg=runs/gensim_sg gensim_cbow=runs/gensim_cbow

Every model is scored on (a) all Google analogy questions that fit its own vocabulary and
(b) the *common* subset of questions answerable by the text8 vocabulary (top 30k words), so that
models with different vocabularies (e.g. the 400k-word pretrained GloVe) are comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from w2v.embeddings import Embeddings
from w2v.evaluate import full_report


def load_pretrained_glove(name: str = "glove-wiki-gigaword-300") -> Embeddings:
    """Pretrained GloVe (Wikipedia 2014 + Gigaword 5, 6B tokens) from the ``fse`` mirror on the HF hub."""
    from gensim.models import KeyedVectors
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(f"fse/{name}", f"{name}.model")
    hf_hub_download(f"fse/{name}", f"{name}.model.vectors.npy")
    return Embeddings.from_gensim(KeyedVectors.load(path))


def reorder_like(emb: Embeddings, ref: Embeddings) -> Embeddings:
    """Re-index ``emb`` to ``ref``'s word order (frequency ties are broken differently by different trainers, which would
    change which words fall inside the top-30k candidate set)."""
    if set(emb.words) != set(ref.words) or emb.words == ref.words:
        return emb
    return Embeddings(ref.words, emb.vectors[[emb.index[w] for w in ref.words]])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="name=path/to/run_dir (containing vectors.npz)")
    p.add_argument("--common-from", default=None, help="name of the run whose top-30k vocabulary defines the common question subset")
    p.add_argument("--pretrained", action="store_true", help="also evaluate pretrained GloVe 6B-300d")
    p.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    args = p.parse_args()

    embs: dict[str, Embeddings] = {}
    for spec in args.runs:
        name, path = spec.split("=")
        embs[name] = Embeddings.load(Path(path) / "vectors.npz")
    common = None
    if args.common_from:
        ref = embs[args.common_from]
        embs = {k: reorder_like(v, ref) for k, v in embs.items()}
        common = set(ref.words[:30_000])
    if args.pretrained:
        embs["glove_pretrained_6B"] = load_pretrained_glove()
    out: dict[str, dict] = {}
    for name, emb in embs.items():
        rep = full_report(emb)
        rep["vocab"] = len(emb)
        rep["dim"] = emb.dim
        if common is not None:
            sub = full_report(emb, only_words=common)
            rep.update({f"common_{k}": v for k, v in sub.items() if k.startswith("analogy")})
        out[name] = rep
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
