"""Baseline: gensim Word2Vec on exactly the same text8 token stream and hyper-parameters.

Example::

    python scripts/train_gensim.py --sg 1 --epochs 5 --out runs/gensim_sg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gensim.models import Word2Vec as GensimW2V

from w2v.corpus import prepare_text8
from w2v.embeddings import Embeddings
from w2v.evaluate import evaluate_analogies


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sg", type=int, default=1, help="1 = skip-gram, 0 = CBOW")
    p.add_argument("--dim", type=int, default=200)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--negative", type=int, default=5)
    p.add_argument("--sample", type=float, default=1e-4)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.025)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/gensim_sg"))
    args = p.parse_args()

    vocab, ids = prepare_text8(args.min_count)
    words = vocab.itos
    # same token stream as our trainer, in 10k-token "sentences" like gensim's Text8Corpus
    sents = [[words[i] for i in ids[s : s + 10_000]] for s in range(0, len(ids), 10_000)]
    t0 = time.time()
    model = GensimW2V(
        sentences=sents,
        vector_size=args.dim,
        window=args.window,
        negative=args.negative,
        hs=0,
        sg=args.sg,
        sample=args.sample,
        min_count=args.min_count,
        epochs=args.epochs,
        alpha=args.lr,
        min_alpha=args.lr * 1e-4,
        workers=args.workers,
        seed=args.seed,
        cbow_mean=1,
    )
    secs = time.time() - t0
    emb = Embeddings.from_gensim(model.wv)
    acc = evaluate_analogies(emb)
    print(f"gensim sg={args.sg}: {secs:.0f}s  analogy sem {acc.semantic:.3f} syn {acc.syntactic:.3f} all {acc.overall:.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    emb.save(args.out / "vectors.npz")
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.out / "history.json").write_text(json.dumps({"config": {**cfg, "train_seconds": secs}}, indent=1))


if __name__ == "__main__":
    main()
