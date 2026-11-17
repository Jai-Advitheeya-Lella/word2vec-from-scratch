"""Skip-gram hyper-parameter ablations on text8 (5 epochs, dim 200 so each run takes about a minute).

    python scripts/run_ablations.py --out results/ablations.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from w2v.evaluate import evaluate_analogies, evaluate_similarity, wordsim_path
from w2v.train import parse_args, train

BASE = ["--model", "skipgram", "--dim", "200", "--epochs", "5", "--window", "5", "--negative", "5", "--sample", "1e-4", "--lr", "0.025"]
GRID = {
    "negative": [("1", "1"), ("2", "2"), ("5", "5"), ("10", "10"), ("20", "20")],
    "sample": [("0", "off"), ("1e-3", "1e-3"), ("1e-4", "1e-4"), ("1e-5", "1e-5")],
    "window": [("2", "2"), ("5", "5"), ("10", "10")],
    "dim": [("50", "50"), ("100", "100"), ("200", "200"), ("300", "300")],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/ablations.json"))
    a = p.parse_args()
    results: dict[str, list[dict]] = {}
    for name, values in GRID.items():
        results[name] = []
        for val, label in values:
            argv = list(BASE)
            argv[argv.index(f"--{name}") + 1] = val
            t = time.time()
            emb = train(parse_args(argv + ["--out", f"runs/abl_{name}_{label}"]))
            acc = evaluate_analogies(emb)
            ws, _ = evaluate_similarity(emb, wordsim_path())
            results[name].append({"value": label, "semantic": acc.semantic, "syntactic": acc.syntactic, "overall": acc.overall,
                                  "wordsim353": ws, "seconds": time.time() - t})
            print(name, label, {k: round(v, 4) for k, v in results[name][-1].items() if k != "value"}, flush=True)
            a.out.parent.mkdir(exist_ok=True)
            a.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
