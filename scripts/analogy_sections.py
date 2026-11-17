"""Per-section Google-analogy accuracy (3CosAdd, top-30k candidates, common question subset) -> results/analogy_sections.md."""

from __future__ import annotations

from pathlib import Path

from evaluate_all import load_pretrained_glove, reorder_like
from w2v.embeddings import Embeddings
from w2v.evaluate import evaluate_analogies

RUNS = {"skip-gram": "runs/sgns", "CBOW": "runs/cbow", "GloVe (10 ep)": "runs/glove10", "gensim skip-gram": "runs/gensim_sg"}


def main() -> None:
    embs = {k: Embeddings.load(Path(v) / "vectors.npz") for k, v in RUNS.items()}
    ref = embs["skip-gram"]
    embs = {k: reorder_like(v, ref) for k, v in embs.items()}
    embs["GloVe 6B (pretrained)"] = load_pretrained_glove()
    common = set(ref.words[:30_000])
    res = {k: evaluate_analogies(e, device="cpu", only_words=common) for k, e in embs.items()}
    secs = list(res["skip-gram"].total)
    lines = ["| section | n | " + " | ".join(res) + " |", "|---|---|" + "---|" * len(res)]
    for s in secs:
        n = res["skip-gram"].total[s]
        lines.append(f"| {s} | {n} | " + " | ".join(f"{100 * r.correct[s] / max(r.total[s], 1):.1f}" for r in res.values()) + " |")
    Path("results/analogy_sections.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
