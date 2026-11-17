"""Figures and tables for the README (written to ``assets/`` and ``results/``).

    python scripts/make_figures.py --sgns runs/sgns --cbow runs/cbow --glove runs/glove
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from w2v.bias import OCCUPATIONS, direct_bias, gender_direction, projections
from w2v.embeddings import Embeddings

ASSETS = Path("assets")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

CAPITALS = [("france", "paris"), ("germany", "berlin"), ("italy", "rome"), ("spain", "madrid"), ("japan", "tokyo"),
            ("russia", "moscow"), ("greece", "athens"), ("poland", "warsaw"), ("egypt", "cairo"), ("england", "london"),
            ("china", "beijing"), ("sweden", "stockholm")]
GENDER = [("man", "woman"), ("king", "queen"), ("uncle", "aunt"), ("brother", "sister"), ("father", "mother"),
          ("son", "daughter"), ("boy", "girl"), ("husband", "wife"), ("prince", "princess"), ("he", "she")]
PAST = [("go", "went"), ("take", "took"), ("make", "made"), ("come", "came"), ("see", "saw"), ("give", "gave"),
        ("find", "found"), ("write", "wrote"), ("run", "ran"), ("speak", "spoke"), ("begin", "began"), ("know", "knew")]
CATEGORIES = {
    "countries": "france germany italy spain japan china russia india brazil canada egypt greece poland sweden norway turkey".split(),
    "capitals": "paris berlin rome madrid tokyo beijing moscow delhi athens warsaw stockholm oslo cairo vienna london".split(),
    "animals": "dog cat horse cow sheep pig bird fish lion tiger wolf bear rabbit deer elephant".split(),
    "colors": "red blue green yellow black white orange purple brown pink gray".split(),
    "numbers": "one two three four five six seven eight nine ten twenty hundred thousand".split(),
    "family": "father mother brother sister son daughter uncle aunt husband wife grandfather".split(),
    "professions": "doctor nurse lawyer teacher scientist engineer soldier farmer priest artist writer".split(),
}


def offset_projection(P: np.ndarray, n_pairs: int) -> tuple[np.ndarray, float]:
    """2-D view of ``n_pairs`` (source, target) vector pairs stored as rows ``[s0, t0, s1, t1, ...]``.

    The x-axis is the unit mean offset ``mean(t_i - s_i)``; the y-axis is the leading principal direction of the points
    after removing their x-component. Parallel arrows in this view mean the offsets agree *in the full space*;
    the returned number is the mean cosine between each pair's offset and the mean offset in the original space.
    """
    src, tgt = P[0::2], P[1::2]
    off = tgt - src
    u = off.mean(0)
    u /= np.linalg.norm(u)
    cos = float(np.mean((off / np.linalg.norm(off, axis=1, keepdims=True)) @ u))
    X = P - P.mean(0)
    X = X - np.outer(X @ u, u)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    Y = (P - P.mean(0)) @ np.stack([u, v], 1)
    return Y, cos


def usable(emb: Embeddings, pairs: list[tuple[str, str]], max_rank: int = 30_000) -> list[tuple[str, str]]:
    return [(a, b) for a, b in pairs if emb.index.get(a, 10**9) < max_rank and emb.index.get(b, 10**9) < max_rank]


def pair_panel(ax, emb: Embeddings, pairs, title: str, c_from: str, c_to: str, note: str | None = None) -> None:
    pairs = usable(emb, pairs)
    words = [w for p in pairs for w in p]
    P, cos = offset_projection(emb.unit[[emb.index[w] for w in words]], len(pairs))
    for i, (a, b) in enumerate(pairs):
        ax.annotate("", xy=P[2 * i + 1], xytext=P[2 * i], arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2, shrinkA=3, shrinkB=3))
        ax.scatter(*P[2 * i], color=c_from, s=26, zorder=3)
        ax.scatter(*P[2 * i + 1], color=c_to, s=26, zorder=3)
        ax.annotate(a, P[2 * i], textcoords="offset points", xytext=(-4, 4), ha="right", fontsize=8, color=c_from)
        ax.annotate(b, P[2 * i + 1], textcoords="offset points", xytext=(4, 4), ha="left", fontsize=8, color=c_to)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.22)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.45 * (hi - lo))  # head-room for the analogy box
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.text(0.02, 0.02, f"mean cosine(pair offset, mean offset), full 300-D space: {cos:.2f}", transform=ax.transAxes, fontsize=7, color="#555555")
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=8, va="top", bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))


def analogy_note(emb: Embeddings, a: str, b: str, c: str, section: str, acc: dict[str, tuple[int, int]]) -> str:
    top = emb.most_similar([b, c], [a], topn=3, restrict=30_000)
    k, n = acc[section]
    return f"{b} − {a} + {c} →\n{', '.join(f'{w} ({s:.2f})' for w, s in top)}\n[{section}: {100 * k / n:.0f}% top-1, n={n}]"


def hero(emb: Embeddings, path: Path, label: str) -> None:
    from w2v.evaluate import evaluate_analogies

    res = evaluate_analogies(emb, device="cpu")
    acc = {s: (res.correct[s], res.total[s]) for s in res.total}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    pair_panel(axes[0], emb, CAPITALS, "country → capital", "#2e86ab", "#d1495b", analogy_note(emb, "france", "paris", "germany", "capital-common-countries", acc))
    pair_panel(axes[1], emb, GENDER, "male → female", "#2e86ab", "#d1495b", analogy_note(emb, "man", "king", "woman", "family", acc))
    pair_panel(axes[2], emb, PAST, "verb → past tense", "#2e86ab", "#d1495b", analogy_note(emb, "write", "wrote", "take", "gram7-past-tense", acc))
    fig.suptitle(f"Linear relations in {label}: x = mean offset direction, y = leading residual component", fontsize=11.5, y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def tsne_figure(emb: Embeddings, path: Path) -> None:
    from sklearn.manifold import TSNE

    words, cats = [], []
    for c, ws in CATEGORIES.items():
        for w in ws:
            if emb.index.get(w, 10**9) < 30_000:
                words.append(w)
                cats.append(c)
    X = emb.unit[[emb.index[w] for w in words]]
    Y = TSNE(2, perplexity=12, init="pca", random_state=0, metric="cosine").fit_transform(X)
    cmap = dict(zip(CATEGORIES, plt.get_cmap("tab10").colors))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for c in CATEGORIES:
        m = np.array(cats) == c
        ax.scatter(Y[m, 0], Y[m, 1], s=18, color=cmap[c], label=c)
    for (x, y), w in zip(Y, words):
        ax.annotate(w, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_title("t-SNE of skip-gram vectors (text8): semantic categories separate", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def training_curves(runs: dict[str, Path], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = {"skip-gram": "#d1495b", "CBOW": "#2e86ab", "GloVe": "#edae49", "GloVe (30-epoch run)": "#8c6d1f"}
    for name, d in runs.items():
        h = json.loads((d / "history.json").read_text())
        ev = h["epoch_eval"]
        ep = [e["epoch"] for e in ev]
        axes[1].plot(ep, [100 * e["overall"] for e in ev], "o-" if "30" not in name else "s--", color=colors[name], label=name, ms=3)
        if not name.startswith("GloVe"):
            prog = np.array(h["progress"]) * h["config"]["epochs"]
            loss = np.array(h["loss"])
            k = max(1, len(loss) // 200)
            sm = np.convolve(loss, np.ones(k) / k, mode="valid")
            axes[0].plot(prog[k - 1 :], sm, color=colors[name], label=name)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("negative-sampling loss per example")
    axes[0].set_title("Training loss (SGNS / CBOW; sub-sampled, K negatives)", fontsize=10)
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Google analogy accuracy (%)")
    axes[1].set_title("Analogy accuracy during training (text8, top-30k candidates)", fontsize=10)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def benchmark_bars(metrics: dict, path: Path, names: dict[str, str]) -> None:
    keys = [("analogy_sem", "analogy: semantic"), ("analogy_syn", "analogy: syntactic"), ("wordsim353", "WordSim-353 ρ"), ("simlex999", "SimLex-999 ρ")]
    fig, ax = plt.subplots(figsize=(10, 4))
    models = [m for m in names if m in metrics]
    w = 0.8 / len(models)
    cols = plt.get_cmap("tab10").colors
    for i, m in enumerate(models):
        vals = [100 * metrics[m][k] for k, _ in keys]
        ax.bar(np.arange(len(keys)) + (i - (len(models) - 1) / 2) * w, vals, w, label=names[m], color=cols[i])
        for j, v in enumerate(vals):
            ax.text(j + (i - (len(models) - 1) / 2) * w, v + 0.8, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(keys)), [l for _, l in keys])
    ax.set_ylabel("accuracy / Spearman ρ × 100")
    ax.set_title("Intrinsic evaluation (common question subset for analogies)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bias_figure(embs: dict[str, Embeddings], path: Path) -> dict:
    stats = {}
    fig, axes = plt.subplots(1, len(embs), figsize=(5.2 * len(embs), 6.2), sharex=True)
    for ax, (name, emb) in zip(np.atleast_1d(axes), embs.items()):
        g, spec = gender_direction(emb)
        proj = projections(emb, OCCUPATIONS, g)
        items = sorted(proj.items(), key=lambda kv: kv[1])
        pick = items[:14] + items[-14:]
        ws, vs = zip(*pick)
        ax.barh(range(len(ws)), vs, color=["#2e86ab" if v < 0 else "#d1495b" for v in vs])
        ax.set_yticks(range(len(ws)), ws, fontsize=8)
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_title(f"{name}\nDirectBias = {direct_bias(emb, OCCUPATIONS, g):.3f}, PC1 explains {100 * spec[0]:.0f}% of pair variance", fontsize=9)
        ax.set_xlabel("cosine with gender direction  (← he   she →)")
        stats[name] = {"direct_bias": direct_bias(emb, OCCUPATIONS, g), "pc1_variance": float(spec[0]), "n_occupations": len(proj)}
    fig.suptitle("Gender projection of occupation words (Bolukbasi et al. 2016 direction)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return stats


def ablation_figure(path: Path) -> None:
    d = json.loads(Path("results/ablations.json").read_text())
    titles = {"negative": "negative samples K", "sample": "sub-sampling threshold t", "window": "max window size", "dim": "dimension"}
    fig, axes = plt.subplots(1, len(d), figsize=(3.6 * len(d), 3.3), sharey=True)
    for ax, (name, rows) in zip(np.atleast_1d(axes), d.items()):
        x = range(len(rows))
        for key, lab, c in (("semantic", "semantic", "#d1495b"), ("syntactic", "syntactic", "#2e86ab"), ("overall", "overall", "#333333")):
            ax.plot(x, [100 * r[key] for r in rows], "o-", color=c, label=lab, ms=4)
        ax.set_xticks(list(x), [r["value"] for r in rows])
        ax.set_xlabel(titles.get(name, name))
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("Google analogy accuracy (%)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Skip-gram ablations on text8 (5 epochs, dim 200 unless varied)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def neighbours_table(embs: dict[str, Embeddings], words: list[str], topn: int = 6) -> str:
    lines = ["| word | " + " | ".join(embs) + " |", "|---|" + "---|" * len(embs)]
    for w in words:
        cells = []
        for emb in embs.values():
            cells.append(", ".join(x for x, _ in emb.most_similar([w], topn=topn, restrict=30_000)) if w in emb else "–")
        lines.append(f"| **{w}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sgns", type=Path, default=Path("runs/sgns"))
    p.add_argument("--cbow", type=Path, default=Path("runs/cbow"))
    p.add_argument("--glove", type=Path, default=Path("runs/glove"))
    p.add_argument("--glove30", type=Path, default=Path("runs/glove"))
    p.add_argument("--gensim-sg", type=Path, default=Path("runs/gensim_sg"))
    p.add_argument("--metrics", type=Path, default=Path("results/metrics.json"))
    p.add_argument("--pretrained", action="store_true")
    args = p.parse_args()
    ASSETS.mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    sg = Embeddings.load(args.sgns / "vectors.npz")
    hero(sg, ASSETS / "hero.png", "skip-gram vectors trained from scratch on text8 (17M tokens)")
    tsne_figure(sg, ASSETS / "tsne_categories.png")
    runs = {k: d for k, d in (("skip-gram", args.sgns), ("CBOW", args.cbow), ("GloVe", args.glove), ("GloVe (30-epoch run)", args.glove30)) if (d / "history.json").exists()}
    training_curves(runs, ASSETS / "training_curves.png")
    embs: dict[str, Embeddings] = {"skip-gram (ours)": sg}
    if (args.glove / "vectors.npz").exists():
        embs["GloVe (ours)"] = Embeddings.load(args.glove / "vectors.npz")
    if args.pretrained:
        from evaluate_all import load_pretrained_glove

        pre = load_pretrained_glove()
        embs["GloVe 6B (pretrained)"] = pre
    nb = neighbours_table(embs, ["king", "france", "computer", "apple", "music", "good", "run", "einstein"])
    Path("results/neighbours.md").write_text(nb + "\n")
    print(nb)
    bias_embs = {k: v for k, v in embs.items() if k in ("skip-gram (ours)", "GloVe (ours)", "GloVe 6B (pretrained)")}
    stats = bias_figure(bias_embs, ASSETS / "bias_projection.png")
    Path("results/bias.json").write_text(json.dumps(stats, indent=1))
    print(stats)
    if Path("results/ablations.json").exists():
        ablation_figure(ASSETS / "ablations.png")
    if args.metrics.exists():
        m = json.loads(args.metrics.read_text())
        names = {"sgns": "skip-gram (ours)", "gensim_sg": "skip-gram (gensim)", "cbow": "CBOW (ours)", "gensim_cbow": "CBOW (gensim)",
                 "glove": "GloVe (ours, 10 ep)", "glove30": "GloVe (ours, 30 ep)", "glove_pretrained_6B": "GloVe 6B (pretrained)"}
        mm = {k: {**v, "analogy_sem": v.get("common_analogy_sem", v["analogy_sem"]), "analogy_syn": v.get("common_analogy_syn", v["analogy_syn"])} for k, v in m.items()}
        benchmark_bars(mm, ASSETS / "benchmarks.png", names)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
