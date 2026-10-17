"""Nemenyi CD diagram from the rerun Friedman ranks (English labels).

Reads results/stats_ranks_rerun.csv and results/stats_friedman_rerun.csv
and writes figures/fig_cd_diagram.pdf (and .png).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "NEU" / "manuscript" / "figures"

METHODS = ["NRS", "GBRS", "HTBS"]
COLORS = {"NRS": "#1f77b4", "GBRS": "#d62728", "HTBS": "#2ca02c"}


def cliques(ranks: dict, cd: float) -> list[list[str]]:
    """Maximal groups of methods whose rank range is <= CD."""
    order = sorted(METHODS, key=lambda m: ranks[m])
    groups = []
    for i, a in enumerate(order):
        for j in range(len(order), i, -1):
            chunk = order[i:j]
            if ranks[chunk[-1]] - ranks[chunk[0]] <= cd + 1e-12:
                groups.append(chunk)
                break
    # drop groups contained in a larger one
    keep = []
    for g in groups:
        if not any(set(g) < set(h) for h in groups):
            keep.append(g)
    return keep


def _fmt_p(p: float) -> str:
    return f"{p:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def draw_panel(ax, ranks: dict, cd: float, title: str, p_chi2: float, n: int):
    """Demsar-style CD diagram: clique bars sit immediately under the axis so a
    tie is visibly *joined* (the caption's claim), and the CD ruler is placed
    clear of the method labels.  Bars carry a small visual padding because a
    tie can be far narrower than a line width (e.g. 0.03 rank on kNN-3)."""
    xmin, xmax = 0.80, 3.20
    ax.set_xlim(xmin, xmax)  # rank 1 = best, on the left (Demsar 2006)
    ax.set_ylim(-1.05, 1.75)
    ax.axhline(0.0, color="black", lw=1.2, zorder=1)
    for t in (1.0, 1.5, 2.0, 2.5, 3.0):
        big = (t % 1.0 == 0.0)
        ax.plot([t, t], [-0.05, 0.05] if big else [-0.03, 0.03],
                color="black", lw=1.1 if big else 0.8, zorder=2)
        if big:
            ax.text(t, 0.09, f"{t:g}", ha="center", va="bottom", fontsize=8.5)

    # CD ruler, kept above every method label
    y_cd = 1.46
    cd_left = 1.02
    ax.plot([cd_left, cd_left + cd], [y_cd, y_cd], color="black", lw=2.0, zorder=3)
    for xe in (cd_left, cd_left + cd):
        ax.plot([xe, xe], [y_cd - 0.09, y_cd + 0.09], color="black", lw=2.0, zorder=3)
    ax.text(cd_left + cd / 2, y_cd + 0.14, f"CD = {cd:.2f}", ha="center",
            va="bottom", fontsize=9, fontweight="bold")

    # method markers with stems and labels, staggered so near-equal ranks
    # (GBRS 2.37 vs HTBS 2.40) do not print on top of one another
    order = sorted(METHODS, key=lambda m: ranks[m])
    levels = [0.34, 0.86, 0.34]
    for i, m in enumerate(order):
        r = ranks[m]
        y = levels[i]
        ax.plot([r, r], [0.0, y], color=COLORS[m], lw=1.4, zorder=3)
        ax.plot(r, 0.0, "o", color=COLORS[m], ms=7, zorder=4)
        ax.text(r, y + 0.04, f"{m}\n{ranks[m]:.2f}", ha="center", va="bottom",
                fontsize=9, color=COLORS[m], fontweight="bold", linespacing=1.15)

    # clique bars: immediately below the axis, so "joined by a bar" is literal
    pad = 0.055
    ybar = -0.30
    drawn = 0
    for g in cliques(ranks, cd):
        if len(g) < 2:
            continue
        lo = min(ranks[m] for m in g) - pad
        hi = max(ranks[m] for m in g) + pad
        yy = ybar - 0.26 * drawn
        ax.plot([lo, hi], [yy, yy], color="#c0392b", lw=4.0,
                solid_capstyle="round", zorder=5)
        gap = max(ranks[m] for m in g) - min(ranks[m] for m in g)
        ax.text((lo + hi) / 2, yy - 0.10,
                r"$\Delta$ rank $=%.2f \leq$ CD" % gap, ha="center", va="top",
                fontsize=7.4, color="#c0392b")
        drawn += 1

    ax.set_title(title + f"\nFriedman $p={_fmt_p(p_chi2)}$, $n={n}$",
                 fontsize=10, pad=8)
    ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False)


def main() -> None:
    ranks_df = pd.read_csv(RESULTS / "stats_ranks_rerun.csv")
    fried = pd.read_csv(RESULTS / "stats_friedman_rerun.csv").set_index("metric")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35))
    panels = [
        ("knn3_acc", "kNN-3"),
        ("svm_acc", "SVM-RBF"),
    ]
    for ax, (metric, name) in zip(axes, panels):
        sub = ranks_df[ranks_df.metric == metric].set_index("method")
        ranks = {m: float(sub.loc[m, "mean_rank"]) for m in METHODS}
        row = fried.loc[metric]
        draw_panel(ax, ranks, float(row["nemenyi_CD_alpha05"]),
                   name, float(row["p_chi2"]), int(row["n_datasets"]))

    fig.suptitle(
        "Nemenyi CD diagram — average rank (rank 1 = best); "
        "red bar = not significantly different",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()
    pdf = FIGS / "fig_cd_diagram.pdf"
    png = FIGS / "fig_cd_diagram.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
