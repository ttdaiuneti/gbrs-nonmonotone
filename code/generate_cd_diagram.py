"""Nemenyi CD diagram from the rerun Friedman ranks (English labels).

Reads results/stats_ranks_rerun.csv and results/stats_friedman_rerun.csv
and writes figures/fig_cd_diagram.pdf (and .png).
"""
from __future__ import annotations

from pathlib import Path

import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Embed TrueType (Type 42) fonts, as journals require; no Type 3 fonts.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# Serif text and STIX math, to match the manuscript font.
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures"

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
    mant, exp = f"{p:.1e}".split("e")
    return rf"{mant}\times10^{{{int(exp)}}}"


def draw_panel(ax, ranks: dict, cd: float, title: str, p_chi2: float, n: int):
    """Demsar-style critical-difference diagram.

    The rank axis runs along the top (rank 1 = best, on the left).  Each method
    hangs from its rank with an elbow line to a label on the nearer side of the
    axis; methods whose ranks differ by less than CD are joined by a thick bar.
    Elbow depths are chosen so that no line crosses another one."""
    lo, hi = 1.0, 3.0
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(-1.55, 1.45)
    ax.axis("off")

    # rank axis on top
    ax.plot([lo, hi], [0, 0], color="black", lw=1.2)
    for t in (1.0, 1.5, 2.0, 2.5, 3.0):
        big = t % 1.0 == 0.0
        ax.plot([t, t], [0, 0.07 if big else 0.045], color="black", lw=1.1)
        if big:
            ax.text(t, 0.12, f"{t:g}", ha="center", va="bottom", fontsize=10)

    # CD ruler above the axis
    y_cd = 0.82
    x0 = lo
    ax.plot([x0, x0 + cd], [y_cd, y_cd], color="black", lw=1.8)
    for xe in (x0, x0 + cd):
        ax.plot([xe, xe], [y_cd - 0.07, y_cd + 0.07], color="black", lw=1.8)
    ax.text(x0 + cd / 2, y_cd + 0.12, f"CD = {cd:.2f}", ha="center",
            va="bottom", fontsize=10)

    order = sorted(METHODS, key=lambda m: ranks[m])
    left = [m for m in order if ranks[m] < (lo + hi) / 2]
    right = [m for m in order if ranks[m] >= (lo + hi) / 2]
    # shallowest elbow for the method nearest the outer end of its group
    def depths(group, outer_first):
        seq = group if outer_first else group[::-1]
        return {m: -(0.55 + 0.42 * k) for k, m in enumerate(seq)}
    dl = depths(left, outer_first=True)      # leftmost = shallowest
    dr = depths(right[::-1], outer_first=True)  # rightmost = shallowest
    for group, dep, side in ((left, dl, "L"), (right, dr, "R")):
        for m in group:
            r, y = ranks[m], dep[m]
            xl = lo - 0.12 if side == "L" else hi + 0.12
            ax.plot([r, r, xl], [0, y, y], color="black", lw=1.0)
            ax.text(xl + (-0.06 if side == "L" else 0.06), y,
                    f"{m} ({r:.2f})", ha="right" if side == "L" else "left",
                    va="center", fontsize=10)

    # thick bars joining methods that are not significantly different
    pad = 0.03
    drawn = 0
    for g in cliques(ranks, cd):
        if len(g) < 2:
            continue
        lo_g = min(ranks[m] for m in g) - pad
        hi_g = max(ranks[m] for m in g) + pad
        yy = -0.20 - 0.14 * drawn
        ax.plot([lo_g, hi_g], [yy, yy], color="black", lw=3.5,
                solid_capstyle="butt")
        drawn += 1

    ax.text((lo + hi) / 2, -1.5, title + rf", Friedman $p={_fmt_p(p_chi2)}$",
            ha="center", va="bottom", fontsize=10.5)


def main() -> None:
    ranks_df = pd.read_csv(RESULTS / "stats_ranks_rerun.csv")
    fried = pd.read_csv(RESULTS / "stats_friedman_rerun.csv").set_index("metric")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    panels = [
        ("knn3_acc", "kNN-3"),
        ("svm_acc", "SVM-RBF"),
    ]
    for ax, (metric, name), tag in zip(axes, panels, "ab"):
        sub = ranks_df[ranks_df.metric == metric].set_index("method")
        ranks = {m: float(sub.loc[m, "mean_rank"]) for m in METHODS}
        row = fried.loc[metric]
        draw_panel(ax, ranks, float(row["nemenyi_CD_alpha05"]),
                   f"({tag}) {name}", float(row["p_chi2"]),
                   int(row["n_datasets"]))

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
