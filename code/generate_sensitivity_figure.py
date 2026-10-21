"""Figure: mean kNN-3 accuracy against the parameter of each reducer.

Reads results/param_sensitivity_summary.csv (written by
summarize_param_sensitivity.py) and results/classification_rerun.csv; writes a
PDF to the directory given as argv[1] (default: current directory).
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "STIXGeneral",
                            "mathtext.fontset": "stix", "font.size": 9})
HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
s = pd.read_csv(HERE.parent / "results" / "param_sensitivity_summary.csv")
c = pd.read_csv(HERE.parent / "results" / "classification_rerun.csv")
c = c[c.dataset.isin(["ecoli", "diabetes", "yeast", "wine", "heart_statlog", "vehicle",
                      "breast_cancer", "segment", "steel_plates", "ionosphere",
                      "dermatology", "pc3", "waveform", "spectf", "sonar", "digits"])]
full = c.drop_duplicates("dataset").full_knn3.mean()
# released GBRS pipeline (its own grid and 5-NN selection), drawn as a reference line
rel = pd.read_csv(HERE.parent / "results" / "gbrs_released_eval.csv").knn3_acc.mean()


S = pd.read_csv(HERE.parent / "results" / "param_sensitivity.csv").rename(columns={"|R|": "R"})
C = c.rename(columns={"|R|": "R"})
MAIN = {"GBRS": "adaptive", "HTBS": "alpha=0.1", "NRS": "scale=1"}


def rows(m, st):
    if st == MAIN[m]:
        return C[C.method == m].set_index("dataset")
    return S[(S.method == m) & (S.setting == st)].set_index("dataset")


def val(m, st):
    """Mean kNN-3 over the datasets on which both the setting and the main setting give a nonempty reduct."""
    g, ref = rows(m, st), rows(m, MAIN[m])
    ds = [d for d in g.index if d in ref.index and g.loc[d, "R"] > 0 and ref.loc[d, "R"] > 0]
    return float(g.loc[ds, "knn3_acc"].mean())


panels = [
    ("GBRS", r"(a) Minimum ball size",
     ["adaptive", "nmin=2", "nmin=3", "nmin=5", "nmin=10"],
     ["adaptive", "2", "3", "5", "10"], "adaptive"),
    ("HTBS", r"(b) HTBS: entropy fraction $\alpha$",
     ["alpha=0.05", "alpha=0.1", "alpha=0.2", "alpha=0.3"],
     ["0.05", "0.10", "0.20", "0.30"], "alpha=0.1"),
    ("NRS", r"(c) NRS: radius",
     ["scale=0.5", "scale=1", "scale=2", "fixed"],
     [r"$0.5\,\bar\ell_A/\sqrt{|A|}$", r"$1\,\bar\ell_A/\sqrt{|A|}$", r"$2\,\bar\ell_A/\sqrt{|A|}$", "fixed"], "scale=1"),
]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
for ax, (m, title, keys, labels, main) in zip(axes, panels):
    y = [val(m, k) for k in keys]
    ax.axhline(full, color="#444444", ls="--", lw=1)
    if m == "GBRS":
        ax.axhline(rel, color="#2a8a3a", ls=":", lw=1.3)
    n_line = 3 if m == "NRS" else len(keys)
    ax.plot(range(n_line), y[:n_line], color="#8fa8d0", lw=1, zorder=1)
    for i, (k, v) in enumerate(zip(keys, y)):
        ax.plot(i, v, "o", ms=6, color="#1f4e9c" if k == main else "white",
                mec="#1f4e9c", zorder=2)
        ax.text(i, v + (0.014 if m == "GBRS" and i > 0 else -0.028), f"{v:.3f}", ha="center", va="bottom" if (m == "GBRS" and i > 0) else "top", fontsize=7)
    if m == "GBRS":
        hk = ["alpha=0.1", "nmin=2", "nmin=3", "nmin=5", "nmin=10"]
        hy = [val("HTBS", k) for k in hk]
        ax.plot(range(5), hy, color="#d08a3c", lw=1, zorder=1)
        for i, (k, v) in enumerate(zip(hk, hy)):
            ax.plot(i, v, "s", ms=5.5, color="#c06a10" if i == 0 else "white", mec="#c06a10", zorder=2)
            ax.text(i, v - 0.028, f"{v:.3f}", ha="center", va="top", fontsize=7, color="#a05a10")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=0 if m != "NRS" else 20,
                       ha="center" if m != "NRS" else "right")
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0.58, 0.90)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
axes[0].set_ylabel(r"mean $k$NN-3 accuracy")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color="#1f4e9c", marker="o", mfc="white", ms=6, label="GBRS-A"),
                    Line2D([], [], color="#c06a10", marker="s", mfc="white", ms=5.5, label="HTBS ($\\alpha=0.10$)"),
                    Line2D([], [], color="#2a8a3a", ls=":", lw=1.3, label="GBRS (released)"),
                    Line2D([], [], color="#444444", ls="--", lw=1, label="all attributes")],
           loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out / "fig_sensitivity.pdf")
print("wrote", out / "fig_sensitivity.pdf", "full =", round(full, 3), "released =", round(rel, 3))
