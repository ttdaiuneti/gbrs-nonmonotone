"""Rewrite the data rows of the accuracy tables with three decimals.

Reads results/classification_rerun.csv and results/stats_ranks_rerun.csv and
rewrites the dataset rows and the Average rows of the kNN-3 and SVM tables in
the given tab_results_block.tex.  Bold marks the best reducer in a row and
underline the second best (ties are broken in the order GBRS, HTBS, NRS).
Values are formatted from the raw CSV, not from rounded table cells.

Usage: python format_results_tables.py path/to/tab_results_block.tex
"""
import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
ORDER = ["ecoli", "diabetes", "yeast", "wine", "heart_statlog", "vehicle",
         "breast_cancer", "segment", "steel_plates", "ionosphere", "dermatology",
         "pc3", "waveform", "spectf", "sonar", "digits"]
NAME = {"ecoli": "Ecoli", "diabetes": "Diabetes", "yeast": "Yeast", "wine": "Wine",
        "heart_statlog": "Heart-Statlog", "vehicle": "Vehicle",
        "breast_cancer": "BreastCancer", "segment": "Segment",
        "steel_plates": "SteelPlates$^{\\dagger}$", "ionosphere": "Ionosphere",
        "dermatology": "Dermatology", "pc3": "PC3", "waveform": "Waveform",
        "spectf": "SPECTF", "sonar": "Sonar", "digits": "Digits"}
M = ["GBRS", "HTBS", "NRS"]


def build(df, ranks, kind):
    acc, sd, full = (("knn3_acc", "knn3_std", "full_knn3") if kind == "knn"
                     else ("svm_acc", "svm_std", "full_svm"))
    rk = "knn3_acc" if kind == "knn" else "svm_acc"
    rows = []
    for ds in ORDER:
        sub = df[df.dataset == ds].set_index("method")
        vals = {}
        for m in M:
            r = sub.loc[m]
            if m == "NRS" and r["|R|"] == 0:
                continue
            vals[m] = r
        srt = sorted(vals, key=lambda m: -vals[m][acc])  # stable
        best, second = srt[0], srt[1] if len(srt) > 1 else None
        cells = []
        for m in M:
            if m not in vals:
                cells += [str(int(sub.loc[m, "|R|"])), "--"]
                continue
            r = vals[m]
            c = f"{r[acc]:.3f}$\\pm${r[sd]:.3f}"
            if m == best:
                c = f"\\textbf{{{c}}}"
            elif m == second:
                c = f"\\underline{{{c}}}"
            cells += [str(int(r["|R|"])), c]
        rows.append(f"{NAME[ds]} & {int(sub.iloc[0]['d'])} & "
                    f"{sub.iloc[0][full]:.3f} & " + " & ".join(cells) + r" \\")
    def avg(m, drop_wave):
        s = df[df.method == m]
        if drop_wave:
            s = s[s.dataset != "waveform"]
        return s["|R|"].mean(), s[acc].mean()
    f15 = df[(df.method == "GBRS") & (df.dataset != "waveform")][full].mean()
    f16 = df[df.method == "GBRS"][full].mean()
    a15 = [avg(m, True) for m in M]
    a16 = [avg(m, False) for m in ("GBRS", "HTBS")]
    r15 = " & ".join(f"{r:.1f} & {a:.3f}" for r, a in a15)
    r16 = " & ".join(f"{r:.1f} & {a:.3f}" for r, a in a16)
    rrk = [ranks.loc[(rk, m)] for m in M]
    tail = [
        "\\midrule",
        f"\\textit{{Average, 15 datasets}} & & {f15:.3f} & {r15} \\\\",
        f"\\textit{{Average, 16 datasets}} & & {f16:.3f} & {r16} & -- & -- \\\\",
        f"\\textit{{Average rank, 15 datasets}} & & & & {rrk[0]:.2f} & & {rrk[1]:.2f} & & {rrk[2]:.2f} \\\\",
    ]
    return rows, tail


def main(path):
    df = pd.read_csv(RES / "classification_rerun.csv")
    df = df[~df.dataset.isin({"iris", "glass", "banknote", "balance_scale"})]
    df = df[df.method.isin(M)]
    ranks = pd.read_csv(RES / "stats_ranks_rerun.csv").set_index(["metric", "method"]).mean_rank
    text = Path(path).read_text().split("\n")
    out, i, tab = [], 0, 0
    kinds = ["knn", "svm"]
    while i < len(text):
        out.append(text[i])
        if text[i].startswith("\\midrule") and tab < 2 and any(
                l.startswith("Ecoli &") for l in text[i + 1:i + 2]):
            rows, tail = build(df, ranks, kinds[tab])
            out += rows + tail
            j = i + 1
            while not text[j].startswith("\\bottomrule"):
                j += 1
            i = j
            tab += 1
            continue
        i += 1
    Path(path).write_text("\n".join(out))
    print("rewrote", tab, "tables")


if __name__ == "__main__":
    main(sys.argv[1])
