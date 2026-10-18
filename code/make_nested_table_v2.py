"""Nested-CV table for the four reducers (Table tab:nested).

GBRS (released pipeline): per outer fold, the minimum ball size is chosen inside the
training fold by the first maximum of the 5-NN score in i order, as in the released
script. Small datasets: results/gbrs_official_nested.csv (full grid, choice already
made); six larger datasets: results/gbrs_official_nested_grid.csv (coarse grid, one
row per fold and i). The leaky value is the main-table score
(results/gbrs_released_eval.csv). GBRS-A, HTBS, NRS: results/nested_cv_16sets.csv.

Usage: python3 make_nested_table_v2.py <output .tex>
Writes the table and results/gbrs_official_nested_summary.csv; prints the tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results"
LARGE = ["yeast", "pc3", "steel_plates", "segment", "waveform", "digits"]
NAME = {"steel_plates": "SteelPlates", "spectf": "SPECTF", "pc3": "PC3", "breast_cancer": "BreastCancer",
        "heart_statlog": "Heart-Statlog"}


def gbrs_nested():
    sm = pd.read_csv(RES / "gbrs_official_nested.csv")
    sm = sm[sm.error.isna() & ~sm.dataset.isin(LARGE)]
    gr = pd.read_csv(RES / "gbrs_official_nested_grid.csv")
    gr = gr[gr.error.isna()].sort_values(["dataset", "fold", "i"])
    picks = []
    for (ds, f), g in gr.groupby(["dataset", "fold"]):
        picks.append(g.loc[g["cv5nn"].idxmax()])
    lg = pd.DataFrame(picks)
    allf = pd.concat([sm[["dataset", "fold", "|R|", "knn3_acc", "svm_acc"]],
                      lg[["dataset", "fold", "|R|", "knn3_acc", "svm_acc"]]])
    cnt = allf.groupby("dataset").size()
    assert (cnt == 5).all() and len(cnt) == 16, cnt.to_dict()
    return allf.groupby("dataset")[["|R|", "knn3_acc", "svm_acc"]].mean()


def main(out):
    nest = gbrs_nested()
    leak = pd.read_csv(RES / "gbrs_released_eval.csv").set_index("dataset")
    g = pd.DataFrame({"leaky_knn": leak.knn3_acc, "nested_knn": nest.knn3_acc,
                      "leaky_svm": leak.svm_acc, "nested_svm": nest.svm_acc, "R_fold_mean": nest["|R|"]})
    g["gap_knn"] = g.leaky_knn - g.nested_knn
    g.to_csv(RES / "gbrs_official_nested_summary.csv")
    o = pd.read_csv(RES / "nested_cv_16sets.csv")
    rows = [("GBRS", g.leaky_knn.mean(), g.nested_knn.mean(), 100 * g.gap_knn.mean(),
             100 * g.gap_knn.max(), g.gap_knn.idxmax())]
    nested = {"GBRS": g.nested_knn}
    for m, lab in (("GBRS", "GBRS-A"), ("HTBS", "HTBS"), ("NRS", "NRS")):
        s = o[o.method == m].set_index("dataset")
        if m == "NRS":
            s = s[s.R_all > 0]
        rows.append((lab, s.leaky_knn_acc.mean(), s.nested_knn_acc.mean(), 100 * s.gap_knn_acc.mean(),
                     100 * s.gap_knn_acc.max(), s.gap_knn_acc.idxmax()))
        nested[lab] = s.nested_knn_acc
    body = "\n".join(f"{m} & {a:.3f} & {b:.3f} & ${c:+.2f}$ & ${d:+.1f}$ ({NAME.get(e, e.capitalize())}) \\\\"
                     for m, a, b, c, d, e in rows)
    # nested-score tests
    tests = []
    for A, B in (("GBRS", "GBRS-A"), ("GBRS", "HTBS"), ("GBRS", "NRS"), ("GBRS-A", "HTBS")):
        ds = nested[A].index.intersection(nested[B].index)
        x = nested[A].loc[ds] - nested[B].loc[ds]
        p = stats.wilcoxon(x).pvalue
        tests.append((A, B, len(ds), x.mean(), (x > 1e-12).sum(), (x.abs() <= 1e-12).sum(), (x < -1e-12).sum(), p))
        print(f"nested {A} vs {B}: n={len(ds)} diff={x.mean():+.4f} W/D/L={tests[-1][4]}/{tests[-1][5]}/{tests[-1][6]} p={p:.4f}")
    full = pd.read_csv(RES / "classification_rerun.csv")
    full = full[full.method == "GBRS"].set_index("dataset").full_knn3
    x = g.nested_knn - full.loc[g.index]
    print(f"nested GBRS vs full: diff={x.mean():+.4f} W/L={(x>0).sum()}/{(x<0).sum()} p={stats.wilcoxon(x).pvalue:.4f}")
    print(g.round(3).to_string())
    print(rows)
    tex = ("\\begin{table}[pos=!htbp]\n\\centering\n"
           "\\caption{Selection-bias audit with nested stratified five-fold CV (outer folds for evaluation; the whole "
           "reduction, including the choice of the minimum ball size for GBRS, re-run on each outer training fold). Means "
           "over the sixteen datasets (fifteen for NRS, which returns an empty reduct on Waveform).}\n"
           "\\label{tab:nested}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\n"
           "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
           "Method & Leaky $k$NN-3 & Nested $k$NN-3 & Gap (pp) & Max gap (pp) \\\\\n\\midrule\n" + body +
           "\n\\bottomrule\n\\end{tabular}\n\n\\tabnote{%\n"
           "``Leaky'' reproduces the protocol of Tables~\\ref{tab:knn3}--\\ref{tab:svm} (reduct fit on the full labeled set, "
           "then CV on that fixed reduct); ``Nested'' is the unbiased estimate. Gap $=$ leaky $-$ nested in percentage points. "
           "For GBRS the grid of minimum ball sizes is the same as in the main tables. Full per-dataset rows are in the repository.}\n"
           "\\end{table}\n")
    Path(out).write_text(tex)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
