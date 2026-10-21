"""Write the accuracy, runtime, balanced-metric and statistics tables for four reducers.

GBRS is the released pipeline (results/gbrs_released_eval.csv); GBRS-A, HTBS and
NRS come from results/classification_rerun.csv and classification_balanced.csv.
Statistics come from run_statistics_v2.py. Every cell is formatted from the raw
CSVs. Bold marks the best reducer in a row and underline the second best (ties
broken in the column order).

Usage: python3 make_results_tables_v2.py <tables directory>
Writes tab_results_block.tex, tab_balanced.tex, tab_wilcoxon.tex, tab_friedman.tex
"""
import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
ORDER = ["ecoli", "diabetes", "yeast", "wine", "heart_statlog", "vehicle", "segment",
         "steel_plates", "breast_cancer", "ionosphere", "dermatology", "pc3", "waveform",
         "spectf", "sonar", "digits"]
NAME = {"ecoli": "Ecoli", "diabetes": "Diabetes", "yeast": "Yeast", "wine": "Wine",
        "heart_statlog": "Heart-Statlog", "vehicle": "Vehicle",
        "breast_cancer": "BreastCancer", "segment": "Segment",
        "steel_plates": "SteelPlates$^{\\dagger}$", "ionosphere": "Ionosphere",
        "dermatology": "Dermatology", "pc3": "PC3", "waveform": "Waveform",
        "spectf": "SPECTF", "sonar": "Sonar", "digits": "Digits"}
M = ["GBRS", "GBRS-A", "HTBS", "NRS"]
EXCL = {"iris", "glass", "banknote", "balance_scale"}


def load():
    c = pd.read_csv(RES / "classification_rerun.csv")
    c = c[~c.dataset.isin(EXCL) & c.method.isin(["GBRS", "HTBS", "NRS"])].copy()
    c["method"] = c["method"].replace({"GBRS": "GBRS-A"})
    e = pd.read_csv(RES / "gbrs_released_eval.csv")
    e["method"] = "GBRS"
    e["runtime_s"] = e["runtime_selected_s"]
    full = c[c.method == "GBRS-A"].set_index("dataset")[["full_knn3", "full_svm"]]
    e = e.join(full, on="dataset")
    df = pd.concat([e, c], ignore_index=True)
    return {m: df[df.method == m].set_index("dataset") for m in M}, e.set_index("dataset")


def head(first, sub):
    cm = "".join(f"\\cmidrule(lr){{{4 + 2 * k}-{5 + 2 * k}}}" for k in range(4))
    return ("\\toprule\n"
            f"Dataset & $m$ & {first} & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in M)
            + " \\\\\n" + cm + "\n & & & " + " & ".join(f"$|R|$ & {sub}" for _ in M) + " \\\\\n\\midrule\n")


def acc_table(W, kind):
    acc, sd, full = (("knn3_acc", "knn3_std", "full_knn3") if kind == "knn"
                     else ("svm_acc", "svm_std", "full_svm"))
    ranks = pd.read_csv(RES / "stats_ranks_v2.csv").set_index(["metric", "method"]).mean_rank
    rows = []
    for ds in ORDER:
        vals = {m: W[m].loc[ds] for m in M if W[m].loc[ds, "|R|"] > 0}
        srt = sorted(vals, key=lambda m: -vals[m][acc])
        cells = []
        for m in M:
            r = W[m].loc[ds]
            if m not in vals:
                cells += [str(int(r["|R|"])), "--"]
                continue
            c = f"{r[acc]:.3f}$\\pm${r[sd]:.3f}"
            if m == srt[0]:
                c = f"\\textbf{{{c}}}"
            elif len(srt) > 1 and m == srt[1]:
                c = f"\\underline{{{c}}}"
            cells += [str(int(r["|R|"])), c]
        rows.append(f"{NAME[ds]} & {int(W['GBRS-A'].loc[ds, 'd'])} & {W['GBRS-A'].loc[ds, full]:.3f} & "
                    + " & ".join(cells) + " \\\\")
    keep = [d for d in ORDER if d != "waveform"]
    f15, f16 = W["GBRS-A"].loc[keep, full].mean(), W["GBRS-A"].loc[ORDER, full].mean()
    a15 = " & ".join(f"{W[m].loc[keep, '|R|'].mean():.1f} & {W[m].loc[keep, acc].mean():.3f}" for m in M)
    a16 = " & ".join(f"{W[m].loc[ORDER, '|R|'].mean():.1f} & {W[m].loc[ORDER, acc].mean():.3f}" for m in M[:3])
    rk = " & ".join(f" & {ranks.loc[(acc, m)]:.2f}" for m in M)
    tail = ("\\midrule\n"
            f"\\textit{{Average, 15 datasets}} & & {f15:.3f} & {a15} \\\\\n"
            f"\\textit{{Average, 16 datasets}} & & {f16:.3f} & {a16} & -- & -- \\\\\n"
            f"\\textit{{Average rank, 15 datasets}} & & & {rk} \\\\\n")
    return "\n".join(rows) + "\n" + tail


def runtime_table(W, E):
    rows = []
    for ds in ORDER:
        cells = [f"{int(W['GBRS'].loc[ds, '|R|'])} & {E.loc[ds, 'runtime_selected_s']:.1f} & "
                 f"{E.loc[ds, 'runtime_sweep_s']:.0f}"]
        cells += [f"{int(W[m].loc[ds, '|R|'])} & {W[m].loc[ds, 'runtime_s']:.1f}" for m in M[1:]]
        rows.append(f"{NAME[ds]} & {int(W['GBRS-A'].loc[ds, 'd'])} & " + " & ".join(cells) + " \\\\")
    keep = [d for d in ORDER if d != "waveform"]
    def avg(ix, ms):
        out = [f"{W['GBRS'].loc[ix, '|R|'].mean():.1f} & {E.loc[ix, 'runtime_selected_s'].mean():.1f} & "
               f"{E.loc[ix, 'runtime_sweep_s'].mean():.0f}"]
        out += [f"{W[m].loc[ix, '|R|'].mean():.1f} & {W[m].loc[ix, 'runtime_s'].mean():.1f}" for m in ms]
        return " & ".join(out)
    return ("\n".join(rows) + "\n\\midrule\n"
            f"\\textit{{Average, 15 datasets}} & & {avg(keep, M[1:])} \\\\\n"
            f"\\textit{{Average, 16 datasets}} & & {avg(ORDER, M[1:3])} & -- & -- \\\\\n")


def results_block(W, E):
    spec = "@{}l @{\\extracolsep{\\fill}} r r r c r c r c r c@{}"
    knn = ("\\begin{table*}[pos=!htbp]\n\\centering\n"
           "\\caption{$k$NN-3 post-reduction accuracy (mean$\\pm$std, 5-fold CV, seed~0). GBRS is the released "
           "pipeline; GBRS-A is our size-adaptive implementation with the target stopping rule of Algorithm~\\ref{alg:greedy}.}\n"
           "\\label{tab:knn3}\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.12}\n"
           f"\\begin{{tabular*}}{{\\linewidth}}{{{spec}}}\n" + head("Acc$_{\\text{full}}$", "Acc$\\,\\pm\\,$std")
           + acc_table(W, "knn") + "\\bottomrule\n\\end{tabular*}\n\\tabnote{%\n"
           "\\textbf{Bold}: best per row; \\underline{underline}: second best. $|R|=0$: empty reduct. The 15-dataset "
           "averages and the average ranks exclude Waveform, where NRS returns an empty reduct (ranks as in "
           "Table~\\ref{tab:friedman}). GBRS keeps, for each dataset, the reduct with the best 5-NN cross-validation "
           "score over its grid of minimum ball sizes, so its reducts are selected with the labels and a classifier "
           "(Section~\\ref{sec:nested} measures the effect).\n"
           "$^{\\dagger}$SteelPlates uses the leakage-free feature set: the eight one-hot fault-indicator columns that "
           "jointly determine the label are removed ($m:33\\to25$; Section~\\ref{sec:dataqual}).}\n\\end{table*}\n")
    svm = ("\\begin{table*}[pos=!htbp]\n\\centering\n"
           "\\caption{SVM-RBF ($C=1$, \\texttt{gamma=scale}) post-reduction accuracy on the same reducts as Table~\\ref{tab:knn3}.}\n"
           "\\label{tab:svm}\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.12}\n"
           f"\\begin{{tabular*}}{{\\linewidth}}{{{spec}}}\n" + head("SVM$_{\\text{full}}$", "Acc$\\,\\pm\\,$std")
           + acc_table(W, "svm") + "\\bottomrule\n\\end{tabular*}\n\\tabnote{%\n"
           "$^{\\dagger}$SteelPlates uses the leakage-free feature set ($m:33\\to25$; see Table~\\ref{tab:knn3}).}\n\\end{table*}\n")
    rt = ("\\begin{table*}[pos=!htbp]\n\\centering\n"
          "\\caption{Reduct size and wall-clock reduction time (seconds, single CPU core) for the runs of Table~\\ref{tab:knn3}.}\n"
          "\\label{tab:runtime}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.12}\n"
          "\\begin{tabular*}{\\linewidth}{@{}l @{\\extracolsep{\\fill}} r r r r r r r r r r@{}}\n\\toprule\n"
          "Dataset & $m$ & \\multicolumn{3}{c}{GBRS} & \\multicolumn{2}{c}{GBRS-A} & \\multicolumn{2}{c}{HTBS} & "
          "\\multicolumn{2}{c}{NRS} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\n"
          " & & $|R|$ & Run & Sweep & $|R|$ & Time & $|R|$ & Time & $|R|$ & Time \\\\\n\\midrule\n"
          + runtime_table(W, E) + "\\bottomrule\n\\end{tabular*}\n\\tabnote{%\n"
          "Time covers the whole forward-greedy loop. For GBRS, Run is the reduction at the selected minimum ball size "
          "and Sweep is the total over its grid ($m+1$ values; five values on Yeast, PC3, SteelPlates, Segment, Waveform "
          "and Digits), without the 5-NN scoring. $^{\\dagger}$SteelPlates on the leakage-free feature set ($m:33\\to25$).}\n"
          "\\end{table*}\n")
    return knn + "\n" + svm + "\n" + rt


def balanced(W, E):
    b = pd.read_csv(RES / "classification_balanced.csv")
    b = b[(b.dataset != "steel_plates") | (b.steel_cleaned == 1)].set_index(["dataset", "method"])
    rows = []
    for ds in ORDER:
        trip = {"GBRS": (E.loc[ds, "knn_bal_acc"], E.loc[ds, "knn_f1_macro"], E.loc[ds, "knn_mcc"])}
        for m, src in (("GBRS-A", "GBRS"), ("HTBS", "HTBS"), ("NRS", "NRS")):
            r = b.loc[(ds, src)]
            trip[m] = None if r["|R|"] == 0 else (r["knn_bal_acc"], r["knn_f1_macro"], r["knn_mcc"])
        top = max(round(v[0], 3) for v in trip.values() if v)
        cells = []
        for m in M:
            v = trip[m]
            if v is None:
                cells.append("-- & -- & --")
                continue
            ba = f"{v[0]:.3f}"
            if round(v[0], 3) == top:
                ba = f"\\textbf{{{ba}}}"
            cells.append(f"{ba} & {v[1]:.3f} & {v[2]:.3f}")
        plain = NAME[ds].replace("$^{\\dagger}$", "")
        rows.append(f"{plain} & " + " && ".join(cells) + " \\\\")
    return ("\\begin{table*}[pos=!htbp]\n\\centering\n"
            "\\caption{Imbalance-aware post-reduction metrics ($k$NN-3): balanced accuracy (BAcc), macro-$F_1$, and Matthews correlation (MCC).}\n"
            "\\label{tab:balanced}\n\\small\n\\setlength{\\tabcolsep}{2.6pt}\n"
            "\\begin{tabular*}{\\linewidth}{@{}l @{\\extracolsep{\\fill}} ccc c ccc c ccc c ccc@{}}\n\\toprule\n"
            " & " + " && ".join(f"\\multicolumn{{3}}{{c}}{{{m}}}" for m in M) + " \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){6-8}\\cmidrule(lr){10-12}\\cmidrule(lr){14-16}\n"
            "Dataset & " + " && ".join("BAcc & $F_1$ & MCC" for _ in M) + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\tabnote{%\n"
            "Reduct sizes $|R|$ are in Table~\\ref{tab:knn3}. Bold marks the best BAcc per dataset (ties in bold). "
            "Reducts that collapse to the majority class show BAcc$\\approx0.5$ and MCC$\\approx0$ despite high plain "
            "accuracy (PC3). SteelPlates uses the leakage-free feature set; the Waveform NRS reduct is empty.}\n\\end{table*}\n")


def pfmt(p):
    return f"{p:.4f}" if p >= 1e-4 else f"$<$0.0001"


def wilcoxon_table():
    w = pd.read_csv(RES / "stats_wilcoxon_holm_v2.csv")
    lines = []
    for metric, lab in (("knn3_acc", "kNN-3"), ("svm_acc", "SVM-RBF")):
        for _, r in w[w.metric == metric].iterrows():
            md = lambda x: f"{x:+.3f}".replace("-", "$-$")
            sig = "$<0.05$" if r.p_holm < 0.05 else "n.s."
            lines.append(f"{r.A} vs {r.B} & {lab} & {int(r.n)} & {md(r.mean_diff)} [{md(r.ci_lo)}, {md(r.ci_hi)}] & "
                         f"{int(r.wins)}/{int(r.draws)}/{int(r.losses)} & {pfmt(r.p)} & {pfmt(r.p_holm)} & {sig} \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    return ("\\begin{table*}[pos=!htbp]\n\\centering\n\\caption{Pairwise comparison of the reducers on post-reduction accuracy.}\n"
            "\\label{tab:wilcoxon}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\n"
            "\\begin{tabular*}{\\linewidth}{@{}ll @{\\extracolsep{\\fill}} rcrrrl@{}}\n\\toprule\n"
            "Comparison & Classifier & $n$ & Mean diff [95\\% CI] & W/D/L & $p$ & $p_{\\text{Holm}}$ & Sig. \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular*}\n\\tabnote{%\n"
            "Mean diff: mean accuracy difference over the datasets, with a $t$-based 95\\% confidence interval (not adjusted "
            "for multiplicity). W/D/L: datasets on which the first method is more accurate, tied, or less accurate. $p$: "
            "two-tailed Wilcoxon signed-rank test, zero differences dropped from the rank sum; $p_{\\text{Holm}}$: Holm "
            "correction over the six pairs for each classifier. Pairs without NRS use all $n=16$ datasets; pairs with NRS "
            "use the $n=15$ datasets on which NRS returns a non-empty reduct (Waveform excluded). Positive mean diff: the "
            "first method has higher accuracy.}\n\\end{table*}\n")


def friedman_table():
    rk = pd.read_csv(RES / "stats_ranks_v2.csv").set_index(["metric", "method"]).mean_rank
    fr = pd.read_csv(RES / "stats_friedman_v2.csv").set_index("metric")
    def sci(p):
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    rows = []
    for metric, lab in (("knn3_acc", "$k$NN-3"), ("svm_acc", "SVM-RBF")):
        best = min(rk.loc[(metric, m)] for m in M)
        cells = [(f"\\textbf{{{rk.loc[(metric, m)]:.2f}}}" if rk.loc[(metric, m)] == best else f"{rk.loc[(metric, m)]:.2f}")
                 for m in M]
        rows.append(f"{lab} & " + " & ".join(cells) + f" & {sci(fr.loc[metric, 'p_chi2'])} \\\\")
    cd = fr["nemenyi_CD_alpha05"].iloc[0]
    return ("\\begin{table}[pos=!htbp]\n\\centering\n"
            "\\caption{Friedman test and average ranks over the $n=15$ datasets with non-empty reducts for all four reducers (rank~1 = best).}\n"
            "\\label{tab:friedman}\n\\small\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
            " & \\multicolumn{4}{c}{Average rank} & \\\\\n\\cmidrule(lr){2-5}\n"
            "Classifier & " + " & ".join(M) + " & Friedman $p$ \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n\\tabnote{%\n"
            f"Waveform is excluded because NRS returns $|R|=0$. At $\\alpha=0.05$ the Nemenyi critical difference is "
            f"$\\mathrm{{CD}}={cd:.2f}$.}}\n\\end{{table}}\n")


def main(outdir):
    W, E = load()
    o = Path(outdir)
    (o / "tab_results_block.tex").write_text(results_block(W, E))
    (o / "tab_balanced.tex").write_text(balanced(W, E))
    (o / "tab_wilcoxon.tex").write_text(wilcoxon_table())
    (o / "tab_friedman.tex").write_text(friedman_table())
    print("wrote 4 files to", o)


if __name__ == "__main__":
    main(sys.argv[1])
