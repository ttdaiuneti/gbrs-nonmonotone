"""Write the LaTeX sensitivity table (Table 17) from the raw CSVs.

Usage: python3 make_sensitivity_table.py <output .tex path>
Also prints the matched GBRS-vs-HTBS comparison at equal minimum ball size.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = Path(__file__).resolve().parent.parent / "results"
s = pd.read_csv(R / "param_sensitivity.csv").rename(columns={"|R|": "R"})
c = pd.read_csv(R / "classification_rerun.csv").rename(columns={"|R|": "R"})
c = c[c.dataset.isin(set(s.dataset)) & c.method.isin(["GBRS", "HTBS", "NRS"])]
ref = {m: c[c.method == m].set_index("dataset") for m in ("GBRS", "HTBS", "NRS")}
MAIN = {"GBRS": "adaptive", "HTBS": "alpha=0.1", "NRS": "scale=1"}


def get(m, st):
    return ref[m] if st == MAIN[m] else s[(s.method == m) & (s.setting == st)].set_index("dataset")


order = [("GBRS", "adaptive", "Size-adaptive (main tables)"),
         ("GBRS", "nmin=2", r"$\nmin=2$"), ("GBRS", "nmin=3", r"$\nmin=3$"),
         ("GBRS", "nmin=5", r"$\nmin=5$"), ("GBRS", "nmin=10", r"$\nmin=10$"),
         ("HTBS", "alpha=0.05", r"$\alpha=0.05$"), ("HTBS", "alpha=0.1", r"$\alpha=0.10$ (main tables)"),
         ("HTBS", "alpha=0.2", r"$\alpha=0.20$"), ("HTBS", "alpha=0.3", r"$\alpha=0.30$"),
         ("HTBS", "nmin=2", r"$\nmin=2$, $\alpha=0.10$"), ("HTBS", "nmin=3", r"$\nmin=3$, $\alpha=0.10$"),
         ("HTBS", "nmin=5", r"$\nmin=5$, $\alpha=0.10$"), ("HTBS", "nmin=10", r"$\nmin=10$, $\alpha=0.10$"),
         ("NRS", "scale=0.5", r"$c=0.5$"), ("NRS", "scale=1", r"$c=1$ (main tables)"),
         ("NRS", "scale=2", r"$c=2$"), ("NRS", "fixed", "Fixed radius")]
L, prev = [], None
for m, st, lab in order:
    g, A = get(m, st), ref[m]
    ds = [d for d in g.index if g.loc[d, "R"] > 0 and A.loc[d, "R"] > 0]
    x = g.loc[ds, "knn3_acc"] - A.loc[ds, "knn3_acc"]
    if st == MAIN[m]:
        cell = "-- & -- & --"
    else:
        p = wilcoxon(x).pvalue if (x.abs() > 1e-9).any() else float("nan")
        ps = "--" if np.isnan(p) else ("$<0.001$" if p < 0.001 else f"{p:.3f}")
        cell = f"${x.mean():+.3f}$ & {(x > 1e-9).sum()}/{(x.abs() <= 1e-9).sum()}/{(x < -1e-9).sum()} & {ps}"
    if prev and prev != m:
        L.append("\\midrule")
    prev = m
    L.append(f"{'GBRS-A' if m == 'GBRS' else m} & {lab} & {len(ds)} & {g.loc[ds, 'R'].mean():.1f} & {g.loc[ds, 'knn3_acc'].mean():.3f} & {g.loc[ds, 'svm_acc'].mean():.3f} & {cell} \\\\")
body = "\n".join(L)
tex = r"""\begin{table*}[pos=!htbp]
\centering
\small
\caption{Sensitivity of the reducts to the parameter of each reducer (mean over datasets). GBRS-A is our implementation of fresh generation with the target rule.}
\label{tab:sens}
\begin{tabular*}{\linewidth}{ll @{\extracolsep{\fill}} rrrrrrr}
\toprule
Reducer & Setting & $N$ & $|R|$ & $k$NN-3 & SVM & $\Delta$ & W/D/L & $p$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular*}
\tabnote{%
Means are over the datasets on which both the setting and the main-table setting return a nonempty reduct: sixteen for GBRS-A and HTBS, fifteen for NRS, fourteen for the fixed radius. $\Delta$ is the change in $k$NN-3 accuracy against the main-table setting of the same reducer; W/D/L counts datasets; $p$ is a two-sided Wilcoxon signed-rank test without correction for multiplicity. HTBS rows with $\nmin$ use a constant minimum ball size and $\alpha=0.10$. For NRS the radius is $c\,\bar d/\sqrt{|A|}$; the fixed radius is $\bar d_{\mathrm{full}}/\sqrt d$ for every subset.}
\end{table*}
"""
Path(sys.argv[1]).write_text(tex)
print("matched GBRS - HTBS at equal n_min (kNN-3, 16 datasets):")
ps = []
for k in (2, 3, 5, 10):
    A = get("GBRS", f"nmin={k}"); B = get("HTBS", f"nmin={k}")
    x = A.knn3_acc - B.knn3_acc.reindex(A.index)
    p = wilcoxon(x).pvalue
    ps.append(p)
    print(k, f"gap {x.mean():+.3f}  W/D/L {(x>1e-9).sum()}/{(x.abs()<=1e-9).sum()}/{(x<-1e-9).sum()}  p={p:.4f}  "
          f"CI95=[{x.mean()-2.131*x.std(ddof=1)/4:.3f},{x.mean()+2.131*x.std(ddof=1)/4:.3f}]")
srt = sorted(ps)
print("Holm-adjusted:", [round(min(1, (len(srt) - i) * v), 4) for i, v in enumerate(srt)])
