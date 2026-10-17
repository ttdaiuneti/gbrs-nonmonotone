"""Summarize results/param_sensitivity.csv (and the main-table defaults)."""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

s = pd.read_csv("results/param_sensitivity.csv")
m = pd.read_csv("results/classification_rerun.csv")
m = m[~m.dataset.isin({"iris", "glass", "banknote", "balance_scale"})]
m = m[m.method.isin(["GBRS", "HTBS", "NRS"])].copy()
m["setting"] = m.method.map({"GBRS": "adaptive", "HTBS": "alpha=0.1", "NRS": "scale=1"})
m = m.rename(columns={"|R|": "R"})
s = s.rename(columns={"|R|": "R"})
s["setting"] = s.setting.replace({"alpha=0.1": "alpha=0.1", "scale=1": "scale=1"})
# main-table defaults for GBRS (adaptive) and consistency rows from the sweep
cols = ["dataset", "method", "setting", "R", "knn3_acc", "svm_acc"]
allr = pd.concat([s[cols], m[m.setting == "adaptive"][cols]], ignore_index=True)
# NRS scale=1 and HTBS alpha=0.1 come from the sweep if present, else main table
for meth, st in (("NRS", "scale=1"), ("HTBS", "alpha=0.1")):
    have = set(allr[(allr.method == meth) & (allr.setting == st)].dataset)
    extra = m[(m.method == meth) & (m.setting == st) & (~m.dataset.isin(have))][cols]
    allr = pd.concat([allr, extra], ignore_index=True)
print(allr.groupby(["method", "setting"]).size())
rows = []
for (meth, st), g in allr.groupby(["method", "setting"]):
    empty = int((g.R == 0).sum())
    gg = g[g.R > 0]
    rows.append(dict(method=meth, setting=st, n=len(gg), empty=empty,
                     R=gg.R.mean(), knn=gg.knn3_acc.mean(), svm=gg.svm_acc.mean()))
t = pd.DataFrame(rows)
print(t.round(3).to_string())
t.to_csv("results/param_sensitivity_summary.csv", index=False)
# paired comparisons on the datasets where both have a reduct
def pair(a, b):
    A = allr[(allr.method == a[0]) & (allr.setting == a[1])].set_index("dataset")
    B = allr[(allr.method == b[0]) & (allr.setting == b[1])].set_index("dataset")
    ds = [d for d in A.index if d in B.index and A.loc[d, "R"] > 0 and B.loc[d, "R"] > 0]
    x = A.loc[ds, "knn3_acc"].values - B.loc[ds, "knn3_acc"].values
    r = wilcoxon(A.loc[ds, "knn3_acc"], B.loc[ds, "knn3_acc"]) if np.any(x != 0) else None
    return len(ds), x.mean(), (x > 0).sum(), (x == 0).sum(), (x < 0).sum(), (r.pvalue if r else float("nan"))
for a in [("GBRS", f"nmin={k}") for k in (2, 3, 5, 10)] + [("GBRS", "adaptive")]:
    print(a, "vs HTBS default", [round(v, 4) if isinstance(v, float) else v for v in pair(a, ("HTBS", "alpha=0.1"))])
for a in [("HTBS", f"alpha={x:g}") for x in (0.05, 0.1, 0.2, 0.3)]:
    print(a, "vs GBRS adaptive", [round(v, 4) if isinstance(v, float) else v for v in pair(a, ("GBRS", "adaptive"))])
