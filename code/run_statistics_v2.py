"""Statistics over datasets for the four reducers of the main tables.

Reducers: GBRS (the released pipeline, results/gbrs_released_eval.csv), GBRS-A
(our size-adaptive implementation, method "GBRS" in classification_rerun.csv),
HTBS and NRS (classification_rerun.csv). For each classifier:

  * pairwise two-tailed Wilcoxon signed-rank tests (zero differences dropped),
    Holm correction over the six pairs, mean difference with a t-based 95% CI and
    win/draw/loss counts; pairs with NRS use the datasets where NRS is non-empty;
  * each reducer against the unreduced attribute set (same test, no correction);
  * Friedman test, average ranks (rank 1 = best, ties averaged) and the Nemenyi
    critical difference on the datasets where every reducer is non-empty.

Outputs: results/stats_wilcoxon_holm_v2.csv, stats_vs_full_v2.csv,
stats_ranks_v2.csv, stats_friedman_v2.csv
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results"
M = ["GBRS", "GBRS-A", "HTBS", "NRS"]
EXCL = {"iris", "glass", "banknote", "balance_scale"}
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}


def load():
    c = pd.read_csv(RES / "classification_rerun.csv")
    c = c[~c.dataset.isin(EXCL) & c.method.isin(["GBRS", "HTBS", "NRS"])].copy()
    c["method"] = c["method"].replace({"GBRS": "GBRS-A"})
    e = pd.read_csv(RES / "gbrs_released_eval.csv")
    full = c[c.method == "GBRS-A"].set_index("dataset")[["full_knn3", "full_svm"]]
    e = e.join(full, on="dataset")
    e["method"] = "GBRS"
    cols = ["dataset", "method", "|R|", "knn3_acc", "svm_acc", "full_knn3", "full_svm"]
    return pd.concat([e[cols], c[cols]], ignore_index=True)


def nonempty(df):
    bad = set(df[df["|R|"] == 0].dataset)
    return sorted(set(df.dataset) - bad), sorted(bad)


def paired(a, b):
    diff = a - b
    n = len(diff)
    md = float(diff.mean())
    se = float(diff.std(ddof=1)) / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    # same call as the earlier tables: zeros are dropped by zero_method="wilcox",
    # and scipy then uses the normal approximation when zeros or ties are present
    nz = np.abs(diff) > 1e-12
    W, p = stats.wilcoxon(diff) if nz.any() else (float("nan"), 1.0)
    return dict(n=n, mean_diff=md, ci_lo=md - h, ci_hi=md + h, W=float(W), p=float(p),
                wins=int((diff > 1e-12).sum()), draws=int((np.abs(diff) <= 1e-12).sum()),
                losses=int((diff < -1e-12).sum()))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    df = load()
    wide = {m: df[df.method == m].set_index("dataset") for m in M}
    keep, excl = nonempty(df)
    out_w, out_f, out_r, out_fr = [], [], [], []
    for metric, full in (("knn3_acc", "full_knn3"), ("svm_acc", "full_svm")):
        rows = []
        for A, B in itertools.combinations(M, 2):
            ds = sorted(set(wide[A].index))
            if "NRS" in (A, B):
                ds = [d for d in ds if wide["NRS"].loc[d, "|R|"] > 0]
            r = paired(wide[A].loc[ds, metric].values, wide[B].loc[ds, metric].values)
            rows.append(dict(metric=metric, A=A, B=B, **r))
        adj = holm(np.array([r["p"] for r in rows]))
        for r, a in zip(rows, adj):
            r["p_holm"] = float(a)
        out_w += rows
        for m in M:
            ds = [d for d in wide[m].index if wide[m].loc[d, "|R|"] > 0]
            r = paired(wide[m].loc[ds, metric].values, wide[m].loc[ds, full].values)
            out_f.append(dict(metric=metric, method=m, **r))
        mat = np.array([[wide[m].loc[d, metric] for m in M] for d in keep])
        rk = np.array([stats.rankdata(-row) for row in mat])
        chi2, p = stats.friedmanchisquare(*mat.T)
        k, N = len(M), len(keep)
        cd = Q05[k] * math.sqrt(k * (k + 1) / (6 * N))
        for m, r in zip(M, rk.mean(0)):
            out_r.append(dict(metric=metric, method=m, mean_rank=float(r), n_datasets=N))
        out_fr.append(dict(metric=metric, n_datasets=N, k_methods=k, datasets_excluded=";".join(excl),
                           chi2=float(chi2), p_chi2=float(p), nemenyi_CD_alpha05=cd))
    for name, rows in (("stats_wilcoxon_holm_v2", out_w), ("stats_vs_full_v2", out_f),
                       ("stats_ranks_v2", out_r), ("stats_friedman_v2", out_fr)):
        pd.DataFrame(rows).to_csv(RES / f"{name}.csv", index=False)
        print(name)
        print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
