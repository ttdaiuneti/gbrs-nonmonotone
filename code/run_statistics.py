"""Regenerate every statistical test in the paper from the per-dataset results.

This script exists because ``results/friedman_nemenyi.csv`` could not be traced
to any code in this repository and its statistic could not be reproduced from
any subset of the released per-dataset numbers.  Every number the manuscript
reports as a test result must come from a script that a reader can run; this is
that script.

Design decisions, all of which the manuscript must state explicitly:

  * **Inclusion rule.**  A dataset enters the comparison only if *all* compared
    methods returned a non-empty reduct on it.  A method that returns |R| = 0
    has not produced a classifier, so its accuracy is not a comparable
    measurement.  Averaging over different denominators per method (which the
    published tables do) compares different things.
  * **Friedman + Nemenyi.**  Reported with the exact n used, so the critical
    difference is checkable by hand.
  * **Pairwise Wilcoxon.**  Holm-corrected across the family of pairwise
    comparisons.  Uncorrected pairwise p-values after an omnibus test inflate
    the family-wise error rate.

Outputs (all in ``results/``):
  ``stats_friedman.csv``, ``stats_wilcoxon_holm.csv``, ``stats_ranks.csv``.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parent.parent / "results"

# The sixteen datasets the manuscript reports on (iris, glass, balance_scale
# and banknote are run but excluded from the paper's tables).
PAPER16 = ["wine", "sonar", "heart_statlog", "ecoli", "spectf", "ionosphere",
           "dermatology", "breast_cancer", "diabetes", "vehicle", "waveform",
           "yeast", "pc3", "digits", "steel_plates", "segment"]

METHODS = ["GBRS", "HTBS", "NRS"]


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference for the Nemenyi post-hoc test.

    q_alpha values are the Studentised range statistic divided by sqrt(2),
    tabulated for alpha = 0.05 (Demsar 2006, Table 5).
    """
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
           7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    if k not in q05:
        raise ValueError(f"no tabulated q_alpha for k={k}")
    return q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))


def build_matrix(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, list]:
    """Wide matrix over datasets where every method has a non-empty reduct."""
    sub = df[df.dataset.isin(PAPER16) & df.method.isin(METHODS)]
    nonempty = sub[sub["|R|"] > 0]
    counts = nonempty.groupby("dataset").method.nunique()
    keep = sorted(counts[counts == len(METHODS)].index)
    wide = (nonempty[nonempty.dataset.isin(keep)]
            .pivot(index="dataset", columns="method", values=metric)[METHODS])
    dropped = sorted(set(sub.dataset.unique()) - set(keep))
    return wide, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="classification.csv",
                    help="per-dataset results file inside results/")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    a = ap.parse_args()

    df = pd.read_csv(RESULTS / a.source)
    fried_rows, wil_rows, rank_rows = [], [], []

    for metric in ["knn3_acc", "svm_acc"]:
        wide, dropped = build_matrix(df, metric)
        n, k = wide.shape
        chi2, p = stats.friedmanchisquare(*[wide[m].values for m in METHODS])
        # Iman-Davenport F, which is less conservative than the chi-square form
        f_stat = (n - 1) * chi2 / (n * (k - 1) - chi2)
        p_f = 1 - stats.f.cdf(f_stat, k - 1, (k - 1) * (n - 1))
        cd = nemenyi_cd(k, n)
        # rank 1 = best, so rank the negated accuracies
        ranks = wide.apply(lambda r: stats.rankdata(-r.values), axis=1,
                           result_type="expand")
        ranks.columns = METHODS
        mean_ranks = ranks.mean()

        fried_rows.append({"metric": metric, "n_datasets": n, "k_methods": k,
                           "datasets_used": ";".join(wide.index),
                           "datasets_excluded": ";".join(dropped),
                           "chi2": chi2, "p_chi2": p,
                           "iman_davenport_F": f_stat, "p_F": p_f,
                           "nemenyi_CD_alpha05": cd})
        for m in METHODS:
            rank_rows.append({"metric": metric, "method": m,
                              "mean_rank": mean_ranks[m], "n_datasets": n})

        pairs = list(itertools.combinations(METHODS, 2))
        raw = []
        for m1, m2 in pairs:
            st, pv = stats.wilcoxon(wide[m1].values, wide[m2].values)
            raw.append({"metric": metric, "pair": f"{m1} vs {m2}",
                        "W": st, "p_raw": pv,
                        "mean_diff_pp": (wide[m1] - wide[m2]).mean() * 100,
                        "rank_diff": mean_ranks[m1] - mean_ranks[m2],
                        "exceeds_CD": abs(mean_ranks[m1] - mean_ranks[m2]) > cd})
        # Holm across the pairwise family, within this metric
        raw.sort(key=lambda r: r["p_raw"])
        mtot, running = len(raw), 0.0
        for i, r in enumerate(raw):
            adj = min(1.0, (mtot - i) * r["p_raw"])
            running = max(running, adj)  # Holm is monotone non-decreasing
            r["p_holm"] = running
            r["signif_holm_05"] = running < 0.05
        wil_rows.extend(raw)

    tag = f"_{a.tag}" if a.tag else ""
    pd.DataFrame(fried_rows).to_csv(RESULTS / f"stats_friedman{tag}.csv", index=False)
    pd.DataFrame(wil_rows).to_csv(RESULTS / f"stats_wilcoxon_holm{tag}.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(RESULTS / f"stats_ranks{tag}.csv", index=False)

    show = pd.DataFrame(fried_rows).drop(columns=["datasets_used", "datasets_excluded"])
    print("=== Friedman ===")
    print(show.round(4).to_string(index=False))
    print("\nexcluded (a method returned an empty reduct):",
          pd.DataFrame(fried_rows).datasets_excluded.iloc[0] or "none")
    print("\n=== mean ranks (1 = best) ===")
    print(pd.DataFrame(rank_rows).pivot(index="metric", columns="method",
                                        values="mean_rank").round(3).to_string())
    print("\n=== pairwise Wilcoxon, Holm-corrected ===")
    print(pd.DataFrame(wil_rows)[["metric", "pair", "W", "p_raw", "p_holm",
                                  "signif_holm_05", "mean_diff_pp",
                                  "exceeds_CD"]].round(5).to_string(index=False))
    # CD diagram is regenerated from these rank files by generate_cd_diagram.py.


if __name__ == "__main__":
    main()
