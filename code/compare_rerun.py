"""Compare the published classification numbers against the cache-patched re-run.

Answers one question: does fixing the ``id(X)`` gamma-cache key and adding the
per-dataset cache reset move the numbers in ``classification.csv``?

Reports, per (dataset, method): change in reduct size, in gamma*, and in kNN-3
and SVM accuracy; then aggregates by method and flags the rows a reader of the
manuscript would notice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
KEY = ["dataset", "method"]


def main() -> None:
    pub = pd.read_csv(RESULTS / "classification_published.csv")
    new = pd.read_csv(RESULTS / "classification_rerun.csv")
    new = new[new.get("error").isna() | (new.get("error") == "")] \
        if "error" in new.columns else new

    m = pub.merge(new, on=KEY, suffixes=("_pub", "_new"), how="inner")
    if m.empty:
        sys.exit("no overlapping (dataset, method) rows")

    for c in ["|R|", "gamma_star", "knn3_acc", "svm_acc"]:
        m[f"d_{c}"] = m[f"{c}_new"] - m[f"{c}_pub"]

    m["moved"] = (m["d_|R|"].abs() > 0) | (m["d_knn3_acc"].abs() > 5e-4) \
        | (m["d_svm_acc"].abs() > 5e-4)

    print(f"compared {len(m)} rows over {m.dataset.nunique()} datasets\n")
    print("=== rows that moved ===")
    moved = m[m.moved]
    if moved.empty:
        print("none — the published classification numbers reproduce exactly")
    else:
        cols = ["dataset", "method", "|R|_pub", "|R|_new",
                "knn3_acc_pub", "knn3_acc_new", "svm_acc_pub", "svm_acc_new"]
        print(moved[cols].round(4).to_string(index=False))

    print("\n=== per-method mean absolute change (percentage points) ===")
    agg = m.groupby("method").agg(
        n=("dataset", "size"),
        mean_abs_dR=("d_|R|", lambda s: np.abs(s).mean()),
        mean_abs_dknn_pp=("d_knn3_acc", lambda s: np.abs(s).mean() * 100),
        mean_abs_dsvm_pp=("d_svm_acc", lambda s: np.abs(s).mean() * 100),
        max_abs_dknn_pp=("d_knn3_acc", lambda s: np.abs(s).max() * 100),
    )
    print(agg.round(4).to_string())

    print("\n=== headline means, published vs re-run ===")
    for tag, df in [("published", pub), ("re-run", new)]:
        sub = df[df.dataset.isin(m.dataset.unique())]
        g = sub.groupby("method")[["knn3_acc", "svm_acc"]].mean() * 100
        print(f"-- {tag}")
        print(g.round(2).to_string())

    m.to_csv(RESULTS / "classification_rerun_diff.csv", index=False)
    print(f"\nwrote {RESULTS / 'classification_rerun_diff.csv'}")


if __name__ == "__main__":
    main()
