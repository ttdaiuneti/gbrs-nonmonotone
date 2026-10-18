"""Score the reducts that the released GBRS pipeline selects, with our protocol.

For each dataset the released pipeline (github.com/syxiaa/GBRS) keeps the reduct
with the best 5-NN cross-validation score over its grid of minimum ball sizes
(first maximum, i ascending, as in its ``__main__`` loop). This script takes that
reduct from results/gbrs_official_reduct.csv and scores it with the same kNN-3 /
SVM-RBF five-fold protocol and imbalance-aware metrics as the other reducers
(``evaluate_reduct`` and ``cv_metrics``). It also records the time of the selected
run and of the whole sweep.

Output: results/gbrs_released_eval.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from run_balanced_metrics import clean_X, cv_metrics  # noqa: E402
from run_experiments import evaluate_reduct, load_dataset  # noqa: E402

RES = HERE.parent / "results"


def main():
    df = pd.read_csv(RES / "gbrs_official_reduct.csv")
    df = df[df["error"].isna()].sort_values(["dataset", "i"])
    rows = []
    for ds, g in df.groupby("dataset"):
        best = g.loc[g["cv5nn"].idxmax()]  # idxmax returns the first maximum in i order
        X, y = load_dataset(ds)
        X = clean_X(ds, X)
        R = [int(a) for a in str(best["reduct"]).split()]
        ev = evaluate_reduct(X, y, R)
        bm = cv_metrics(X, y, R)
        rows.append({
            "dataset": ds, "n": int(best["n"]), "d": int(best["d"]), "grid_size": len(g),
            "i": int(best["i"]), "min_sample": int(best["min_sample"]), "|R|": len(R),
            "reduct": best["reduct"], "cv5nn": best["cv5nn"],
            "knn3_acc": ev["knn3_mean"], "knn3_std": ev["knn3_std"],
            "svm_acc": ev["svm_mean"], "svm_std": ev["svm_std"],
            "knn_bal_acc": bm["bal_acc"], "knn_f1_macro": bm["f1_macro"], "knn_mcc": bm["mcc"],
            "svm_bal_acc": bm["svm_bal"], "svm_f1_macro": bm["svm_f1"], "svm_mcc": bm["svm_mcc"],
            "runtime_selected_s": best["runtime_s"], "runtime_sweep_s": g["runtime_s"].sum(),
            "knn3_grid_min": g["knn3_acc"].min(), "knn3_grid_max": g["knn3_acc"].max(),
        })
    out = RES / "gbrs_released_eval.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(pd.DataFrame(rows)[["dataset", "i", "|R|", "knn3_acc", "svm_acc", "knn_bal_acc",
                              "runtime_selected_s", "runtime_sweep_s"]].round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
