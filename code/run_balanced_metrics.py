"""R1 revision: balanced metrics (addresses Major #3).

For each dataset x method, recompute the reduct with forward_greedy, then
evaluate 5-fold CV with accuracy, balanced accuracy, F1-macro and MCC.
Also flags rows where accuracy collapses to the majority-class rate.
SteelPlates is run in a cleaned variant (indicator columns dropped).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "SCIKIT_LEARN_DATA",
    str(Path(__file__).resolve().parents[1] / ".skcache"),
)
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from gbrs_core import forward_greedy  # noqa: E402
from run_experiments import DATASETS, load_dataset  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

RESULTS_DIR = CODE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAPER16 = [
    "wine", "sonar", "heart_statlog", "ecoli", "ionosphere", "dermatology",
    "spectf", "breast_cancer", "diabetes", "vehicle", "yeast", "pc3",
    "waveform", "digits", "steel_plates", "segment",
]
METHODS = [("gbrs", False, "GBRS"), ("deg", True, "HTBS"), ("nrs", False, "NRS")]


def clean_X(name: str, X: np.ndarray) -> np.ndarray:
    """Drop binary one-hot indicator columns for steel_plates (leakage)."""
    if name != "steel_plates":
        return X
    d = X.shape[1]
    keep = [j for j in range(d) if len(np.unique(X[:, j])) > 2]
    return X[:, keep]


def cv_metrics(X, y, reduct, seed=0):
    if not reduct:
        return {k: 0.0 for k in
                ("acc", "bal_acc", "f1_macro", "mcc",
                 "svm_acc", "svm_bal", "svm_f1", "svm_mcc")}
    Xr = X[:, reduct]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    kn = {m: [] for m in ("acc", "bal_acc", "f1_macro", "mcc")}
    sv = {m: [] for m in ("acc", "bal_acc", "f1_macro", "mcc")}
    for tr, te in skf.split(Xr, y):
        for clf, store in ((KNeighborsClassifier(3), kn),
                           (SVC(kernel="rbf", C=1.0, gamma="scale"), sv)):
            clf.fit(Xr[tr], y[tr])
            p = clf.predict(Xr[te])
            store["acc"].append(accuracy_score(y[te], p))
            store["bal_acc"].append(balanced_accuracy_score(y[te], p))
            store["f1_macro"].append(f1_score(y[te], p, average="macro"))
            store["mcc"].append(matthews_corrcoef(y[te], p))
    return {
        "acc": np.mean(kn["acc"]), "bal_acc": np.mean(kn["bal_acc"]),
        "f1_macro": np.mean(kn["f1_macro"]), "mcc": np.mean(kn["mcc"]),
        "svm_acc": np.mean(sv["acc"]), "svm_bal": np.mean(sv["bal_acc"]),
        "svm_f1": np.mean(sv["f1_macro"]), "svm_mcc": np.mean(sv["mcc"]),
    }


def main() -> None:
    rows = []
    for name in PAPER16:
        X, y = load_dataset(name)
        X = clean_X(name, X)
        maj = float(np.bincount(y).max() / len(y))
        for method, htbs, label in METHODS:
            if htbs:
                R, gs, _ = forward_greedy(X, y, method="deg", htbs=True)
            else:
                R, gs, _ = forward_greedy(X, y, method=method)
            m = cv_metrics(X, y, R)
            degenerate = (len(R) > 0 and abs(m["svm_acc"] - maj) < 1e-3
                          and m["svm_bal"] < 0.55)
            rows.append({
                "dataset": name, "method": label, "|R|": len(R),
                "majority": round(maj, 4),
                "knn_acc": round(m["acc"], 4),
                "knn_bal_acc": round(m["bal_acc"], 4),
                "knn_f1_macro": round(m["f1_macro"], 4),
                "knn_mcc": round(m["mcc"], 4),
                "svm_acc": round(m["svm_acc"], 4),
                "svm_bal_acc": round(m["svm_bal"], 4),
                "svm_f1_macro": round(m["svm_f1"], 4),
                "svm_mcc": round(m["svm_mcc"], 4),
                "degenerate_acc": int(degenerate),
                "steel_cleaned": int(name == "steel_plates"),
            })
            print(f"[ok] {name:14s} {label:5s} |R|={len(R):2d} "
                  f"knn_acc={m['acc']:.3f} bal={m['bal_acc']:.3f} "
                  f"mcc={m['mcc']:.3f}", flush=True)
        path = RESULTS_DIR / "classification_balanced.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
