"""R1 revision: diagnose SteelPlates leakage + PC3 degenerate metric.

Reproduces every number in results/leakage_diagnosis.md.
Run:  python diagnose_leakage.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "SCIKIT_LEARN_DATA",
    str(Path(__file__).resolve().parents[1] / ".skcache"),
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gbrs_core import forward_greedy  # noqa: E402
from run_experiments import evaluate_reduct, load_dataset  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.svm import SVC  # noqa: E402


def steel_plates() -> None:
    print("=== SteelPlates leakage ===")
    X, y = load_dataset("steel_plates")
    d = X.shape[1]
    bincols = [j for j in range(d) if len(np.unique(X[:, j])) == 2]
    print("binary indicator cols:", ["V%d" % (j + 1) for j in bincols])

    R, gs, _ = forward_greedy(X, y, method="nrs")
    print("NRS reduct:", ["V%d" % (j + 1) for j in R], "gamma*=%.4f" % gs)

    B = X[:, bincols]
    groups: dict = {}
    for i in range(len(y)):
        groups.setdefault(tuple(B[i].tolist()), set()).add(int(y[i]))
    pure = sum(1 for v in groups.values() if len(v) == 1)
    puresamp = sum(1 for i in range(len(y))
                   if len(groups[tuple(B[i].tolist())]) == 1)
    print(f"bit-pattern groups={len(groups)} pure={pure} "
          f"frac_samples_in_pure_group={puresamp/len(y):.4f}")

    ev0 = evaluate_reduct(X, y, R)
    keep = [j for j in range(d) if j not in bincols]
    Xc = X[:, keep]
    Rc, _, _ = forward_greedy(Xc, y, method="nrs")
    ev1 = evaluate_reduct(Xc, y, Rc)
    print(f"WITH indicators : |R|={len(R)} knn3={ev0['knn3_mean']:.4f} "
          f"svm={ev0['svm_mean']:.4f}")
    print(f"WITHOUT (cleaned): |R|={len(Rc)} knn3={ev1['knn3_mean']:.4f} "
          f"svm={ev1['svm_mean']:.4f}")


def pc3() -> None:
    print("\n=== PC3 degenerate metric ===")
    X, y = load_dataset("pc3")
    maj = np.bincount(y).max() / len(y)
    print(f"majority-class fraction={maj:.4f}")
    R, _, _ = forward_greedy(X, y, method="gbrs")
    Xr = X[:, R] if R else X[:, :1]
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    ba = []
    for tr, te in skf.split(Xr, y):
        m = SVC(kernel="rbf").fit(Xr[tr], y[tr])
        ba.append(balanced_accuracy_score(y[te], m.predict(Xr[te])))
    print(f"GBRS |R|={len(R)} SVM balanced_acc={np.mean(ba):.4f}")


if __name__ == "__main__":
    steel_plates()
    pc3()
