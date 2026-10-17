"""Re-run breast_cancer (WDBC, d=30) only."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import RESULTS_DIR, evaluate_reduct, full_attr_baseline, load_dataset
from gbrs_core import forward_greedy

path = RESULTS_DIR / "classification.csv"
rows = [r for r in csv.DictReader(path.open()) if r["dataset"] != "breast_cancer"]

X, y = load_dataset("breast_cancer")
d = X.shape[1]
base = full_attr_baseline(X, y)
for method, htbs, label in [
    ("gbrs", False, "GBRS"),
    ("deg", False, "DEG-GB"),
    ("deg", True, "HTBS"),
    ("nrs", False, "NRS"),
]:
    if htbs:
        R, gamma_star, runtime = forward_greedy(X, y, method="deg", htbs=True)
    else:
        R, gamma_star, runtime = forward_greedy(X, y, method=method)
    ev = evaluate_reduct(X, y, R)
    rows.append({
        "dataset": "breast_cancer",
        "d": d,
        "method": label,
        "|R|": len(R),
        "gamma_star": gamma_star,
        "runtime_s": runtime,
        "knn3_acc": ev["knn3_mean"],
        "knn3_std": ev["knn3_std"],
        "svm_acc": ev["svm_mean"],
        "svm_std": ev["svm_std"],
        "full_knn3": base["knn3_mean"],
        "full_svm": base["svm_mean"],
    })

with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"Updated breast_cancer (d={d}) in {path}")
