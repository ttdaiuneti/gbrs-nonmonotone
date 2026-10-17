"""Complete missing datasets and re-run mono audit."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import (
    DATASETS,
    RESULTS_DIR,
    evaluate_reduct,
    full_attr_baseline,
    load_dataset,
    run_mono_experiments,
)
from gbrs_core import clear_gamma_cache, forward_greedy

MISSING = ["breast_cancer", "segment", "steel_plates"]
EXTRA_DIGITS_NRS = True


def append_rows():
    path = RESULTS_DIR / "classification.csv"
    existing = set()
    if path.exists():
        with path.open() as f:
            for r in csv.DictReader(f):
                existing.add((r["dataset"], r["method"]))
    rows = []
    if path.exists():
        with path.open() as f:
            rows = list(csv.DictReader(f))

    targets = list(MISSING)
    if EXTRA_DIGITS_NRS and ("digits", "NRS") not in existing:
        targets.append("digits_nrs_only")

    methods = [
        ("gbrs", False, "GBRS"),
        ("deg", False, "DEG-GB"),
        ("deg", True, "HTBS"),
        ("nrs", False, "NRS"),
    ]

    for ds_name in targets:
        if ds_name == "digits_nrs_only":
            ds_name = "digits"
            methods_run = [("nrs", False, "NRS")]
        else:
            methods_run = methods

        print(f"Supplement {ds_name}")
        X, y = load_dataset(ds_name)
        d = X.shape[1]
        base = full_attr_baseline(X, y)
        for method, htbs, label in methods_run:
            if (ds_name, label) in existing:
                continue
            if htbs:
                R, gamma_star, runtime = forward_greedy(X, y, method="deg", htbs=True)
            else:
                R, gamma_star, runtime = forward_greedy(X, y, method=method)
            ev = evaluate_reduct(X, y, R)
            row = {
                "dataset": ds_name,
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
            }
            rows.append(row)
            existing.add((ds_name, label))
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
    print(f"Updated {path}")


if __name__ == "__main__":
    append_rows()
    run_mono_experiments(seeds=[42, 0, 1, 2, 3])
