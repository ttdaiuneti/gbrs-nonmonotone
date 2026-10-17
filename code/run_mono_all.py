"""Full monotonicity audit with incremental CSV writes."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gbrs_core import clear_gamma_cache, monotonicity_audit
from run_experiments import DATASETS, RESULTS_DIR, load_dataset

OUT = RESULTS_DIR / "mono_audit.csv"
FIELDS = [
    "dataset", "d", "method", "violation_rate", "violations", "total_pairs",
    "mean_delta", "max_delta", "median_delta",
]


def main():
    rows = []
    for ds_name, _, _ in DATASETS:
        print(f"[mono] {ds_name}", flush=True)
        try:
            X, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  skip: {e}", flush=True)
            continue
        clear_gamma_cache()
        d = X.shape[1]
        n_sub = 100 if d > 20 else 200
        for method in ("gbrs", "deg", "htbs", "nrs"):
            r = monotonicity_audit(X, y, method=method, seed=42, n_subsets=n_sub)
            row = {
                "dataset": ds_name,
                "d": d,
                "method": method,
                "violation_rate": r["violation_rate"],
                "violations": r["violations"],
                "total_pairs": r["total_pairs"],
                "mean_delta": r["mean_delta"],
                "max_delta": r["max_delta"],
                "median_delta": r["median_delta"],
            }
            rows.append(row)
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
