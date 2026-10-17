"""R1 revision: multi-seed monotonicity audit (addresses Major #1).

Separates STRUCTURAL violations (stable across k-means seeds) from
SEED-DEPENDENT violations (initialization noise). Reports mean +/- std of
violation rate over multiple seeds for each dataset/method.
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
from gbrs_core import clear_gamma_cache, monotonicity_audit  # noqa: E402
from run_experiments import DATASETS, load_dataset  # noqa: E402

RESULTS_DIR = CODE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAPER16 = {
    "wine", "sonar", "heart_statlog", "ecoli", "spectf", "ionosphere",
    "dermatology", "breast_cancer", "diabetes", "vehicle", "waveform",
    "yeast", "pc3", "digits", "steel_plates", "segment",
}

SEEDS = [42, 0, 1, 2, 3]
# The CONTESTED numbers (Major #1) are the GBRS/DEG violation rates, so those
# get the full 5-seed treatment on all 16 datasets. HTBS is 0% by construction
# and NRS is ~0%; re-running those 5x over high-d datasets rebuilds balls
# thousands of times for no scientific payoff. We therefore run HTBS/NRS
# multi-seed only on the smaller datasets and rely on the single-seed
# mono_audit.csv for the rest (they are essentially seed-invariant anyway).
HEAVY_METHOD_MAX_N = 400   # nrs/htbs multiseed only where n <= this
HEAVY_METHOD_MAX_D = 20    # and d <= this


def subsets_for_n(n: int) -> int:
    """Scale subset count so large-n datasets stay feasible over 5 seeds.

    gamma is DETERMINISTIC (class-centroid k-means init), so the seed only
    varies which random subsets are drawn; fewer subsets per seed still yields
    a valid mean +/- std of the violation-rate estimator.
    """
    if n <= 400:
        return 200
    if n <= 1000:
        return 120
    if n <= 2000:
        return 60
    return 25


def main() -> None:
    rows = []
    for ds_name, _, _ in DATASETS:
        if ds_name not in PAPER16:
            continue
        X, y = load_dataset(ds_name)
        n, d = X.shape
        ns = subsets_for_n(n)
        for method in ("gbrs", "deg", "htbs", "nrs"):
            if method in ("nrs", "htbs") and (n > HEAVY_METHOD_MAX_N
                                              or d > HEAVY_METHOD_MAX_D):
                # record that this cell is intentionally skipped (too expensive)
                rows.append({
                    "dataset": ds_name, "d": d, "n": n, "method": method,
                    "n_subsets": ns, "n_seeds": 0, "rate_mean": "",
                    "rate_std": "", "rate_min": "", "rate_max": "",
                    "note": "skipped_large_n",
                })
                print(f"[skip] {ds_name} {method} (n={n})", flush=True)
                continue
            rates = []
            for seed in SEEDS:
                clear_gamma_cache()
                res = monotonicity_audit(
                    X, y, method=method, seed=seed, n_subsets=ns
                )
                rates.append(res["violation_rate"])
            rates = np.array(rates)
            rows.append({
                "dataset": ds_name, "d": d, "n": n, "method": method,
                "n_subsets": ns, "n_seeds": len(SEEDS),
                "rate_mean": round(float(rates.mean()), 4),
                "rate_std": round(float(rates.std()), 4),
                "rate_min": round(float(rates.min()), 4),
                "rate_max": round(float(rates.max()), 4),
                "note": "",
            })
            print(f"[ok] {ds_name:14s} {method:5s} "
                  f"{rates.mean():.2f} +/- {rates.std():.2f} "
                  f"(min {rates.min():.2f} max {rates.max():.2f})", flush=True)
        # incremental save
        path = RESULTS_DIR / "mono_audit_multiseed.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
