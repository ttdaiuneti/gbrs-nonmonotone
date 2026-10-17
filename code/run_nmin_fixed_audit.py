"""R1 revision: monotonicity audit with the ball-splitting floor HELD FIXED.

The manuscript's audit computes gamma(A) and gamma(A + {a}) with
    nmin = nmin_value(|A|, n) = max(2, min(|A| + 1, n / 10)),
i.e. a stopping rule that GROWS with the size of the attribute subset. The two
dependency values being compared are therefore produced under two different
stopping rules, and the larger subset is forbidden from splitting as deeply --
which on its own can lower gamma and be recorded as a monotonicity violation.

This is the exact confound that `run_nrs_fixed_delta.py` already controls for on
the NRS side (by holding the neighbourhood radius delta constant). This script
supplies the missing symmetric control for the granular-ball side: it re-runs the
audit unchanged except that nmin is pinned to a constant.

Outputs results/nmin_fixed_audit.csv with one row per
(dataset, method, config, seed).

Usage:  python run_nmin_fixed_audit.py [--seeds 42,0,1,2,3] [--jobs 12]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault(
    "SCIKIT_LEARN_DATA", str(Path(__file__).resolve().parents[1] / ".skcache")
)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

RESULTS_DIR = CODE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_DIR / "nmin_fixed_audit.csv"

PAPER16 = [
    "wine", "sonar", "heart_statlog", "ecoli", "spectf", "ionosphere",
    "dermatology", "breast_cancer", "diabetes", "vehicle", "waveform",
    "yeast", "pc3", "digits", "steel_plates", "segment",
]
METHODS = ["gbrs", "deg"]
# config label -> nmin_override (None = the manuscript's |A|-dependent schedule)
CONFIGS = {"scaled": None, "nmin2": 2, "nmin5": 5}


def n_subsets_for(d: int) -> int:
    """Match run_mono_all.py exactly."""
    return 100 if d > 20 else 200


def one_job(args):
    ds, method, cfg, seed = args
    from gbrs_core import clear_gamma_cache, monotonicity_audit
    from run_experiments import load_dataset

    X, y = load_dataset(ds)
    clear_gamma_cache()
    t0 = time.perf_counter()
    r = monotonicity_audit(
        X, y, method=method, seed=seed,
        n_subsets=n_subsets_for(X.shape[1]),
        nmin_override=CONFIGS[cfg],
    )
    clear_gamma_cache()
    return {
        "dataset": ds, "d": int(X.shape[1]), "n": int(X.shape[0]),
        "method": method, "config": cfg,
        "nmin_override": "" if CONFIGS[cfg] is None else CONFIGS[cfg],
        "seed": seed,
        "violation_rate": round(r["violation_rate"], 4),
        "violations": r["violations"], "total_pairs": r["total_pairs"],
        "mean_delta": round(r["mean_delta"], 6),
        "max_delta": round(r["max_delta"], 6),
        "runtime_s": round(time.perf_counter() - t0, 1),
    }


def one_job_safe(args):
    """Never let a single dataset failure abort the whole sweep."""
    ds, method, cfg, seed = args
    try:
        return one_job(args)
    except Exception as e:
        return {"dataset": ds, "method": method, "config": cfg, "seed": seed,
                "violation_rate": "", "runtime_s": "", "error": str(e)[:200]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--datasets", default=",".join(PAPER16))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    datasets = a.datasets.split(",")

    jobs = [(ds, m, c, s) for ds in datasets for m in METHODS
            for c in CONFIGS for s in seeds]
    # heaviest first so the long tail does not strand idle cores
    order = {ds: i for i, ds in enumerate(PAPER16)}
    jobs.sort(key=lambda j: -order.get(j[0], 0))

    cols = ["dataset", "n", "d", "method", "config", "nmin_override", "seed",
            "violation_rate", "violations", "total_pairs", "mean_delta",
            "max_delta", "runtime_s", "error"]
    # Append each result as it lands: a long sweep must never lose finished work
    # if it is interrupted.
    part = OUT.with_suffix(".partial.csv")
    with part.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore").writeheader()

    rows, done, t0 = [], 0, time.perf_counter()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=a.jobs) as pool:
        for r in pool.imap_unordered(one_job_safe, jobs):
            rows.append(r)
            with part.open("a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=cols,
                               extrasaction="ignore").writerow(r)
            done += 1
            print(f"[{done}/{len(jobs)}] {r['dataset']} {r['method']} "
                  f"{r['config']} seed={r['seed']} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: (order.get(r["dataset"], 99), r["method"],
                             r["config"], r["seed"]))
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows) in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
