"""Re-run the classification experiments with the patched gamma cache.

Semantics are identical to ``run_experiments.run_classification_experiments``:
the same ``forward_greedy``, ``evaluate_reduct`` and ``full_attr_baseline`` are
called, in the same order, with the same arguments.  The only differences are
operational:

  * datasets are processed in parallel (one worker per dataset, so the cache
    reset at the top of each worker is exactly the per-dataset reset the
    sequential driver performs);
  * results are appended as they land, so an interrupted run keeps its
    finished work;
  * output goes to ``classification_rerun.csv`` rather than overwriting the
    published ``classification.csv``.

Purpose: the published classification numbers were produced by a driver that
never reset the gamma cache between datasets, while the cache key was
``id(X)``.  This script establishes how far, if at all, those numbers move
once the key is content-addressed and the reset is in place.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Keep the OpenML cache inside the repo.  scikit-learn's default is
# ``~/scikit_learn_data``; on a machine where the home directory is not
# readable the ``makedirs(..., exist_ok=True)`` in ``get_data_home`` raises
# FileExistsError rather than succeeding, which surfaces as every dataset
# failing to load.  Pinning it here also makes a run self-contained.
os.environ.setdefault("SCIKIT_LEARN_DATA", str(HERE.parent / ".sklearn_cache"))
Path(os.environ["SCIKIT_LEARN_DATA"]).mkdir(parents=True, exist_ok=True)

RESULTS = HERE.parent / "results"
OUT = RESULTS / "classification_rerun.csv"

COLS = ["dataset", "d", "method", "|R|", "gamma_star", "runtime_s",
        "knn3_acc", "knn3_std", "svm_acc", "svm_std",
        "full_knn3", "full_svm", "error"]

METHODS = [("gbrs", False, "GBRS"),
           ("deg", False, "DEG-GB"),
           ("deg", True, "HTBS"),
           ("nrs", False, "NRS")]


def one_dataset(ds_name: str) -> list:
    from gbrs_core import clear_gamma_cache
    from run_experiments import (load_dataset, forward_greedy,
                                 evaluate_reduct, full_attr_baseline)

    from run_balanced_metrics import clean_X

    clear_gamma_cache()
    try:
        X, y = load_dataset(ds_name)
    except Exception as exc:  # a loader failure must not abort the sweep
        return [{"dataset": ds_name, "method": m[2], "error": str(exc)[:200]}
                for m in METHODS]

    # steel_plates carries binary one-hot indicator columns that leak the
    # label; the published classification.csv reports d = 25 for it (not the
    # loader's 33), so the released numbers were produced on the cleaned
    # matrix.  run_experiments.py as released does not do this — applying it
    # here is what makes the re-run comparable to the published file.
    X = clean_X(ds_name, X)

    d = X.shape[1]
    base = full_attr_baseline(X, y)
    out = []
    for method, htbs, label in METHODS:
        t0 = time.perf_counter()
        try:
            if htbs:
                R, gamma_star, _ = forward_greedy(X, y, method="deg", htbs=True)
            else:
                R, gamma_star, _ = forward_greedy(X, y, method=method)
            ev = evaluate_reduct(X, y, R)
            out.append({"dataset": ds_name, "d": d, "method": label,
                        "|R|": len(R), "gamma_star": gamma_star,
                        "runtime_s": time.perf_counter() - t0,
                        "knn3_acc": ev["knn3_mean"], "knn3_std": ev["knn3_std"],
                        "svm_acc": ev["svm_mean"], "svm_std": ev["svm_std"],
                        "full_knn3": base["knn3_mean"],
                        "full_svm": base["svm_mean"], "error": ""})
        except Exception as exc:
            out.append({"dataset": ds_name, "d": d, "method": label,
                        "error": str(exc)[:200]})
    return out


def main() -> None:
    from run_experiments import DATASETS

    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--datasets", type=str, default="")
    a = ap.parse_args()

    names = ([s.strip() for s in a.datasets.split(",") if s.strip()]
             or [d[0] for d in DATASETS])
    # Heaviest first: the long tail dominates wall-clock under a fixed pool.
    order = {d[0]: i for i, d in enumerate(DATASETS)}
    names.sort(key=lambda n: -order.get(n, 0))

    with OUT.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore").writeheader()

    t0, done = time.perf_counter(), 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=a.jobs) as pool:
        for rows in pool.imap_unordered(one_dataset, names):
            with OUT.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
                for r in rows:
                    w.writerow(r)
            done += 1
            print(f"[{done}/{len(names)}] {rows[0]['dataset']} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"wrote {OUT} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
