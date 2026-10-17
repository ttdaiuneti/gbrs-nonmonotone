"""HTBS with a fixed minimum ball size (companion to run_param_sensitivity.py).

HTBS uses the same size-adaptive schedule as GBRS in the main tables. This
script pins the minimum ball size to a constant for HTBS, with alpha = 0.10,
so that GBRS and HTBS can be compared under matched settings. Rows are
appended to results/param_sensitivity.csv with method "HTBS" and setting
"nmin=<k>"; ``--resume`` skips rows already present.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_param_sensitivity as rps  # noqa: E402  (sets thread env vars)

NMINS = (2, 3, 5, 10)


def one_job(args):
    ds, k = args
    row = {"dataset": ds, "method": "HTBS", "setting": f"nmin={k}"}
    import gbrs_core
    original = gbrs_core.nmin_value
    try:
        from gbrs_core import clear_gamma_cache, forward_greedy
        from run_experiments import load_dataset, evaluate_reduct
        from run_balanced_metrics import clean_X
        gbrs_core.nmin_value = lambda n_attrs, n_samples: k
        clear_gamma_cache()
        X, y = load_dataset(ds)
        X = clean_X(ds, X)
        row["d"] = X.shape[1]
        t0 = time.perf_counter()
        R, g, _ = forward_greedy(X, y, method="deg", alpha=0.10, htbs=True)
        row["runtime_s"] = time.perf_counter() - t0
        ev = evaluate_reduct(X, y, R)
        row.update({"|R|": len(R), "gamma_star": g,
                    "knn3_acc": ev["knn3_mean"], "knn3_std": ev["knn3_std"],
                    "svm_acc": ev["svm_mean"], "svm_std": ev["svm_std"]})
    except Exception as exc:
        row["error"] = str(exc)[:200]
    finally:
        gbrs_core.nmin_value = original
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    done = set()
    if a.resume and rps.OUT.exists():
        with rps.OUT.open() as fh:
            for r in csv.DictReader(fh):
                if not r.get("error"):
                    done.add((r["dataset"], r["method"], r["setting"]))
    todo = [(ds, k) for ds in rps.COST_ORDER for k in NMINS
            if (ds, "HTBS", f"nmin={k}") not in done]
    fh = rps.OUT.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=rps.COLS)
    t0, n = time.perf_counter(), 0
    with mp.get_context("fork").Pool(processes=a.jobs) as pool:
        for r in pool.imap_unordered(one_job, todo):
            w.writerow({c: r.get(c, "") for c in rps.COLS})
            fh.flush()
            n += 1
            print(f"[{n}/{len(todo)}] {r['dataset']} {r['setting']} ({time.perf_counter() - t0:.0f}s)"
                  + (f"  ERROR {r['error']}" if r.get("error") else ""), flush=True)
    fh.close()


if __name__ == "__main__":
    main()
