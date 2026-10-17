"""Sensitivity of the accuracy of the reducts to the parameters of each reducer.

For every dataset of the main study this script reruns the forward-greedy
reduction under a grid of parameter values and scores each reduct with the
same kNN-3 / SVM-RBF five-fold protocol as ``rerun_classification.py``:

  * GBRS : fixed minimum ball size  n_min in {2, 3, 5, 10}
           (the main tables use the size-adaptive schedule);
  * HTBS : entropy fraction alpha in {0.05, 0.10, 0.20, 0.30}
           (the main tables use alpha = 0.10);
  * NRS  : radius delta = c * dbar / sqrt(|A|) with c in {0.5, 1, 2}, and a
           fixed radius dbar_full / sqrt(d) that does not depend on |A|
           (the main tables use c = 1).

Output: results/param_sensitivity.csv (one row per dataset and setting).
Each row is appended as soon as it finishes, so an interrupted run keeps its
finished work and ``--resume`` skips rows already present.
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("SCIKIT_LEARN_DATA", str(HERE.parent / ".sklearn_cache"))
Path(os.environ["SCIKIT_LEARN_DATA"]).mkdir(parents=True, exist_ok=True)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

OUT = HERE.parent / "results" / "param_sensitivity.csv"
COLS = ["dataset", "d", "method", "setting", "|R|", "gamma_star", "runtime_s",
        "knn3_acc", "knn3_std", "svm_acc", "svm_std", "error"]

DATASETS = ["ecoli", "diabetes", "yeast", "wine", "heart_statlog", "vehicle",
            "breast_cancer", "segment", "steel_plates", "ionosphere",
            "dermatology", "pc3", "waveform", "spectf", "sonar", "digits"]
COST_ORDER = ["digits", "waveform", "pc3", "spectf", "sonar", "segment",
              "ionosphere", "dermatology", "steel_plates", "breast_cancer",
              "vehicle", "yeast", "heart_statlog", "wine", "diabetes", "ecoli"]

SETTINGS = (
    [("GBRS", f"nmin={m}", ("gbrs", m)) for m in (2, 3, 5, 10)]
    + [("HTBS", f"alpha={a:g}", ("htbs", a)) for a in (0.05, 0.10, 0.20, 0.30)]
    + [("NRS", f"scale={c:g}", ("nrs", c)) for c in (0.5, 1.0, 2.0)]
    + [("NRS", "fixed", ("nrs", None))]
)


def greedy_gbrs_fixed_nmin(X, y, nmin):
    """Forward greedy of Algorithm 1 with the minimum ball size pinned."""
    from gbrs_core import compute_gamma
    d = X.shape[1]
    allp = list(range(d))
    g_star = compute_gamma(X, y, allp, "gbrs", 0.1, nmin_override=nmin)
    R, g_cur, rest = [], 0.0, set(allp)
    while g_cur < g_star - 1e-10 and rest:
        best_a, best_g = None, g_cur
        for a in rest:
            g = compute_gamma(X, y, R + [a], "gbrs", 0.1, nmin_override=nmin)
            if g > best_g + 1e-10:
                best_g, best_a = g, a
        if best_a is None:
            break
        R.append(best_a)
        rest.remove(best_a)
        g_cur = best_g
    return R, g_star


def _nrs_gamma(X, y, attrs, scale, fixed_delta):
    if not attrs:
        return 0.0
    Xa = X[:, list(attrs)]
    n = len(y)
    D = np.sqrt(((Xa[:, None, :] - Xa[None, :, :]) ** 2).sum(2))
    if fixed_delta is None:
        off = D[np.triu_indices(n, k=1)]
        delta = scale * float(off.mean()) / math.sqrt(len(attrs))
    else:
        delta = fixed_delta
    pos = 0
    for i in range(n):
        m = D[i] <= delta + 1e-12
        if m.any() and np.all(y[m] == y[i]):
            pos += 1
    return pos / n


def greedy_nrs(X, y, scale):
    """Forward greedy for NRS with radius c*dbar/sqrt(|A|), or a fixed radius."""
    d = X.shape[1]
    allp = list(range(d))
    fixed = None
    if scale is None:
        n = len(y)
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
        fixed = float(D[np.triu_indices(n, k=1)].mean()) / math.sqrt(d)
    sc = 1.0 if scale is None else scale
    g_star = _nrs_gamma(X, y, allp, sc, fixed)
    R, g_cur, rest = [], 0.0, set(allp)
    while g_cur < g_star - 1e-10 and rest:
        best_a, best_g = None, g_cur
        for a in rest:
            g = _nrs_gamma(X, y, R + [a], sc, fixed)
            if g > best_g + 1e-10:
                best_g, best_a = g, a
        if best_a is None:
            break
        R.append(best_a)
        rest.remove(best_a)
        g_cur = best_g
    return R, g_star


def one_job(args):
    ds, (method, setting, spec) = args
    row = {"dataset": ds, "method": method, "setting": setting}
    try:
        from gbrs_core import clear_gamma_cache, forward_greedy
        from run_experiments import load_dataset, evaluate_reduct
        from run_balanced_metrics import clean_X
        clear_gamma_cache()
        X, y = load_dataset(ds)
        X = clean_X(ds, X)
        row["d"] = X.shape[1]
        t0 = time.perf_counter()
        kind, p = spec
        if kind == "gbrs":
            R, g = greedy_gbrs_fixed_nmin(X, y, p)
        elif kind == "htbs":
            R, g, _ = forward_greedy(X, y, method="deg", alpha=p, htbs=True)
        else:
            R, g = greedy_nrs(X, y, p)
        row["runtime_s"] = time.perf_counter() - t0
        ev = evaluate_reduct(X, y, R)
        row.update({"|R|": len(R), "gamma_star": g,
                    "knn3_acc": ev["knn3_mean"], "knn3_std": ev["knn3_std"],
                    "svm_acc": ev["svm_mean"], "svm_std": ev["svm_std"]})
    except Exception as exc:  # keep the sweep going
        row["error"] = str(exc)[:200]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--datasets", default=",".join(COST_ORDER))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    done = set()
    if a.resume and OUT.exists():
        with OUT.open() as fh:
            for r in csv.DictReader(fh):
                if not r.get("error"):
                    done.add((r["dataset"], r["method"], r["setting"]))
    todo = [(ds, s) for ds in a.datasets.split(",") for s in SETTINGS
            if (ds, s[0], s[1]) not in done]
    new = not OUT.exists() or not a.resume
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT.open("a" if a.resume else "w", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()
    t0, n = time.perf_counter(), 0
    with mp.get_context("fork").Pool(processes=a.jobs) as pool:
        for r in pool.imap_unordered(one_job, todo):
            w.writerow({k: r.get(k, "") for k in COLS})
            fh.flush()
            n += 1
            print(f"[{n}/{len(todo)}] {r['dataset']} {r['method']} {r['setting']}"
                  f" ({time.perf_counter() - t0:.0f}s)"
                  + (f"  ERROR {r['error']}" if r.get("error") else ""),
                  flush=True)
    fh.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
