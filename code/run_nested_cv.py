"""R1 revision: leakage-free (nested) evaluation of attribute reduction.

The reported protocol in run_experiments.py is

    X, y = load_dataset(ds)          # MinMaxScaler fitted on ALL of X
    R, _, _ = forward_greedy(X, y)   # reduct selected using ALL labels
    ev = evaluate_reduct(X, y, R)    # only NOW is the data split into folds

so both the scaler and the attribute-selection step see the test folds. Reported
accuracies are therefore optimistically biased, and the bias is not uniform
across methods: it grows with the number of attributes selected, and NRS selects
roughly six times as many attributes as GBRS. The very comparison the paper
draws is the one most distorted.

This script re-evaluates each method with the selection INSIDE the resampling
loop:

    for each outer fold:
        fit MinMaxScaler on the TRAIN fold only
        run forward_greedy on the TRAIN fold only
        transform the TEST fold with the train-fitted scaler
        train/score kNN-3 and RBF-SVM on the selected attributes

Both protocols are run on identical folds so the per-dataset gap is a direct
estimate of the optimism. Output: results/nested_cv.csv

Usage:
  python run_nested_cv.py --datasets wine,ecoli,... --jobs 12
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "SCIKIT_LEARN_DATA", str(Path(__file__).resolve().parents[1] / ".skcache")
)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from sklearn.metrics import (  # noqa: E402
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

RESULTS_DIR = CODE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_DIR / "nested_cv.csv"

METHODS = [("gbrs", False, "GBRS"), ("deg", True, "HTBS"), ("nrs", False, "NRS")]
SEED = 0
N_SPLITS = 5


def load_raw(name):
    """Dataset WITHOUT the global MinMaxScaler (scaling belongs inside folds)."""
    from sklearn.datasets import fetch_openml
    from run_experiments import OPENML_MAP

    data = fetch_openml(OPENML_MAP[name], version=1, as_frame=False,
                        parser="auto")
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target)
    if y.dtype.kind in "OUS":
        _, y = np.unique(y, return_inverse=True)
    y = y.astype(np.int64)
    if np.any(np.isnan(X)):
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    # steel_plates carries binary one-hot indicator columns that leak the
    # label; drop them exactly as rerun_classification.py does (d = 25).
    if name == "steel_plates":
        keep = [j for j in range(X.shape[1]) if len(np.unique(X[:, j])) > 2]
        X = X[:, keep]
    return X, y


def _score(Xtr, ytr, Xte, yte, R):
    out = {}
    if not R:
        return None
    A, B = Xtr[:, R], Xte[:, R]
    for tag, clf in (("knn", KNeighborsClassifier(n_neighbors=3)),
                     ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))):
        clf.fit(A, ytr)
        p = clf.predict(B)
        out[f"{tag}_acc"] = accuracy_score(yte, p)
        out[f"{tag}_bal"] = balanced_accuracy_score(yte, p)
        out[f"{tag}_f1"] = f1_score(yte, p, average="macro")
        out[f"{tag}_mcc"] = matthews_corrcoef(yte, p)
    return out


def one_job(args):
    ds, method, htbs, label = args
    from gbrs_core import clear_gamma_cache, forward_greedy

    Xraw, y = load_raw(ds)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.perf_counter()

    # --- Protocol B (leaky, as published): scale + select on everything ------
    clear_gamma_cache()
    Xall = MinMaxScaler().fit_transform(Xraw)
    R_all, _, _ = (forward_greedy(Xall, y, method="deg", htbs=True) if htbs
                   else forward_greedy(Xall, y, method=method))
    leaky, nested, sizes = [], [], []
    for tr, te in skf.split(Xall, y):
        s = _score(Xall[tr], y[tr], Xall[te], y[te], list(R_all))
        if s:
            leaky.append(s)

    # --- Protocol A (nested): scale + select on the TRAIN fold only ----------
    for tr, te in skf.split(Xraw, y):
        sc = MinMaxScaler().fit(Xraw[tr])
        Xtr, Xte = sc.transform(Xraw[tr]), sc.transform(Xraw[te])
        clear_gamma_cache()
        Rf, _, _ = (forward_greedy(Xtr, y[tr], method="deg", htbs=True) if htbs
                    else forward_greedy(Xtr, y[tr], method=method))
        sizes.append(len(Rf))
        s = _score(Xtr, y[tr], Xte, y[te], list(Rf))
        if s:
            nested.append(s)
    clear_gamma_cache()

    def agg(rs, k):
        return float(np.mean([r[k] for r in rs])) if rs else 0.0

    row = {"dataset": ds, "n": int(Xraw.shape[0]), "d": int(Xraw.shape[1]),
           "method": label,
           "R_all": len(R_all), "R_fold_mean": float(np.mean(sizes)),
           "R_fold_min": int(np.min(sizes)), "R_fold_max": int(np.max(sizes)),
           "n_empty_folds": int(N_SPLITS - len(nested)),
           "runtime_s": round(time.perf_counter() - t0, 1)}
    for k in ("knn_acc", "knn_bal", "knn_f1", "knn_mcc",
              "svm_acc", "svm_bal", "svm_f1", "svm_mcc"):
        row[f"leaky_{k}"] = round(agg(leaky, k), 6)
        row[f"nested_{k}"] = round(agg(nested, k), 6)
        row[f"gap_{k}"] = round(row[f"leaky_{k}"] - row[f"nested_{k}"], 6)
    return row


def one_job_safe(args):
    try:
        return one_job(args)
    except Exception as e:
        return {"dataset": args[0], "method": args[3], "error": str(e)[:200]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default=None,
                    help="output CSV (default: results/nested_cv.csv)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    datasets = a.datasets.split(",")
    jobs = [(ds, m, h, lab) for ds in datasets for m, h, lab in METHODS]

    rows, done, t0 = [], 0, time.perf_counter()
    with mp.get_context("fork").Pool(processes=a.jobs) as pool:
        for r in pool.imap_unordered(one_job_safe, jobs):
            rows.append(r)
            done += 1
            print(f"[{done}/{len(jobs)}] {r['dataset']} {r.get('method')} "
                  f"({time.perf_counter() - t0:.0f}s)"
                  + (f"  ERROR {r['error']}" if "error" in r else ""),
                  flush=True)

    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    order = {ds: i for i, ds in enumerate(datasets)}
    rows.sort(key=lambda r: (order.get(r["dataset"], 99), r.get("method", "")))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows) in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
