"""Nested evaluation of the released GBRS pipeline (github.com/syxiaa/GBRS).

The released code runs ``attribute_reduce`` for every minimum ball size
2*(d+2-i), i = 0..d, and keeps the reduct with the best 5-NN cross-validation
accuracy on the full labelled data. Here the whole pipeline (the sweep over i, the
5-NN selection and the reduction) is run inside the training part of each outer
stratified fold, and the chosen reduct is scored on the held-out part with kNN-3
and SVM-RBF.

Usage: python3 run_gbrs_official_nested.py --gbrs /path/to/GBRS.py [--datasets ...]
Output: results/gbrs_official_nested.csv (one row per dataset and outer fold).
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
import run_gbrs_official as base  # noqa: E402  (sets thread env vars, loader)

import numpy as np  # noqa: E402

OUT = HERE.parent / "results" / "gbrs_official_nested.csv"
COLS = ["dataset", "fold", "n_train", "d", "best_i", "min_sample", "|R|", "cv5nn",
        "knn3_acc", "svm_acc", "runtime_s", "error"]
GRID_FRAC = None


def job(args):
    ds, fold = args
    row = {"dataset": ds, "fold": fold}
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        G = base.load_official()
        X, y, D = base.prepare(ds)
        n, d = X.shape
        tr, te = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))[fold]
        Dtr = D[tr].copy()
        Dtr[:, -1] = np.arange(len(tr))
        row.update(n_train=len(tr), d=d)
        t0 = time.perf_counter()
        best = None
        for i in base.grid(d, False):
            R = [int(a) for a in G.attribute_reduce(Dtr, pur=1, d2=i)]
            sc = float(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=5),
                                               Dtr[:, R], Dtr[:, -2], cv=5)))
            if best is None or sc > best[0]:
                best = (sc, i, R)
        sc, i, R = best
        row.update(best_i=i, min_sample=2 * (D.shape[1] - i), cv5nn=sc)
        row["|R|"] = len(R)
        k = KNeighborsClassifier(n_neighbors=3).fit(X[tr][:, R], y[tr])
        row["knn3_acc"] = float(k.score(X[te][:, R], y[te]))
        s = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X[tr][:, R], y[tr])
        row["svm_acc"] = float(s.score(X[te][:, R], y[te]))
        row["runtime_s"] = time.perf_counter() - t0
    except Exception as exc:
        row["error"] = str(exc)[:200]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbrs", required=True)
    ap.add_argument("--datasets", default=",".join(base.SMALL))
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    base.GBRS_PATH = a.gbrs
    done = set()
    if OUT.exists():
        with OUT.open() as fh:
            for r in csv.DictReader(fh):
                if not r.get("error"):
                    done.add((r["dataset"], r["fold"]))
    todo = [(ds, f) for ds in a.datasets.split(",") for f in range(5) if (ds, str(f)) not in done]
    new = not OUT.exists()
    fh = OUT.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()
    t0, k = time.perf_counter(), 0
    with mp.get_context("fork").Pool(processes=a.jobs) as pool:
        for r in pool.imap_unordered(job, todo):
            w.writerow({c: r.get(c, "") for c in COLS})
            fh.flush()
            k += 1
            print(f"[{k}/{len(todo)}] {r['dataset']} fold {r['fold']} ({time.perf_counter() - t0:.0f}s)"
                  + (f" ERROR {r['error']}" if r.get("error") else ""), flush=True)
    fh.close()


if __name__ == "__main__":
    main()
