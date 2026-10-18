"""Run the released GBRS code (github.com/syxiaa/GBRS, GBRS/Code/GBRS.py) on our datasets.

The released code is imported unchanged. This script
  (A) runs its forward-greedy reduction ``attribute_reduce`` over a grid of the
      minimum ball size 2*(d+2-i), i = 0..d, as in its ``__main__`` block, and
      scores each reduct with our kNN-3 / SVM-RBF five-fold protocol and with the
      5-NN cross-validation that the released code uses to pick the best i;
  (B) audits, with the released ball generator, how often the positive-region
      count falls when one attribute is added to a random subset, at a few values
      of the minimum ball size.
Labels are mapped to 1..k because the released splitter treats label 1 specially.

Usage: python3 run_gbrs_official.py --gbrs /path/to/GBRS.py [--phase A|B] [--datasets ...]
Output: results/gbrs_official_reduct.csv, results/gbrs_official_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("SCIKIT_LEARN_DATA", str(HERE.parent / ".sklearn_cache"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

OUT_A = HERE.parent / "results" / "gbrs_official_reduct.csv"
OUT_B = HERE.parent / "results" / "gbrs_official_audit.csv"
COLS_A = ["dataset", "n", "d", "i", "min_sample", "|R|", "reduct", "runtime_s",
          "cv5nn", "knn3_acc", "svm_acc", "error"]
COLS_B = ["dataset", "n", "d", "i", "min_sample", "subsets", "pairs", "violations",
          "rate", "runtime_s", "error"]
SMALL = ["wine", "sonar", "heart_statlog", "ecoli", "spectf", "ionosphere",
         "dermatology", "breast_cancer", "diabetes", "vehicle"]
LARGE = ["yeast", "pc3", "steel_plates", "segment", "waveform", "digits"]
GBRS_PATH = None


def load_official():
    spec = importlib.util.spec_from_file_location("gbrs_official", GBRS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare(ds):
    from run_experiments import load_dataset
    from run_balanced_metrics import clean_X
    from sklearn.preprocessing import MinMaxScaler
    X, y = load_dataset(ds)
    X = MinMaxScaler().fit_transform(clean_X(ds, X))
    cls = sorted(set(y.tolist()))
    m = {c: k + 1 for k, c in enumerate(cls)}
    yl = np.array([m[c] for c in y], dtype=float)
    D = np.hstack([X, yl.reshape(-1, 1), np.arange(len(y)).reshape(-1, 1)])
    return X, y, D


def grid(d, coarse):
    full = list(range(d + 1))
    if not coarse:
        return full
    pts = sorted({0, d // 4, d // 2, (3 * d) // 4, d})
    return pts


def job_A(args):
    ds, i, coarse = args
    row = {"dataset": ds, "i": i}
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from run_experiments import evaluate_reduct
        G = load_official()
        X, y, D = prepare(ds)
        n, d = X.shape
        row.update(n=n, d=d, min_sample=2 * (D.shape[1] - i))
        t0 = time.perf_counter()
        R = G.attribute_reduce(D, pur=1, d2=i)
        row["runtime_s"] = time.perf_counter() - t0
        R = [int(a) for a in R]
        row["|R|"] = len(R)
        row["reduct"] = " ".join(map(str, R))
        row["cv5nn"] = float(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=5), D[:, R], D[:, -2], cv=5)))
        ev = evaluate_reduct(X, y, R)
        row.update(knn3_acc=ev["knn3_mean"], svm_acc=ev["svm_mean"])
    except Exception as exc:
        row["error"] = str(exc)[:200]
    return row


def pos_count(G, D, attrs, ms):
    gb = G.GBList(D, list(attrs))
    gb.init_granular_balls(purity=1, min_sample=ms)
    return sum(b.num for b in gb.granular_balls if b.purity >= 1)


def job_B(args):
    ds, i, subsets, seed = args
    row = {"dataset": ds, "i": i, "subsets": subsets}
    try:
        G = load_official()
        X, y, D = prepare(ds)
        n, d = X.shape
        ms = 2 * (D.shape[1] - i)
        row.update(n=n, d=d, min_sample=ms)
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        pairs = viol = 0
        for _ in range(subsets):
            size = int(rng.integers(1, d))
            A = sorted(rng.choice(d, size=size, replace=False).tolist())
            base = pos_count(G, D, A, ms)
            for a in range(d):
                if a in A:
                    continue
                pairs += 1
                if pos_count(G, D, A + [a], ms) < base:
                    viol += 1
        row.update(pairs=pairs, violations=viol, rate=100.0 * viol / max(pairs, 1),
                   runtime_s=time.perf_counter() - t0)
    except Exception as exc:
        row["error"] = str(exc)[:200]
    return row


def run(jobs, fn, out, cols, nproc, key):
    done = set()
    if out.exists():
        with out.open() as fh:
            for r in csv.DictReader(fh):
                if not r.get("error"):
                    done.add(key(r))
    todo = [j for j in jobs if key({"dataset": j[0], "i": str(j[1])}) not in done]
    new = not out.exists()
    fh = out.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    if new:
        w.writeheader()
    t0, k = time.perf_counter(), 0
    with mp.get_context("fork").Pool(processes=nproc) as pool:
        for r in pool.imap_unordered(fn, todo):
            w.writerow({c: r.get(c, "") for c in cols})
            fh.flush()
            k += 1
            print(f"[{k}/{len(todo)}] {r['dataset']} i={r['i']} ({time.perf_counter() - t0:.0f}s)"
                  + (f" ERROR {r['error']}" if r.get("error") else ""), flush=True)
    fh.close()


def main():
    global GBRS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbrs", required=True)
    ap.add_argument("--phase", default="A")
    ap.add_argument("--datasets", default=",".join(SMALL))
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--subsets", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    GBRS_PATH = a.gbrs
    dss = a.datasets.split(",")
    if a.phase == "A":
        jobs = []
        for ds in dss:
            from run_experiments import load_dataset
            from run_balanced_metrics import clean_X
            d = clean_X(ds, load_dataset(ds)[0]).shape[1]
            jobs += [(ds, i, a.coarse) for i in grid(d, a.coarse)]
        run(jobs, job_A, OUT_A, COLS_A, a.jobs, lambda r: (r["dataset"], str(r["i"])))
    else:
        jobs = []
        for ds in dss:
            from run_experiments import load_dataset
            from run_balanced_metrics import clean_X
            d = clean_X(ds, load_dataset(ds)[0]).shape[1]
            for i in sorted({0, d // 2, d}):
                jobs.append((ds, i, a.subsets, 42))
        run(jobs, job_B, OUT_B, COLS_B, a.jobs, lambda r: (r["dataset"], str(r["i"])))


if __name__ == "__main__":
    main()
