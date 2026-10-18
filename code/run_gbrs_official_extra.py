"""Further runs of the released GBRS code (github.com/syxiaa/GBRS, GBRS/Code/GBRS.py).

The released code is imported unchanged (see run_gbrs_official.py). For each
dataset the minimum ball size is the one that the released pipeline selects in
results/gbrs_official_reduct.csv (first i with the best 5-NN score, i ascending,
as in its ``__main__`` loop). Three kinds of job:

  audit : random-subset audit of the positive-region count with the released ball
          generator; subsets are drawn exactly as in gbrs_core.monotonicity_audit
          (seed 42, 200 subsets when d <= 20 and 100 otherwise);
  path  : the greedy loop of ``attribute_reduce`` re-run step by step, recording at
          every step how many remaining attributes lower the positive count; the
          final reduct is checked against the released function's output;
  nest  : one (outer fold, i) cell of the nested evaluation of the six larger
          datasets (coarse grid of i, as in phase A); the per-fold choice of i is
          made afterwards from the training-fold 5-NN scores.

Usage: python3 run_gbrs_official_extra.py --gbrs /path/to/GBRS.py [--kinds audit,path,nest]
Output: results/gbrs_official_audit_full.csv, results/gbrs_official_path.csv,
        results/gbrs_official_nested_grid.csv
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
import run_gbrs_official as base  # noqa: E402  (thread env vars, loader, data)

import numpy as np  # noqa: E402

RES = HERE.parent / "results"
OUT = {"audit": RES / "gbrs_official_audit_full.csv",
       "path": RES / "gbrs_official_path.csv",
       "nest": RES / "gbrs_official_nested_grid.csv"}
COLS = {"audit": ["dataset", "n", "d", "i", "min_sample", "chunk", "subsets", "pairs",
                  "violations", "sum_drop", "max_drop", "runtime_s", "error"],
        "path": ["dataset", "n", "d", "i", "min_sample", "step", "added", "pos_before",
                 "pos_best", "candidates", "lowering", "reduct_matches", "runtime_s", "error"],
        "nest": ["dataset", "fold", "n_train", "d", "i", "min_sample", "|R|", "cv5nn",
                 "knn3_acc", "svm_acc", "runtime_s", "error"]}
CHUNK = 10
ALL = base.SMALL + base.LARGE


def selected_i():
    """i chosen by the released pipeline on the full data (first maximum, i ascending)."""
    best = {}
    with (RES / "gbrs_official_reduct.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh) if not r.get("error")]
    for r in sorted(rows, key=lambda r: (r["dataset"], int(r["i"]))):
        sc = float(r["cv5nn"])
        if r["dataset"] not in best or sc > best[r["dataset"]][0]:
            best[r["dataset"]] = (sc, int(r["i"]), r["reduct"])
    return {k: (v[1], v[2]) for k, v in best.items()}


def subsets_for(d):
    rng = np.random.default_rng(42)
    out = []
    for _ in range(100 if d > 20 else 200):
        size = int(rng.integers(1, d))
        out.append(sorted(rng.choice(d, size=size, replace=False).tolist()))
    return out


def job_audit(args):
    ds, i, chunk = args
    row = {"dataset": ds, "i": i, "chunk": chunk}
    try:
        G = base.load_official()
        X, y, D = base.prepare(ds)
        n, d = X.shape
        ms = 2 * (D.shape[1] - i)
        subs = subsets_for(d)[chunk * CHUNK:(chunk + 1) * CHUNK]
        row.update(n=n, d=d, min_sample=ms, subsets=len(subs))
        t0 = time.perf_counter()
        pairs = viol = 0
        sdrop = mdrop = 0.0
        for A in subs:
            b = base.pos_count(G, D, A, ms)
            for a in range(d):
                if a in A:
                    continue
                pairs += 1
                e = base.pos_count(G, D, sorted(A + [a]), ms)
                if e < b:
                    viol += 1
                    sdrop += (b - e) / n
                    mdrop = max(mdrop, (b - e) / n)
        row.update(pairs=pairs, violations=viol, sum_drop=sdrop, max_drop=mdrop,
                   runtime_s=time.perf_counter() - t0)
    except Exception as exc:
        row["error"] = str(exc)[:200]
    return [row]


def job_path(args):
    """Same loop as attribute_reduce, with the counts of every candidate recorded."""
    ds, i, reduct = args
    rows = []
    try:
        G = base.load_official()
        X, y, D = base.prepare(ds)
        n, d = X.shape
        ms = 2 * (D.shape[1] - i)
        t0 = time.perf_counter()
        attribu, rest, bal_num, step = [], list(range(d)), -9999, 0
        while rest:
            counts = {a: base.pos_count(G, D, attribu + [a], ms) for a in rest}
            best_a, best_c = -1, -9999
            for a in rest:  # first maximum in the released iteration order
                if counts[a] > best_c:
                    best_a, best_c = a, counts[a]
            step += 1
            rows.append({"dataset": ds, "n": n, "d": d, "i": i, "min_sample": ms, "step": step,
                         "added": best_a if best_c >= bal_num else "",
                         "pos_before": bal_num if bal_num >= 0 else "",
                         "pos_best": best_c, "candidates": len(rest),
                         "lowering": sum(c < bal_num for c in counts.values()) if bal_num >= 0 else 0})
            if best_c >= bal_num:
                bal_num = best_c
                attribu.append(best_a)
                rest.remove(best_a)
            else:
                break
        ok = " ".join(map(str, attribu)) == reduct
        for r in rows:
            r.update(reduct_matches=ok, runtime_s=time.perf_counter() - t0)
    except Exception as exc:
        rows = [{"dataset": ds, "i": i, "error": str(exc)[:200]}]
    return rows


def job_nest(args):
    ds, fold, i = args
    row = {"dataset": ds, "fold": fold, "i": i}
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
        row.update(n_train=len(tr), d=d, min_sample=2 * (D.shape[1] - i))
        t0 = time.perf_counter()
        R = [int(a) for a in G.attribute_reduce(Dtr, pur=1, d2=i)]
        row["runtime_s"] = time.perf_counter() - t0
        row["|R|"] = len(R)
        row["cv5nn"] = float(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=5),
                                                     Dtr[:, R], Dtr[:, -2], cv=5)))
        k = KNeighborsClassifier(n_neighbors=3).fit(X[tr][:, R], y[tr])
        row["knn3_acc"] = float(k.score(X[te][:, R], y[te]))
        s = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X[tr][:, R], y[tr])
        row["svm_acc"] = float(s.score(X[te][:, R], y[te]))
    except Exception as exc:
        row["error"] = str(exc)[:200]
    return [row]


def run_one(job):
    kind, args = job
    return kind, {"audit": job_audit, "path": job_path, "nest": job_nest}[kind](args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbrs", required=True)
    ap.add_argument("--kinds", default="audit,path,nest")
    ap.add_argument("--jobs", type=int, default=13)
    a = ap.parse_args()
    base.GBRS_PATH = a.gbrs
    sel = selected_i()
    kinds = a.kinds.split(",")
    done = {k: set() for k in OUT}
    for k, p in OUT.items():
        if p.exists():
            with p.open() as fh:
                for r in csv.DictReader(fh):
                    if not r.get("error"):
                        done[k].add((r["dataset"], r.get("chunk") or r.get("fold") or "", r["i"]))
    jobs = []
    for ds in ALL:
        i, red = sel[ds]
        d = base.prepare(ds)[0].shape[1]
        if "audit" in kinds:
            nch = (100 if d > 20 else 200) // CHUNK
            jobs += [("audit", (ds, i, c)) for c in range(nch) if (ds, str(c), str(i)) not in done["audit"]]
        if "path" in kinds and not any(t[0] == ds for t in done["path"]):
            jobs.append(("path", (ds, i, red)))
        if "nest" in kinds and ds in base.LARGE:
            jobs += [("nest", (ds, f, j)) for f in range(5) for j in base.grid(d, True)
                     if (ds, str(f), str(j)) not in done["nest"]]
    # longest first: large datasets, small minimum ball sizes
    cost = {"digits": 6, "waveform": 5, "steel_plates": 3, "segment": 3, "pc3": 2, "yeast": 2}
    jobs.sort(key=lambda j: -(cost.get(j[1][0], 1) * 100 + (j[1][2] if j[0] == "nest" else 0)))
    fhs, ws = {}, {}
    for k in kinds:
        new = not OUT[k].exists()
        fhs[k] = OUT[k].open("a", newline="")
        ws[k] = csv.DictWriter(fhs[k], fieldnames=COLS[k])
        if new:
            ws[k].writeheader()
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(processes=a.jobs) as pool:
        for n, (kind, rows) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            for r in rows:
                ws[kind].writerow({c: r.get(c, "") for c in COLS[kind]})
            fhs[kind].flush()
            r = rows[0]
            print(f"[{n}/{len(jobs)}] {kind} {r['dataset']} i={r.get('i')} "
                  f"{r.get('chunk', r.get('fold', ''))} ({time.perf_counter() - t0:.0f}s)"
                  + (f" ERROR {r['error']}" if r.get("error") else ""), flush=True)
    for fh in fhs.values():
        fh.close()


if __name__ == "__main__":
    main()
