"""
Major #7 — broaden the monotonicity audit beyond the purity-based GBRS.

We add a fuzzy / soft-purity granular-ball dependency surrogate (FUZ-GB) that
represents the fuzzy-GB family: it reuses the identical fresh-generation ball
partition (purity split, class-centroid k-means), but replaces the hard
positive-region count (fraction of samples in perfectly pure balls) with a soft
dependency = sum over balls of |ball| * purity(ball) / n.  This is a genuinely
different dependency measure, yet -- because it is still computed on a freshly
regenerated partition per attribute subset -- it inherits partition
incomparability and should show the same non-monotonicity.

Reducers audited (all fresh-generation):
  gbrs  : hard purity gamma (baseline)
  deg   : entropy-threshold admission control (already in paper)
  fuz   : NEW soft/fuzzy purity gamma
Same protocol as the headline audit: N random subsets, seed 42, eps=1e-10.
"""
import os, sys, csv, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # the R2/ package root
sys.path.insert(0, _HERE)
os.environ.setdefault("SCIKIT_LEARN_DATA", os.path.join(_ROOT, ".skcache"))
import numpy as np
import gbrs_core as gc
from run_experiments import load_dataset, DATASETS  # reuse loaders/openml map

def fuzzy_gamma(X, y, attrs, nmin_override=None):
    """Soft-purity dependency on a freshly generated purity partition."""
    if not attrs:
        return 0.0
    Xa = X[:, sorted(attrs)]
    nmin = gc.nmin_value(len(attrs), len(y))
    balls = gc._generate_balls(Xa, y, "purity", 0.0, nmin)
    n_total = sum(len(b[0]) for b in balls)
    if n_total == 0:
        return 0.0
    soft = 0.0
    for idx_f, labels in balls:
        soft += len(idx_f) * gc.purity(labels)
    return soft / n_total

def audit_fuzzy(X, y, n_subsets, seed=42, eps=1e-10):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    total = viol = 0
    deltas = []
    for _ in range(n_subsets):
        size = int(rng.integers(1, d))
        A = sorted(rng.choice(d, size=size, replace=False).tolist())
        remaining = [a for a in range(d) if a not in A]
        gA = fuzzy_gamma(X, y, A)
        for a in remaining:
            total += 1
            gB = fuzzy_gamma(X, y, sorted(A + [a]))
            if gB < gA - eps:
                viol += 1
                deltas.append(gA - gB)
    return {
        "violation_rate": 100.0 * viol / total if total else 0.0,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "max_delta": float(np.max(deltas)) if deltas else 0.0,
    }

def subsets_for_n(n):
    if n <= 400: return 200
    if n <= 900: return 120
    if n <= 1600: return 60
    return 25

# representative spread across d and n; skip the very largest for tractability
SUBSET = ["wine","sonar","heart_statlog","ecoli","ionosphere","dermatology",
          "spectf","breast_cancer","diabetes","vehicle","yeast","pc3"]

def main():
    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "variant_audit.csv")
    rows = []
    for name in SUBSET:
        try:
            X, y = load_dataset(name)
        except Exception as e:
            print(f"[skip] {name}: {e}", flush=True); continue
        d = X.shape[1]; n = X.shape[0]
        ns = subsets_for_n(n)
        t0 = time.perf_counter()
        r_gbrs = gc.monotonicity_audit(X, y, "gbrs", n_subsets=ns, seed=42)
        r_deg  = gc.monotonicity_audit(X, y, "deg",  n_subsets=ns, seed=42)
        r_fuz  = audit_fuzzy(X, y, ns, seed=42)
        rows.append([name, d, n, ns,
                     round(r_gbrs["violation_rate"],3), round(r_gbrs["mean_delta"],5),
                     round(r_deg["violation_rate"],3),  round(r_deg["mean_delta"],5),
                     round(r_fuz["violation_rate"],3),  round(r_fuz["mean_delta"],5)])
        print(f"[ok] {name:14s} gbrs={r_gbrs['violation_rate']:.1f} "
              f"deg={r_deg['violation_rate']:.1f} fuz={r_fuz['violation_rate']:.1f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","d","n","n_subsets",
                    "gbrs_rate","gbrs_mean_delta","deg_rate","deg_mean_delta",
                    "fuz_rate","fuz_mean_delta"])
        w.writerows(rows)
    print("DONE", out, flush=True)

if __name__ == "__main__":
    main()
