"""R1 revision: NRS with FIXED delta vs size-scaled delta (Major #4).

The manuscript treats NRS as a 'monotone reference'. But the code sets the
neighborhood radius delta = mean_pairwise_distance / sqrt(|A|), which SHRINKS as
attributes are added. A shrinking radius can drop points out of their neighbors'
neighborhoods, so gamma_NRS can DECREASE when an attribute is added -- i.e. the
size-scaled-delta NRS is NOT monotone. This script demonstrates the mechanism
and shows that a delta held FIXED (independent of |A|) restores monotonicity.

We audit both variants with the same random-subset protocol used elsewhere.
"""
from __future__ import annotations

import csv
import math
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
from run_experiments import load_dataset  # noqa: E402

RESULTS_DIR = CODE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAPER16 = [
    "wine", "sonar", "heart_statlog", "ecoli", "ionosphere", "dermatology",
    "spectf", "breast_cancer", "diabetes", "vehicle", "yeast", "pc3",
    "waveform", "digits", "steel_plates", "segment",
]
EPS = 1e-10
SEED = 42
N_SUBSETS = 120


def nrs_gamma_delta(X, y, attrs, delta):
    """NRS gamma on a GIVEN attribute set with an EXPLICIT fixed delta."""
    if not attrs:
        return 0.0
    Xa = X[:, list(attrs)]
    n = len(y)
    diff = Xa[:, None, :] - Xa[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    pos = 0
    for i in range(n):
        mask = D[i] <= delta + 1e-12
        if np.any(mask) and np.all(y[mask] == y[i]):
            pos += 1
    return pos / n


def nrs_gamma_scaled(X, y, attrs):
    """NRS gamma with the paper's size-scaled delta = mean_d / sqrt(|A|)."""
    if not attrs:
        return 0.0
    Xa = X[:, list(attrs)]
    n = len(y)
    diff = Xa[:, None, :] - Xa[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    off = D[np.triu_indices(n, k=1)]
    mean_d = float(np.mean(off)) if off.size else 0.1
    delta = mean_d / math.sqrt(len(attrs))
    pos = 0
    for i in range(n):
        mask = D[i] <= delta + 1e-12
        if np.any(mask) and np.all(y[mask] == y[i]):
            pos += 1
    return pos / n


def audit(X, y, gamma_fn, seed=SEED, n_subsets=N_SUBSETS):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    total = viol = 0
    for _ in range(n_subsets):
        size = int(rng.integers(1, d))
        A = sorted(rng.choice(d, size=size, replace=False).tolist())
        remaining = [a for a in range(d) if a not in A]
        gA = gamma_fn(A)
        for a in remaining:
            total += 1
            gB = gamma_fn(sorted(A + [a]))
            if gB < gA - EPS:
                viol += 1
    return 100.0 * viol / total if total else 0.0


def main() -> None:
    rows = []
    for name in PAPER16:
        X, y = load_dataset(name)
        n, d = X.shape
        if n > 1200:  # O(n^2) audit is too heavy for the largest sets
            print(f"[skip] {name} (n={n})", flush=True)
            rows.append({"dataset": name, "n": n, "d": d,
                         "scaled_delta_viol": "", "fixed_delta_viol": "",
                         "note": "skipped_large_n"})
            continue
        # Fixed delta: use the full-attribute-set mean distance / sqrt(d),
        # held CONSTANT across all subset sizes.
        Xa = X
        diff = Xa[:, None, :] - Xa[None, :, :]
        D = np.sqrt(np.sum(diff * diff, axis=2))
        off = D[np.triu_indices(n, k=1)]
        delta_fixed = float(np.mean(off)) / math.sqrt(d)

        v_scaled = audit(X, y, lambda A: nrs_gamma_scaled(X, y, A))
        v_fixed = audit(X, y, lambda A: nrs_gamma_delta(X, y, A, delta_fixed))
        rows.append({"dataset": name, "n": n, "d": d,
                     "scaled_delta_viol": round(v_scaled, 4),
                     "fixed_delta_viol": round(v_fixed, 4), "note": ""})
        print(f"[ok] {name:14s} scaled={v_scaled:6.2f}%  fixed={v_fixed:6.2f}%",
              flush=True)
        path = RESULTS_DIR / "nrs_fixed_delta.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
