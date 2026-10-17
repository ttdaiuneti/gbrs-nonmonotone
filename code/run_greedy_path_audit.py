"""R1 revision: greedy-path (step-level) monotonicity audit (Major #5).

The random-subset audit measures gamma-monotonicity over arbitrary attribute
pairs. This one measures it ALONG THE ACTUAL forward-greedy trajectory: at each
step the algorithm re-evaluates every candidate attribute a against the current
reduct R, i.e. it computes gamma(R + {a}). We record how many of those per-step
candidate evaluations are DECREASES relative to gamma(R) -- the violations the
greedy heuristic actually encounters while choosing the next attribute.

We also report, at each accepted step, gamma(R) itself, which is monotone by
construction of the accept rule (only strictly-improving adds are taken); the
point of interest is the rejected candidates that would have LOWERED gamma.
"""
from __future__ import annotations

import csv
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

from gbrs_core import (  # noqa: E402
    clear_gamma_cache,
    compute_gamma,
    compute_gamma_htbs,
)
from run_experiments import DATASETS, load_dataset  # noqa: E402

RESULTS_DIR = CODE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAPER16 = [
    "wine", "sonar", "heart_statlog", "ecoli", "ionosphere", "dermatology",
    "spectf", "breast_cancer", "diabetes", "vehicle", "yeast", "pc3",
    "waveform", "digits", "steel_plates", "segment",
]
EPS = 1e-10


def gamma_fn_for(method: str):
    if method == "gbrs":
        return lambda X, y, A: compute_gamma(X, y, A, "gbrs", 0.1)
    if method == "deg":
        return lambda X, y, A: compute_gamma(X, y, A, "deg", 0.1)
    if method == "htbs":
        return lambda X, y, A: compute_gamma_htbs(X, y, A, 0.1)
    raise ValueError(method)


def greedy_path_audit(X, y, method: str):
    """Forward-greedy while logging per-step candidate decreases."""
    gfn = gamma_fn_for(method)
    d = X.shape[1]
    gamma_star = gfn(X, y, list(range(d)))
    R: list = []
    gamma_cur = 0.0
    remaining = set(range(d))
    step_rows = []
    total_cand = 0
    total_dec = 0
    max_drop = 0.0
    while gamma_cur < gamma_star - EPS and remaining:
        best_a, best_g = None, gamma_cur
        n_dec_here = 0
        for a in remaining:
            g = gfn(X, y, sorted(R + [a]))
            if g < gamma_cur - EPS:
                n_dec_here += 1
                total_dec += 1
                max_drop = max(max_drop, gamma_cur - g)
            total_cand += 1
            if g > best_g + EPS:
                best_g, best_a = g, a
        if best_a is None:
            break
        R.append(best_a)
        remaining.remove(best_a)
        step_rows.append({
            "step": len(R), "added": best_a, "gamma_after": round(best_g, 6),
            "n_candidates": len(remaining) + 1,
            "n_decreasing_candidates": n_dec_here,
        })
        gamma_cur = best_g
    return {
        "reduct_size": len(R),
        "gamma_star": gamma_star,
        "gamma_reduct": gamma_cur,
        "total_candidate_evals": total_cand,
        "total_decreasing_evals": total_dec,
        "pct_decreasing": (100.0 * total_dec / total_cand) if total_cand else 0.0,
        "max_drop": max_drop,
    }, step_rows


def main() -> None:
    summary = []
    for name in PAPER16:
        X, y = load_dataset(name)
        for method in ("gbrs", "deg", "htbs"):
            clear_gamma_cache()
            s, _ = greedy_path_audit(X, y, method)
            row = {"dataset": name, "d": X.shape[1], "method": method.upper()}
            row.update(s)
            row["gamma_star"] = round(row["gamma_star"], 6)
            row["gamma_reduct"] = round(row["gamma_reduct"], 6)
            row["pct_decreasing"] = round(row["pct_decreasing"], 4)
            row["max_drop"] = round(row["max_drop"], 6)
            summary.append(row)
            print(f"[ok] {name:14s} {method:5s} |R|={s['reduct_size']:2d} "
                  f"decr={s['total_decreasing_evals']}/{s['total_candidate_evals']} "
                  f"({row['pct_decreasing']:.1f}%) maxdrop={row['max_drop']:.4f}",
                  flush=True)
        path = RESULTS_DIR / "greedy_path_audit.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary[0].keys())
            w.writeheader()
            w.writerows(summary)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
