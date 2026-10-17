"""Check whether the HTBS column of greedy_path_audit.csv is an artefact of sorted().

run_greedy_path_audit.py evaluates every candidate as gfn(X, y, sorted(R + [a])).
For GBRS/DEG-GB that is harmless: compute_gamma regenerates balls from scratch in
the subspace, so gamma does not depend on attribute order.  HTBS is different --
htbs_forward_state builds incrementally in the order given and freezes balls
already admitted to POS, so re-sorting changes the freeze history and therefore
gamma.  This script re-runs the same audit but appends the candidate to the REAL
greedy path (htbs_eval_add), which is HTBS's own semantics, and reports whether
the violations disappear.
"""

import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ.setdefault("SCIKIT_LEARN_DATA", str(HERE.parent / ".sklearn_cache"))

from gbrs_core import (  # noqa: E402
    clear_gamma_cache,
    compute_gamma_htbs,
    global_entropy,
    htbs_eval_add,
    htbs_extend,
    htbs_forward_state,
    nmin_value,
)
from run_experiments import load_dataset  # noqa: E402

EPS = 1e-10


def htbs_path_audit(X, y, alpha=0.1):
    """Forward-greedy HTBS, evaluating candidates by APPENDING to the real path."""
    d = X.shape[1]
    gamma_star = compute_gamma_htbs(X, y, list(range(d)), alpha)
    h_thr = alpha * global_entropy(y)
    R, balls, gamma_cur = [], [], 0.0
    remaining = set(range(d))
    total_cand = total_dec = 0
    max_drop = 0.0
    while gamma_cur < gamma_star - EPS and remaining:
        best_a, best_g, best_balls = None, gamma_cur, balls
        for a in sorted(remaining):
            if not R:
                g = compute_gamma_htbs(X, y, [a], alpha)
            else:
                g = htbs_eval_add(X, y, R, a, balls, alpha)
            total_cand += 1
            if g < gamma_cur - EPS:
                total_dec += 1
                max_drop = max(max_drop, gamma_cur - g)
            if g > best_g + EPS:
                best_g, best_a = g, a
                if not R:
                    best_balls, _ = htbs_forward_state(X, y, [a], alpha)
                else:
                    best_balls = htbs_extend(
                        balls, X, y, R + [a], alpha, nmin_value(len(R) + 1, len(y))
                    )
        if best_a is None:
            break
        R.append(best_a)
        remaining.remove(best_a)
        balls, gamma_cur = best_balls, best_g
    return {
        "reduct_size": len(R),
        "gamma_star": round(gamma_star, 6),
        "gamma_reduct": round(gamma_cur, 6),
        "total_candidate_evals": total_cand,
        "total_decreasing_evals": total_dec,
        "pct_decreasing": round(100.0 * total_dec / total_cand, 4) if total_cand else 0.0,
        "max_drop": round(max_drop, 6),
    }


def main():
    datasets = sys.argv[1].split(",") if len(sys.argv) > 1 else [
        "ecoli", "ionosphere", "dermatology", "spectf", "yeast",
    ]
    out = HERE.parent / "results" / "htbs_path_audit_fixed.csv"
    rows = []
    for ds in datasets:
        clear_gamma_cache()
        X, y = load_dataset(ds)
        r = htbs_path_audit(X, y)
        r = {"dataset": ds, "d": X.shape[1], **r}
        rows.append(r)
        print(f"{ds:14s} |R|={r['reduct_size']:2d} "
              f"decr={r['total_decreasing_evals']}/{r['total_candidate_evals']} "
              f"({r['pct_decreasing']:.2f}%) maxdrop={r['max_drop']:.4f}", flush=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
