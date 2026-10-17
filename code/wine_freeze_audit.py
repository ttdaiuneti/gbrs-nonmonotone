"""Reproduce the Wine first-step HTBS freeze numbers in Section 5.

For each candidate first attribute a, generate the HTBS partition on {a}
and record the frozen mass (instances in mixed balls that cannot split)
and the resulting gamma. After a completed first-step generation every
mixed ball is unsplittable, so freeze_frac = 1 - gamma.

Outputs results/wine_freeze_audit.csv.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ.setdefault("SCIKIT_LEARN_DATA", str(HERE.parent / ".sklearn_cache"))

from gbrs_core import (  # noqa: E402
    _generate_balls,
    compute_gamma,
    forward_greedy,
    gamma_from_balls,
    global_entropy,
    htbs_forward_state,
    local_entropy,
    nmin_value,
)
from run_experiments import load_dataset  # noqa: E402


def freeze_stats(balls, y, h_thr, nmin):
    n = len(y)
    frozen = 0
    pos = 0
    mixed_splittable = 0
    for idx_f, labels in balls:
        m = len(idx_f)
        if local_entropy(labels) <= h_thr + 1e-12:
            pos += m
        elif m <= nmin:
            frozen += m
        else:
            mixed_splittable += m
    return {
        "n": n,
        "pos": pos,
        "frozen": frozen,
        "mixed_splittable": mixed_splittable,
        "freeze_frac": frozen / n,
        "gamma": pos / n,
        "ceiling": 1.0 - frozen / n,
    }


def main() -> None:
    np.random.seed(42)
    X, y = load_dataset("wine")
    n, d = X.shape
    alpha = 0.1
    h_thr = alpha * global_entropy(y)
    nmin = nmin_value(1, n)

    rows = []
    for a in range(d):
        balls, g = htbs_forward_state(X, y, [a], alpha)
        st = freeze_stats(balls, y, h_thr, nmin)
        rows.append({
            "attr": a,
            "gamma": round(st["gamma"], 6),
            "nmin_freeze_frac": round(st["freeze_frac"], 6),
            "nmin_freeze_pct": round(100.0 * st["freeze_frac"], 4),
            "nmin_ceiling": round(st["ceiling"], 6),
            "nonpos_frac": round(1.0 - st["gamma"], 6),
            "nonpos_pct": round(100.0 * (1.0 - st["gamma"]), 4),
            "pos": st["pos"],
            "nmin_frozen": st["frozen"],
            "mixed_splittable": st["mixed_splittable"],
            "nmin": nmin,
        })
        print(f"  a={a:2d}  gamma={st['gamma']:.4f}  "
              f"nmin-freeze={100*st['freeze_frac']:.1f}%  "
              f"nonPOS={100*(1-st['gamma']):.1f}%  leftover={st['mixed_splittable']}",
              flush=True)

    freeze_pcts = [r["nmin_freeze_pct"] for r in rows]
    nonpos_pcts = [r["nonpos_pct"] for r in rows]
    gammas = [r["gamma"] for r in rows]
    best = max(rows, key=lambda r: r["gamma"])

    R_htbs, gstar_htbs, _ = forward_greedy(X, y, method="deg", htbs=True)
    R_gbrs, gstar_gbrs, _ = forward_greedy(X, y, method="gbrs")
    balls0, g0 = htbs_forward_state(X, y, [R_htbs[0]], alpha)
    st0 = freeze_stats(balls0, y, h_thr, nmin)
    _, g_path = htbs_forward_state(X, y, R_htbs, alpha)

    summary = {
        "mean_gamma": float(np.mean(gammas)),
        "mean_nonpos_pct": float(np.mean(nonpos_pcts)),
        "mean_nmin_freeze_pct": float(np.mean(freeze_pcts)),
        "mean_nmin_ceiling": float(np.mean([r["nmin_ceiling"] for r in rows])),
        "best_first_attr": best["attr"],
        "best_first_gamma": best["gamma"],
        "greedy_first_attr": R_htbs[0],
        "greedy_first_gamma": round(g0, 6),
        "greedy_first_nmin_ceiling": round(st0["ceiling"], 6),
        "greedy_path_gamma": round(g_path, 6),
        "htbs_gamma_AT": round(gstar_htbs, 6),
        "gbrs_gamma_star": round(gstar_gbrs, 6),
        "gbrs_reduct": R_gbrs,
        "htbs_reduct": R_htbs,
    }
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = HERE.parent / "results" / "wine_freeze_audit.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

    meta = HERE.parent / "results" / "wine_freeze_summary.csv"
    flat = {k: (";".join(map(str, v)) if isinstance(v, list) else v)
            for k, v in summary.items()}
    with meta.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat.keys()))
        w.writeheader()
        w.writerow(flat)
    print(f"wrote {meta}")


if __name__ == "__main__":
    main()
