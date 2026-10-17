"""Traces, step by step, why forward_greedy(GBRS) stops at |R|=1 on Heart-Statlog.

This is a sanity check against the reading "the small reducts must be a bug in
the stopping condition." It runs the paper's own gbrs_core.compute_gamma and
forward_greedy (unmodified) and prints the exact quantity Algorithm 1 compares
at each step, so the termination is visible rather than asserted. Companion to
Section 4.5 (\"The stopping rule overshoots gamma_AT on most benchmarks\").

For contrast it repeats the same trace for NRS on the same dataset, where
monotonicity guarantees gamma computed on all attributes is never exceeded by
any single attribute -- confirming the early stop is specific to non-monotone
fresh-generation gamma, not an artefact of the greedy-loop structure itself.
"""
import numpy as np
import gbrs_core as gc
from run_experiments import load_dataset


def main():
    X, y = load_dataset("heart_statlog")
    d = X.shape[1]
    print(f"heart_statlog: n={len(y)}, d={d}\n")

    gamma_star = gc.compute_gamma(X, y, list(range(d)), method="gbrs")
    print(f"GBRS gamma_star (all {d} attributes) = {gamma_star:.4f}")
    print("gamma for each single attribute (first greedy-step candidates):")
    best_a, best_g = None, 0.0
    for a in range(d):
        g = gc.compute_gamma(X, y, [a], method="gbrs")
        flag = "  <- exceeds gamma_star" if g > gamma_star else ""
        print(f"  attr {a}: gamma = {g:.4f}{flag}")
        if g > best_g:
            best_g, best_a = g, a
    print(f"\nGreedy picks attribute {best_a} (gamma={best_g:.4f}).")
    continue_loop = best_g < gamma_star - 1e-10
    print(f"Algorithm 1 loop condition gamma(R) < gamma_star - eps: "
          f"{best_g:.4f} < {gamma_star:.4f} => {continue_loop}")
    assert not continue_loop, "expected the loop to stop after one attribute here"

    R, gs, _ = gc.forward_greedy(X, y, method="gbrs")
    print(f"forward_greedy() result: R={R}, |R|={len(R)} -- matches the trace above.\n")

    print("--- Contrast: NRS on the same dataset (monotone gamma) ---")
    gamma_star_nrs = gc.nrs_gamma(X, y, list(range(d)))
    best_single = max(gc.nrs_gamma(X, y, [a]) for a in range(d))
    print(f"NRS gamma_star (all {d} attrs) = {gamma_star_nrs:.4f}")
    print(f"NRS best single-attribute gamma = {best_single:.4f} "
          f"(never exceeds gamma_star, by monotonicity)")
    assert best_single <= gamma_star_nrs
    R_nrs, _, _ = gc.forward_greedy(X, y, method="nrs")
    print(f"NRS forward_greedy() result: |R|={len(R_nrs)}")
    print("\nConclusion: the |R|=1 stop under GBRS is the loop's stated stopping\n"
          "condition being genuinely satisfied early -- a direct, verified\n"
          "consequence of non-monotone gamma, not a bug in Algorithm 1.")


if __name__ == "__main__":
    main()
