"""Verifies the constructive counterexample used in the proof of Proposition 1
(non-monotonicity of fresh-generation GB), Section 4.1 of main.tex.

Runs gbrs_core._generate_balls / gamma_from_balls directly -- the same routine
used everywhere else in the paper's audit -- on the four explicit points given
in the proof, and checks:
  (a) the one-attribute subspace {a1} yields the pure cover {{1,2},{3,4}}
      with gamma = 1;
  (b) the two-attribute subspace {a1,a2} yields the incomparable cover
      {{1,3},{2,4}} with gamma = 0;
  (c) the result is stable under +/-0.3 perturbation of every coordinate
      (200 random trials), i.e. not a floating-point knife-edge.
"""
import numpy as np
import gbrs_core as gc

Y = np.array([0, 0, 1, 1])
X = np.array([
    [0.0, 1.0],
    [1.0, 36.0],
    [3.0, -28.0],
    [4.0, 36.0],
])
NMIN = 2


def run(X_sub):
    balls = gc._generate_balls(X_sub, Y, "purity", 0.0, nmin=NMIN)
    gamma = gc.gamma_from_balls(balls, "purity", 0.0)
    groups = sorted(sorted(idx_f) for idx_f, _ in balls)
    return gamma, groups


def main():
    gamma_A, groups_A = run(X[:, [0]])
    gamma_AB, groups_AB = run(X[:, [0, 1]])

    print(f"Subspace {{a1}}:      balls={groups_A}  gamma={gamma_A}")
    print(f"Subspace {{a1,a2}}:   balls={groups_AB}  gamma={gamma_AB}")

    assert groups_A == [[0, 1], [2, 3]], "expected {{1,2},{3,4}} (0-indexed)"
    assert gamma_A == 1.0
    assert groups_AB == [[0, 2], [1, 3]], "expected {{1,3},{2,4}} (0-indexed)"
    assert gamma_AB == 0.0
    print("Proposition 1 counterexample VERIFIED: gamma drops from 1.0 to 0.0.")

    rng = np.random.default_rng(0)
    ok = 0
    trials = 200
    for _ in range(trials):
        Xp = X.copy()
        Xp += rng.uniform(-0.3, 0.3, size=Xp.shape)
        g, groups = run(Xp[:, [0, 1]])
        if sorted(groups) == [[0, 2], [1, 3]] and g == 0.0:
            ok += 1
    print(f"Stability under +/-0.3 perturbation of every coordinate: {ok}/{trials} trials reproduce the same cover.")
    assert ok == trials, "counterexample is not robust to small perturbation"


if __name__ == "__main__":
    main()
