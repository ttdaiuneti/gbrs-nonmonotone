"""Run the granular-ball non-monotonicity experiments."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from gbrs_core import (  # noqa: E402
    compute_gamma,
    compute_gamma_htbs,
    forward_greedy,
    global_entropy,
    monotonicity_audit,
    nrs_gamma,
)

RESULTS_DIR = CODE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("iris", 150, 4),
    ("wine", 178, 13),
    ("sonar", 208, 60),
    ("glass", 214, 9),
    ("heart_statlog", 270, 13),
    ("ecoli", 336, 7),
    ("ionosphere", 351, 34),
    ("dermatology", 366, 34),
    ("spectf", 349, 44),
    ("breast_cancer", 569, 30),
    ("balance_scale", 625, 4),
    ("diabetes", 768, 8),
    ("vehicle", 846, 18),
    ("banknote", 1372, 4),
    ("yeast", 1484, 8),
    ("pc3", 1563, 37),
    ("waveform", 5000, 40),
    ("digits", 5620, 64),
    ("steel_plates", 1941, 33),
    ("segment", 2310, 19),
]

OPENML_MAP = {
    "iris": "iris",
    "wine": "wine",
    "sonar": "sonar",
    "glass": "glass",
    "heart_statlog": "heart-statlog",
    "ecoli": "ecoli",
    "ionosphere": "ionosphere",
    "dermatology": "dermatology",
    "spectf": "spectf",
    "breast_cancer": "wdbc",
    "balance_scale": "balance-scale",
    "diabetes": "diabetes",
    "vehicle": "vehicle",
    "banknote": "banknote-authentication",
    "yeast": "yeast",
    "pc3": "pc3",
    "waveform": "waveform-5000",
    "digits": "optdigits",
    "steel_plates": "steel-plates-fault",
    "segment": "segment",
}


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    key = OPENML_MAP[name]
    data = fetch_openml(key, version=1, as_frame=False, parser="auto")
    X, y = data.data, data.target
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if y.dtype.kind in "OUS":
        classes, y = np.unique(y, return_inverse=True)
    y = y.astype(np.int64)
    # Handle missing values (e.g., dermatology)
    if np.any(np.isnan(X)):
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y


def evaluate_reduct(
    X: np.ndarray,
    y: np.ndarray,
    reduct: List[int],
    seed: int = 0,
) -> Dict[str, float]:
    if not reduct:
        return {"knn3_mean": 0.0, "knn3_std": 0.0, "svm_mean": 0.0, "svm_std": 0.0}
    Xr = X[:, reduct]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    knn_scores, svm_scores = [], []
    for tr, te in skf.split(Xr, y):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(Xr[tr], y[tr])
        knn_scores.append(knn.score(Xr[te], y[te]))
        svm = SVC(kernel="rbf", C=1.0, gamma="scale")
        svm.fit(Xr[tr], y[tr])
        svm_scores.append(svm.score(Xr[te], y[te]))
    return {
        "knn3_mean": float(np.mean(knn_scores)),
        "knn3_std": float(np.std(knn_scores)),
        "svm_mean": float(np.mean(svm_scores)),
        "svm_std": float(np.std(svm_scores)),
    }


def full_attr_baseline(X: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[str, float]:
    return evaluate_reduct(X, y, list(range(X.shape[1])), seed)


def run_mono_experiments(seeds: List[int] = None) -> None:
    seeds = seeds or [42]
    rows = []
    for ds_name, _, _ in DATASETS:
        print(f"[mono] {ds_name}")
        X, y = load_dataset(ds_name)
        from gbrs_core import clear_gamma_cache
        clear_gamma_cache()
        d = X.shape[1]
        for method in ("gbrs", "deg", "htbs", "nrs"):
            rates = []
            for seed in seeds:
                res = monotonicity_audit(X, y, method=method, seed=seed)
                rates.append(res["violation_rate"])
            row = {
                "dataset": ds_name,
                "d": d,
                "method": method,
                "violation_rate_mean": float(np.mean(rates)),
                "violation_rate_std": float(np.std(rates)),
            }
            if seeds == [42]:
                res = monotonicity_audit(X, y, method=method, seed=42)
                row.update({
                    "violations": res["violations"],
                    "total_pairs": res["total_pairs"],
                    "mean_delta": res["mean_delta"],
                    "max_delta": res["max_delta"],
                    "median_delta": res["median_delta"],
                })
            rows.append(row)
    path = RESULTS_DIR / "mono_audit.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def run_classification_experiments() -> None:
    methods = [
        ("gbrs", False, "GBRS"),
        ("deg", False, "DEG-GB"),
        ("deg", True, "HTBS"),
        ("nrs", False, "NRS"),
    ]
    from gbrs_core import clear_gamma_cache

    rows = []
    for ds_name, _, _ in DATASETS:
        print(f"[cls] {ds_name}", flush=True)
        clear_gamma_cache()  # never carry cached gamma across datasets
        try:
            X, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  SKIP {ds_name}: {e}", flush=True)
            continue
        d = X.shape[1]
        base = full_attr_baseline(X, y)
        for method, htbs, label in methods:
            t0 = time.perf_counter()
            if htbs:
                R, gamma_star, _ = forward_greedy(X, y, method="deg", htbs=True)
            else:
                R, gamma_star, _ = forward_greedy(X, y, method=method)
            runtime = time.perf_counter() - t0
            ev = evaluate_reduct(X, y, R)
            rows.append({
                "dataset": ds_name,
                "d": d,
                "method": label,
                "|R|": len(R),
                "gamma_star": gamma_star,
                "runtime_s": runtime,
                "knn3_acc": ev["knn3_mean"],
                "knn3_std": ev["knn3_std"],
                "svm_acc": ev["svm_mean"],
                "svm_std": ev["svm_std"],
                "full_knn3": base["knn3_mean"],
                "full_svm": base["svm_mean"],
            })
            # incremental save
            path = RESULTS_DIR / "classification.csv"
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
    if not rows:
        print("No classification results.")
        return
    path = RESULTS_DIR / "classification.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def bootstrap_ci(rates: List[float], n_boot: int = 1000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(rates)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--cls", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--quick", action="store_true", help="iris+wine only")
    args = parser.parse_args()

    if args.quick:
        DATASETS[:] = [("iris", 150, 4), ("wine", 178, 13)]

    if args.all or args.mono:
        run_mono_experiments(seeds=[42, 0, 1, 2, 3] if not args.quick else [42])
    if args.all or args.cls:
        run_classification_experiments()

    if not (args.mono or args.cls or args.all):
        parser.print_help()
