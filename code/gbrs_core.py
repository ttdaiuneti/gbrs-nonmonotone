"""Granular-ball rough set core: GBRS, DEG-GB, HTBS, NRS."""
from __future__ import annotations

import hashlib
import math
import time
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

Ball = Tuple[frozenset, NDArray]  # (indices, labels)
_GAMMA_CACHE: Dict[Tuple, float] = {}

# --- Content-addressed cache keys -------------------------------------------
# Keying the gamma cache on id(X) is unsafe: CPython reuses the address of a
# garbage-collected array, so a later dataset can inherit a cache entry
# belonging to an earlier one. We key on a SHA-1 of the array contents instead,
# and hold a strong reference to every array we have hashed so that its id()
# can never be recycled while the corresponding entries are live (this also
# makes the hash a once-per-array cost rather than a per-call one).
_ARRAY_DIGEST: Dict[int, str] = {}
_ARRAY_KEEPALIVE: Dict[int, NDArray] = {}


def array_key(a: NDArray) -> str:
    """Stable content-based identity for a NumPy array."""
    k = id(a)
    if _ARRAY_KEEPALIVE.get(k) is not a:
        _ARRAY_KEEPALIVE[k] = a
        _ARRAY_DIGEST[k] = hashlib.sha1(
            np.ascontiguousarray(a).tobytes()
        ).hexdigest()
    return _ARRAY_DIGEST[k]


def global_entropy(labels: NDArray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-15)))


def local_entropy(labels: NDArray) -> float:
    if len(labels) == 0:
        return 0.0
    return global_entropy(labels)


def purity(labels: NDArray) -> float:
    if len(labels) == 0:
        return 1.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())


def nmin_value(n_attrs: int, n_samples: int) -> int:
    return max(2, min(n_attrs + 1, max(2, int(n_samples / 10))))


def _class_centroid_init(X: NDArray, y: NDArray, k: int) -> NDArray:
    classes = np.unique(y)
    if len(classes) >= k:
        chosen = classes[:k]
    else:
        chosen = classes
    centers = []
    for c in chosen:
        mask = y == c
        centers.append(X[mask].mean(axis=0))
    while len(centers) < k:
        centers.append(X[np.random.randint(len(X))])
    return np.vstack(centers[:k])


def _split_ball(
    X: NDArray,
    y: NDArray,
    idx: NDArray,
    max_iter: int = 30,
) -> List[NDArray]:
    if len(idx) <= 1:
        return [idx]
    Xi = X[idx]
    yi = y[idx]
    k = min(2, len(idx))
    if k < 2:
        return [idx]
    init = _class_centroid_init(Xi, yi, k)
    km = KMeans(n_clusters=k, init=init, n_init=1, max_iter=max_iter)
    labels = km.fit_predict(Xi)
    parts = [idx[labels == c] for c in range(k) if np.any(labels == c)]
    if len(parts) <= 1:
        return [idx]
    return parts


def _generate_balls(
    X: NDArray,
    y: NDArray,
    criterion: str,
    h_thr: float,
    nmin: int,
) -> List[Ball]:
    n = len(y)
    balls: List[Ball] = [(frozenset(range(n)), y.copy())]
    changed = True
    while changed:
        changed = False
        next_balls: List[Ball] = []
        for idx_f, labels in balls:
            idx = np.array(sorted(idx_f), dtype=int)
            lab = y[idx]
            need_split = False
            if criterion == "purity":
                need_split = purity(lab) < 1.0 - 1e-12 and len(idx) > nmin
            else:
                need_split = local_entropy(lab) > h_thr + 1e-12 and len(idx) > nmin
            if need_split:
                parts = _split_ball(X, y, idx)
                if len(parts) > 1:
                    changed = True
                    for p in parts:
                        next_balls.append((frozenset(p.tolist()), y[p]))
                else:
                    next_balls.append((idx_f, lab))
            else:
                next_balls.append((idx_f, lab))
        balls = next_balls
    return balls


def gamma_from_balls(balls: List[Ball], criterion: str, h_thr: float) -> float:
    n_total = sum(len(b[0]) for b in balls)
    if n_total == 0:
        return 0.0
    pos = 0
    for idx_f, labels in balls:
        ok = False
        if criterion == "purity":
            ok = purity(labels) >= 1.0 - 1e-12
        else:
            ok = local_entropy(labels) <= h_thr + 1e-12
        if ok:
            pos += len(idx_f)
    return pos / n_total


def _balls_partition_size(balls: List[Ball]) -> int:
    return sum(len(b[0]) for b in balls)


@lru_cache(maxsize=None)
def _gamma_gbrs_cached(
    X_tuple: Tuple[Tuple[float, ...], ...],
    y_tuple: Tuple[int, ...],
    attr_key: Tuple[int, ...],
    criterion: str,
    alpha_bits: int,
    nmin: int,
) -> float:
    X = np.array(X_tuple, dtype=np.float64)
    y = np.array(y_tuple, dtype=np.int64)
    attrs = list(attr_key)
    Xa = X[:, attrs] if attrs else X[:, :0]
    h_thr = alpha_bits / 1e6  # alpha encoded
    if criterion == "purity":
        h_thr = 0.0
    else:
        h_thr = (alpha_bits / 1000.0) * global_entropy(y)
    balls = _generate_balls(Xa, y, criterion, h_thr, nmin)
    return gamma_from_balls(balls, criterion, h_thr)


def compute_gamma(
    X: NDArray,
    y: NDArray,
    attrs: Sequence[int],
    method: str = "deg",
    alpha: float = 0.1,
    nmin_override: Optional[int] = None,
) -> float:
    """Fresh-generation dependency degree.

    `nmin_override` pins the minimum ball size instead of using the default
    |A|-dependent schedule `nmin_value(|A|, n)`. Because that schedule GROWS
    with |A|, gamma(A) and gamma(A + {a}) are otherwise computed under two
    different stopping rules, which by itself can make gamma decrease. Pinning
    nmin isolates the effect of ball REGENERATION from the effect of the
    stopping rule.
    """
    key = (array_key(X), array_key(y), tuple(sorted(attrs)), method, alpha,
           nmin_override)
    if key in _GAMMA_CACHE:
        return _GAMMA_CACHE[key]
    n_attrs = X.shape[1]
    nmin = (nmin_override if nmin_override is not None
            else nmin_value(len(attrs) if attrs else n_attrs, len(y)))
    criterion = "entropy" if method == "deg" else "purity"
    h_thr = alpha * global_entropy(y) if method == "deg" else 0.0
    Xa = X[:, list(attrs)] if attrs else X[:, :0]
    balls = _generate_balls(Xa, y, criterion, h_thr, nmin)
    val = gamma_from_balls(balls, criterion, h_thr)
    _GAMMA_CACHE[key] = val
    return val


def htbs_extend(
    balls: List[Ball],
    X: NDArray,
    y: NDArray,
    attrs: Sequence[int],
    alpha: float,
    nmin: int,
) -> List[Ball]:
    h_thr = alpha * global_entropy(y)
    Xa = X[:, list(attrs)]
    new_balls: List[Ball] = []
    for idx_f, labels in balls:
        idx = np.array(sorted(idx_f), dtype=int)
        lab = y[idx]
        if local_entropy(lab) <= h_thr + 1e-12:
            new_balls.append((idx_f, lab))
        elif len(idx) > nmin:
            parts = _split_ball(Xa, y, idx)
            for p in parts:
                new_balls.append((frozenset(p.tolist()), y[p]))
        else:
            new_balls.append((idx_f, lab))
    return new_balls


def htbs_gamma_for_attrs(
    X: NDArray,
    y: NDArray,
    attrs: Sequence[int],
    alpha: float = 0.1,
    build_order: Optional[Sequence[int]] = None,
) -> float:
    if not attrs:
        return 0.0
    order = list(build_order) if build_order is not None else sorted(attrs)
    h_thr = alpha * global_entropy(y)
    first = order[0]
    balls = _generate_balls(
        X[:, [first]], y, "entropy", h_thr, nmin_value(1, len(y))
    )
    current = [first]
    for a in order[1:]:
        if a not in current:
            current.append(a)
        balls = htbs_extend(
            balls, X, y, current, alpha, nmin_value(len(current), len(y))
        )
    return gamma_from_balls(balls, "entropy", h_thr)


def compute_gamma_htbs(
    X: NDArray,
    y: NDArray,
    attrs: Sequence[int],
    alpha: float = 0.1,
) -> float:
    return htbs_gamma_for_attrs(X, y, attrs, alpha)


def htbs_forward_state(
    X: NDArray,
    y: NDArray,
    R: List[int],
    alpha: float,
) -> Tuple[List[Ball], float]:
    if not R:
        return [], 0.0
    h_thr = alpha * global_entropy(y)
    first = R[0]
    balls = _generate_balls(
        X[:, [first]], y, "entropy", h_thr, nmin_value(1, len(y))
    )
    current = [first]
    for a in R[1:]:
        current.append(a)
        balls = htbs_extend(
            balls, X, y, current, alpha, nmin_value(len(current), len(y))
        )
    return balls, gamma_from_balls(balls, "entropy", h_thr)


def htbs_eval_add(
    X: NDArray,
    y: NDArray,
    R: List[int],
    a: int,
    balls: List[Ball],
    alpha: float,
) -> float:
    h_thr = alpha * global_entropy(y)
    new_attrs = R + [a]
    new_balls = htbs_extend(
        balls, X, y, new_attrs, alpha, nmin_value(len(new_attrs), len(y))
    )
    return gamma_from_balls(new_balls, "entropy", h_thr)


def clear_gamma_cache() -> None:
    _GAMMA_CACHE.clear()
    _ARRAY_DIGEST.clear()
    _ARRAY_KEEPALIVE.clear()


def nrs_gamma(X: NDArray, y: NDArray, attrs: Sequence[int], delta: Optional[float] = None) -> float:
    # `delta` MUST be part of the key: the fixed-radius audit calls this with
    # the same (X, y, attrs) under two different radii.
    key = ("nrs", array_key(X), array_key(y), tuple(sorted(attrs)), delta)
    if key in _GAMMA_CACHE:
        return _GAMMA_CACHE[key]
    if not attrs:
        val = 0.0
        _GAMMA_CACHE[key] = val
        return val
    Xa = X[:, list(attrs)]
    n = len(y)
    # Vectorised pairwise distances (upper triangle for mean delta)
    diff = Xa[:, None, :] - Xa[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    if delta is None:
        off = D[np.triu_indices(n, k=1)]
        mean_d = float(np.mean(off)) if off.size else 0.1
        delta = mean_d / math.sqrt(len(attrs))
    pos = 0
    for i in range(n):
        mask = D[i] <= delta + 1e-12
        if np.any(mask) and np.all(y[mask] == y[i]):
            pos += 1
    val = pos / n
    _GAMMA_CACHE[key] = val
    return val


def forward_greedy(
    X: NDArray,
    y: NDArray,
    method: str = "gbrs",
    alpha: float = 0.1,
    htbs: bool = False,
) -> Tuple[List[int], float, float]:
    """Return (reduct indices, gamma_target, elapsed_seconds)."""
    t0 = time.perf_counter()
    d = X.shape[1]
    all_attrs = list(range(d))
    if method == "nrs":
        gamma_star = nrs_gamma(X, y, all_attrs)
        R: List[int] = []
        gamma_cur = 0.0
        remaining = set(all_attrs)
        while gamma_cur < gamma_star - 1e-10 and remaining:
            best_a, best_g = None, gamma_cur
            for a in remaining:
                g = nrs_gamma(X, y, R + [a])
                if g > best_g + 1e-10:
                    best_g = g
                    best_a = a
            if best_a is None:
                break
            R.append(best_a)
            remaining.remove(best_a)
            gamma_cur = best_g
        return R, gamma_star, time.perf_counter() - t0

    gamma_fn = compute_gamma_htbs if htbs else (lambda X, y, A: compute_gamma(X, y, A, "deg" if method == "deg" else "gbrs", alpha))
    if htbs:
        d = X.shape[1]
        all_attrs = list(range(d))
        gamma_star = compute_gamma_htbs(X, y, all_attrs, alpha)
        R: List[int] = []
        balls: List[Ball] = []
        gamma_cur = 0.0
        remaining = set(all_attrs)
        while gamma_cur < gamma_star - 1e-10 and remaining:
            best_a, best_g, best_balls = None, gamma_cur, balls
            for a in remaining:
                if not R:
                    g = compute_gamma_htbs(X, y, [a], alpha)
                    if g > best_g + 1e-10:
                        best_g = g
                        best_a = a
                        _, _ = htbs_forward_state(X, y, [a], alpha)
                        best_balls, _ = htbs_forward_state(X, y, [a], alpha)
                else:
                    g = htbs_eval_add(X, y, R, a, balls, alpha)
                    if g > best_g + 1e-10:
                        best_g = g
                        best_a = a
                        h_thr = alpha * global_entropy(y)
                        best_balls = htbs_extend(
                            balls, X, y, R + [a], alpha, nmin_value(len(R) + 1, len(y))
                        )
            if best_a is None:
                break
            R.append(best_a)
            remaining.remove(best_a)
            balls, gamma_cur = htbs_forward_state(X, y, R, alpha)
        return R, gamma_star, time.perf_counter() - t0

    gamma_star = gamma_fn(X, y, all_attrs)
    R = []
    gamma_cur = 0.0
    remaining = set(all_attrs)
    while gamma_cur < gamma_star - 1e-10 and remaining:
        best_a, best_g = None, gamma_cur
        for a in remaining:
            g = gamma_fn(X, y, R + [a])
            if g > best_g + 1e-10:
                best_g = g
                best_a = a
        if best_a is None:
            break
        R.append(best_a)
        remaining.remove(best_a)
        gamma_cur = best_g
    return R, gamma_star, time.perf_counter() - t0


def monotonicity_audit(
    X: NDArray,
    y: NDArray,
    method: str = "gbrs",
    alpha: float = 0.1,
    n_subsets: int = 200,
    seed: int = 42,
    epsilon: float = 1e-10,
    nmin_override: Optional[int] = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    d = X.shape[1]

    def gamma_A_htbs(A_list: List[int]) -> float:
        if not A_list:
            return 0.0
        _, g = htbs_forward_state(X, y, sorted(A_list), alpha)
        return g

    def gamma_extend_htbs(A_list: List[int], a: int) -> float:
        A_list = sorted(A_list)
        if not A_list:
            return compute_gamma_htbs(X, y, [a], alpha)
        balls, _ = htbs_forward_state(X, y, A_list, alpha)
        return htbs_eval_add(X, y, A_list, a, balls, alpha)

    if method == "nrs":
        gamma_fn = lambda A: nrs_gamma(X, y, A)
    elif method in ("deg", "gbrs"):
        gamma_fn = lambda A: compute_gamma(
            X, y, A, "deg" if method == "deg" else "gbrs", alpha,
            nmin_override=nmin_override,
        )
    else:
        gamma_fn = None

    total = 0
    violations = 0
    deltas: List[float] = []
    for _ in range(n_subsets):
        size = int(rng.integers(1, d))
        A = sorted(rng.choice(d, size=size, replace=False).tolist())
        remaining = [a for a in range(d) if a not in A]
        if method == "htbs":
            gA = gamma_A_htbs(A)
            for a in remaining:
                total += 1
                gB = gamma_extend_htbs(A, a)
                if gB < gA - epsilon:
                    violations += 1
                    deltas.append(gA - gB)
        else:
            gA = gamma_fn(A)
            for a in remaining:
                total += 1
                gB = gamma_fn(sorted(A + [a]))
                if gB < gA - epsilon:
                    violations += 1
                    deltas.append(gA - gB)
    rate = 100.0 * violations / total if total else 0.0
    return {
        "violation_rate": rate,
        "violations": violations,
        "total_pairs": total,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "max_delta": float(np.max(deltas)) if deltas else 0.0,
        "median_delta": float(np.median(deltas)) if deltas else 0.0,
    }
