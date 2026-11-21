"""Calibration statistics, with uncertainty that respects how the data was made.

Markets in the same event are not independent: in a 20-candidate election, exactly one
market resolves YES. So every interval here comes from a cluster bootstrap that resamples
whole events, not single markets.
"""
from __future__ import annotations

import numpy as np

EPS = 0.005  # keep prices away from exactly 0 and 1 for logs


def clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def logit(p):
    p = clip(p)
    return np.log(p / (1 - p))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(p, y) -> float:
    p, y = clip(p), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


BINS = np.array([0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0000001])


def bin_index(p, edges=BINS):
    return np.clip(np.searchsorted(edges, np.asarray(p, float), side="right") - 1, 0, len(edges) - 2)


def calibration_table(p, y, edges=BINS):
    """Per price bin: count, mean price, share that resolved YES."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = bin_index(p, edges)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        rows.append({
            "lo": float(edges[b]), "hi": float(min(edges[b + 1], 1.0)), "n": n,
            "mean_price": float(p[m].mean()) if n else None,
            "freq": float(y[m].mean()) if n else None,
        })
    return rows


def murphy(p, y, edges=BINS) -> dict:
    """Brier score split into reliability (miscalibration, lower is better), resolution (how much
    the forecasts separate outcomes, higher is better), and uncertainty (the base rate's variance).
    With forecasts grouped into bins: Brier ~= reliability - resolution + uncertainty."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = bin_index(p, edges)
    base = y.mean()
    rel = res = 0.0
    for b in np.unique(idx):
        m = idx == b
        w = m.mean()
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - base) ** 2
    return {"brier": brier(p, y), "reliability": float(rel), "resolution": float(res), "uncertainty": float(base * (1 - base))}


def fit_recalibration(p, y, iters: int = 50) -> tuple[float, float]:
    """Logistic regression of the outcome on logit(price): P(yes) = sigmoid(a + b * logit(price)).
    b = 1 and a = 0 means prices are already calibrated. b > 1 means prices are too timid (true
    chances are more extreme than the price: favorites are underpriced, longshots overpriced).
    b < 1 means prices are too extreme."""
    x, y = logit(p), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    w = np.array([0.0, 1.0])
    for _ in range(iters):  # Newton's method (IRLS)
        z = X @ w
        mu = 1 / (1 + np.exp(-z))
        g = X.T @ (y - mu)
        H = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w[0]), float(w[1])


def apply_recalibration(p, a: float, b: float):
    return 1 / (1 + np.exp(-(a + b * logit(p))))


def cluster_bootstrap(clusters, stat, n_boot: int = 1000, seed: int = 0):
    """Percentile interval for stat(indices) when rows are resampled by cluster (event).
    `clusters` is one label per row; `stat` takes an array of row indices and returns a number."""
    clusters = np.asarray(clusters)
    uniq, inverse = np.unique(clusters, return_inverse=True)
    if len(uniq) == 0:
        return float("nan"), float("nan")
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        out.append(stat(idx))
    out = np.asarray([v for v in out if v is not None and np.isfinite(v)])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
