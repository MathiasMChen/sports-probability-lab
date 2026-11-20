"""Proper scoring, paired comparisons and uncertainty diagnostics."""
from __future__ import annotations

import numpy as np


def proper_scores(probabilities, outcomes):
    p, y = np.asarray(probabilities, float), np.asarray(outcomes)
    if (p.ndim != 2 or p.shape[1] < 2 or len(p) == 0 or y.shape != (len(p),)
            or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)
            or not np.isfinite(y).all() or np.any(y != np.floor(y))
            or np.any(y < 0) or np.any(y >= p.shape[1])):
        raise ValueError("invalid probability distributions or class indices")
    y = y.astype(int)
    one_hot = np.eye(p.shape[1])[y]
    actual = p[np.arange(len(p)), y]
    return {
        "n": len(p),
        "brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1) / 2)),
        "log_loss": float(-np.log(np.maximum(actual, 1e-15)).mean()),
        "zero_probability_outcomes": int(np.sum(actual == 0)),
    }


def paired_scores(model, market, outcomes):
    """None means an unavailable distribution. Compare exactly the same rows."""
    if not (len(model) == len(market) == len(outcomes)):
        raise ValueError("unaligned paired observations")
    rows = [(p, q, y) for p, q, y in zip(model, market, outcomes)
            if p is not None and q is not None and y is not None]
    if not rows:
        return {"n": 0, "model": None, "market": None}
    p, q, y = zip(*rows)
    return {"n": len(rows), "model": proper_scores(p, y), "market": proper_scores(q, y)}


def cluster_bootstrap_difference(differences, clusters, seed=7, repeats=2000):
    """Resample whole date clusters; negative model-minus-baseline is better.

    Preserves within-date dependence, but not dependence across dates.
    """
    d, c = np.asarray(differences, float), np.asarray(clusters)
    if (d.ndim != 1 or d.shape != c.shape or len(d) == 0
            or not np.isfinite(d).all() or repeats < 100):
        raise ValueError("invalid clustered observations")
    unique = np.unique(c)
    totals = np.array([d[c == group].sum() for group in unique])
    counts = np.array([np.sum(c == group) for group in unique])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(repeats, len(unique)))
    means = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {"mean_difference": float(d.mean()), "clusters": len(unique),
            "interval95": np.quantile(means, [0.025, 0.975]).tolist()}


def calibration_bins(probabilities, outcomes, bins=10):
    p, y = np.asarray(probabilities, float), np.asarray(outcomes, float)
    proper_scores(np.column_stack([1 - p, p]), y)
    labels = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": i, "n": int(np.sum(labels == i)),
             "predicted": float(p[labels == i].mean()),
             "observed": float(y[labels == i].mean())}
            for i in range(bins) if np.any(labels == i)]
