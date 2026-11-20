"""Hypothetical independent-event fractional-Kelly path simulation."""
import math

import numpy as np
from scipy.stats import beta


def simulate_drawdown(probabilities, decimal_odds, fraction, *, seed=19, paths=10000,
                      drawdown_limit=0.30):
    """Sample wins from supplied P, not historical outcomes.

    Full Kelly fractions are clipped at zero, then scaled by fraction. Sequential
    non-overlapping bets and independent Bernoulli outcomes are assumed. This is
    a reference simulator, not the operational daily-balance/cash allocator.
    """
    p, odds = np.asarray(probabilities, float), np.asarray(decimal_odds, float)
    if (p.ndim != 1 or p.shape != odds.shape or len(p) == 0
            or not np.isfinite(p).all() or not np.isfinite(odds).all()
            or np.any(p < 0) or np.any(p > 1) or np.any(odds <= 1)
            or not math.isfinite(fraction) or not 0 <= fraction <= 1
            or paths < 2 or not 0 < drawdown_limit < 1):
        raise ValueError("invalid path simulation inputs")
    sizes = fraction * np.maximum(0, (p * odds - 1) / (odds - 1))
    rng = np.random.default_rng(seed)
    equity = np.ones(paths)
    peak, drawdown = equity.copy(), np.zeros(paths)
    for prob, price, size in zip(p, odds, sizes):
        wins = rng.random(paths) < prob
        equity *= np.where(wins, 1 + size * (price - 1), 1 - size)
        peak = np.maximum(peak, equity)
        drawdown = np.maximum(drawdown, 1 - equity / peak)
    exceedances = int(np.sum(drawdown > drawdown_limit))
    # Exact one-sided 95% Clopper-Pearson bound for the Monte Carlo event count.
    upper = 1.0 if exceedances == paths else float(beta.ppf(.95, exceedances + 1, paths - exceedances))
    return {"paths": paths, "events": len(p), "fraction": fraction,
            "max_drawdown_p95": float(np.quantile(drawdown, .95)),
            "exceedance_rate": exceedances / paths, "exceedance_upper95": upper,
            "drawdown_limit": drawdown_limit}
