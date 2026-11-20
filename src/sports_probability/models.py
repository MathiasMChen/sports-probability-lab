"""Inspectable probability models. All fitted state in the demo is synthetic."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import betabinom, binom, norm, poisson


def weights_at(days, cutoff: int, half_life: float) -> np.ndarray:
    """Require all training observations to precede an exclusive cutoff."""
    days = np.asarray(days, dtype=float)
    if (days.ndim != 1 or len(days) == 0 or not np.isfinite(days).all()
            or not math.isfinite(cutoff) or not math.isfinite(half_life)
            or half_life <= 0 or np.any(days >= cutoff)):
        raise ValueError("finite training dates must precede cutoff; half_life > 0")
    return np.exp2(-(cutoff - days) / half_life)


def _pair_design(home, away, n_teams):
    home, away = np.asarray(home, int), np.asarray(away, int)
    if (home.ndim != 1 or home.shape != away.shape or len(home) == 0
            or n_teams < 2 or np.any(home < 0) or np.any(away < 0)
            or np.any(home >= n_teams) or np.any(away >= n_teams)
            or np.any(home == away)):
        raise ValueError("invalid team pairing")
    x = np.zeros((len(home), n_teams))
    x[np.arange(len(home)), home] = 1
    x[np.arange(len(home)), away] = -1
    return home, away, x


def fit_strength(home, away, wins, trials, days, cutoff, n_teams,
                 half_life=365.0, penalty=2.0):
    """Weighted binomial Bradley-Terry fit with L2 shrinkage.

    Can model binary matches, frames, games or maps. Trials are supplied totals;
    this likelihood does not correct for stopping-rule/selection effects.
    """
    _, _, x = _pair_design(home, away, n_teams)
    w = weights_at(days, cutoff, half_life)
    y, n = np.asarray(wins, float), np.asarray(trials, float)
    if (y.shape != w.shape or n.shape != w.shape or len(w) != len(x)
            or not np.isfinite(y).all() or not np.isfinite(n).all()
            or np.any(n <= 0) or np.any(y < 0) or np.any(y > n)
            or np.any(y != np.floor(y)) or np.any(n != np.floor(n))
            or not math.isfinite(penalty) or penalty <= 0):
        raise ValueError("invalid binomial observations or penalty")

    def objective(theta):
        eta = x @ theta
        loss = np.sum(w * (n * np.logaddexp(0, eta) - y * eta))
        grad = x.T @ (w * (n * expit(eta) - y))
        return loss + penalty * (theta @ theta) / 2, grad + penalty * theta

    result = minimize(objective, np.zeros(n_teams), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    theta = result.x - result.x.mean()
    return theta


def series_probability(p: float, best_of: int, concentration: float | None = None):
    """First-to-majority win probability under iid or beta-binomial units.

    The mixture assumes an exchangeable match-level latent probability. It is
    a modelling assumption, not evidence that real frames are exchangeable.
    """
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be finite and in [0,1]")
    if isinstance(best_of, bool) or not isinstance(best_of, int) or best_of < 1 or best_of % 2 == 0:
        raise ValueError("best_of must be a positive odd integer")
    if concentration is not None and (not math.isfinite(concentration) or concentration <= 0):
        raise ValueError("concentration must be finite and positive")
    if p in (0, 1):
        return float(p)
    target = best_of // 2 + 1
    if concentration is None:
        return float(binom.sf(target - 1, best_of, p))
    return float(betabinom.sf(target - 1, best_of,
                            concentration * p, concentration * (1 - p)))


def football_probabilities(home_rate: float, away_rate: float, rho: float = 0.0):
    """H/D/A from Poisson scores, with feasible Dixon-Coles low-score factors.

    Use a tail-controlled grid; report discarded mass before normalizing.
    Invalid rho fails explicitly rather than being silently clipped.
    """
    if (not all(math.isfinite(v) for v in (home_rate, away_rate, rho))
            or not 0 < home_rate <= 20 or not 0 < away_rate <= 20):
        raise ValueError("rates must be finite, positive and <=20")
    factors = np.array([1 - home_rate * away_rate * rho,
                        1 + home_rate * rho, 1 + away_rate * rho, 1 - rho])
    if np.any(factors <= 0):
        raise ValueError("rho produces non-positive low-score probabilities")
    end = int(max(poisson.ppf(1 - 1e-12, home_rate),
                  poisson.ppf(1 - 1e-12, away_rate), 2))
    grid = np.outer(poisson.pmf(np.arange(end + 1), home_rate),
                    poisson.pmf(np.arange(end + 1), away_rate))
    for pos, factor in zip([(0, 0), (0, 1), (1, 0), (1, 1)], factors):
        grid[pos] *= factor
    retained = float(grid.sum())
    probs = np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])
    return probs / retained, max(0.0, 1 - retained)


def fit_margin(x, margins, days, cutoff, half_life=365.0, penalty=5.0):
    """Time-weighted ridge margin model; first column is an unpenalized intercept.

    Returns fitted coefficients and in-sample residual scale. A production model
    would select/recalibrate the scale using only earlier out-of-fold residuals.
    """
    x, y = np.asarray(x, float), np.asarray(margins, float)
    w = weights_at(days, cutoff, half_life)
    if (x.ndim != 2 or x.shape[0] != len(w) or y.shape != w.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.allclose(x[:, 0], 1) or not math.isfinite(penalty) or penalty <= 0):
        raise ValueError("invalid design, targets or penalty")
    reg = np.eye(x.shape[1]) * penalty
    reg[0, 0] = 0
    beta = np.linalg.solve(x.T @ (w[:, None] * x) + reg, x.T @ (w * y))
    sigma = float(np.sqrt(np.average((y - x @ beta) ** 2, weights=w)))
    if sigma <= 0:
        raise ValueError("non-positive residual scale")
    return beta, sigma


def margin_win_probability(x, beta, sigma):
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    mean = np.asarray(x, float) @ np.asarray(beta, float)
    if not np.isfinite(mean).all():
        raise ValueError("non-finite margin prediction")
    return norm.cdf(mean / sigma)
