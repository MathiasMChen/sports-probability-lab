"""Deterministic, offline demonstration with explicitly synthetic observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from .evaluation import calibration_bins, cluster_bootstrap_difference, paired_scores, proper_scores
from .models import fit_margin, fit_strength, football_probabilities, margin_win_probability, series_probability
from .risk import simulate_drawdown


def _calibration_svg(bins):
    """Dependency-light vector figure. Bubble area is proportional to sample N."""
    bits = ['<svg xmlns="http://www.w3.org/2000/svg" width="620" height="440" viewBox="0 0 620 440">',
            '<rect width="620" height="440" fill="#ffffff"/>',
            '<g font-family="sans-serif" fill="#172e38">',
            '<text x="55" y="30" font-size="19" font-weight="bold">Walk-forward calibration</text>',
            '<text x="55" y="51" font-size="12">SYNTHETIC data · binary strength model · bubble area ∝ N</text>']
    for i in range(6):
        t = i / 5
        x, y = 65 + 475 * t, 350 - 265 * t
        bits += [f'<path d="M {x} 85 V 350 M 65 {y} H 540" stroke="#e1e8eb" fill="none"/>',
                 f'<text x="{x}" y="371" text-anchor="middle" font-size="12">{t:.1f}</text>',
                 f'<text x="52" y="{y+4}" text-anchor="end" font-size="12">{t:.1f}</text>']
    bits += ['<path d="M 65 350 L 540 85" stroke="#788b94" stroke-dasharray="5,5" fill="none"/>']
    for b in bins:
        x, y = 65 + 475 * b['predicted'], 350 - 265 * b['observed']
        bits += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{np.sqrt(b["n"])*1.1:.2f}" fill="#227d85" fill-opacity="0.7"><title>N={b["n"]}</title></circle>']
    bits += ['<text x="302" y="405" text-anchor="middle" font-size="14">Predicted win probability</text>',
             '<text transform="translate(18 220) rotate(-90)" text-anchor="middle" font-size="14">Observed win frequency</text>',
             '</g></svg>']
    return '\n'.join(bits)


def run(seed=2026):
    rng = np.random.default_rng(seed)
    n, teams = 1200, 10
    day = np.repeat(np.arange(400), 3)
    home = rng.integers(0, teams, n)
    away = (home + rng.integers(1, teams, n)) % teams
    truth = np.linspace(-0.9, 0.9, teams)
    true_p = expit(truth[home] - truth[away])
    outcome = (rng.random(n) < true_p).astype(int)
    # This noisy comparator is generated, not bookmaker data.
    benchmark = expit(truth[home] - truth[away] + rng.normal(0, .55, n))
    rest = rng.uniform(-2, 2, n)
    pair = np.eye(teams)[home] - np.eye(teams)[away]
    x = np.column_stack([np.ones(n), pair, rest])
    margin = 2.5 + pair @ (truth * 5) + .4 * rest + rng.normal(0, 12, n)
    strength_p, margin_p, indices, folds = [], [], [], []
    for cutoff in (200, 250, 300, 350):
        train = day < cutoff
        test = (day >= cutoff) & (day < cutoff + 50)
        theta = fit_strength(home[train], away[train], outcome[train],
                             np.ones(train.sum()), day[train], cutoff, teams)
        strength_p.extend(expit(theta[home[test]] - theta[away[test]]))
        beta, sigma = fit_margin(x[train], margin[train], day[train], cutoff)
        margin_p.extend(margin_win_probability(x[test], beta, sigma))
        indices.extend(np.flatnonzero(test))
        folds.append({"exclusive_training_cutoff_day": cutoff,
                      "train_n": int(train.sum()), "test_n": int(test.sum()),
                      "first_test_day": int(day[test].min()),
                      "last_test_day": int(day[test].max())})
    idx, p = np.array(indices), np.array(strength_p)
    y, q = outcome[idx], benchmark[idx]
    distributions = np.column_stack([1-p, p])
    market = [None if j % 7 == 0 else [1-qi, qi] for j, qi in enumerate(q)]
    paired = paired_scores(distributions.tolist(), market, y.tolist())
    all_model = proper_scores(distributions, y)
    benchmark_scores = proper_scores(np.column_stack([1-q, q]), y)
    # Every row here has both probabilities; dates are the resampling clusters.
    diffs = -np.log(np.where(y == 1, p, 1-p)) + np.log(np.where(y == 1, q, 1-q))
    calibration = calibration_bins(p, y)
    football, discarded = football_probabilities(1.55, 1.10, -.08)
    report = {
        "data_kind": "SYNTHETIC_ONLY", "seed": seed,
        "interpretation": "Software demonstration, not historical sports performance or evidence of profit.",
        "walk_forward_folds": folds,
        "binary_strength": {"all_model": all_model, "fair_coin_baseline": proper_scores(np.full((len(y),2), .5), y),
                            "synthetic_noisy_benchmark_all": benchmark_scores,
                            "paired": paired, "calibration": calibration,
                            "log_loss_difference_all_rows": cluster_bootstrap_difference(diffs, day[idx])},
        "basketball_margin": {"scores": proper_scores(np.column_stack([1-np.array(margin_p), margin_p]),
                                                       (margin[idx] > 0).astype(int)),
                              "caveat": "A simplified team/rest model; excludes production player/minute/clutch components."},
        "football_example": {"illustrative_rates": [1.55, 1.10], "rho": -.08,
                             "home_draw_away": football.tolist(), "discarded_tail_mass": discarded},
        "series_example": {"unit_p": .56, "best_of": 11,
                           "iid_probability": series_probability(.56, 11),
                           "beta_binomial_probability_kappa20": series_probability(.56, 11, 20)},
        "risk_examples": [simulate_drawdown(np.full(60, .55), np.full(60, 2.0), f)
                          for f in (.1, .25, .5)],
        "environment_policy": "No network access, credentials, production parameters or external integrations are used by the demo."
    }
    return report, _calibration_svg(calibration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report, svg = run()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "synthetic_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output / "synthetic_calibration.svg").write_text(svg)
    print(f"Wrote {args.output / 'synthetic_report.json'} (SYNTHETIC ONLY)")


if __name__ == "__main__":
    main()
