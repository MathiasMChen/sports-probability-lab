import math
import unittest

import numpy as np
from scipy.special import expit

from sports_probability.demo import run
from sports_probability.evaluation import cluster_bootstrap_difference, paired_scores, proper_scores
from sports_probability.models import fit_margin, fit_strength, football_probabilities, series_probability, weights_at
from sports_probability.risk import simulate_drawdown


class ProbabilityTests(unittest.TestCase):
    def test_cutoff_is_exclusive(self):
        np.testing.assert_allclose(weights_at([0, 10], 20, 10), [.25, .5])
        with self.assertRaises(ValueError):
            weights_at([0, 20], 20, 10)

    def test_strength_recovers_direction_and_identifiability(self):
        theta = fit_strength([0,0,1], [1,2,2], [80,90,65], [100]*3, [0,1,2], 3, 3)
        self.assertGreater(theta[0], theta[1])
        self.assertGreater(theta[1], theta[2])
        self.assertAlmostEqual(theta.sum(), 0)
        self.assertGreater(expit(theta[0] - theta[2]), .75)

    def test_strength_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            fit_strength([0], [1], [2], [1], [0], 1, 2)

    def test_series_symmetry_and_boundaries(self):
        for n in (1, 5, 11, 19):
            for kappa in (None, 20):
                self.assertAlmostEqual(series_probability(.5, n, kappa), .5)
                self.assertAlmostEqual(series_probability(.7, n, kappa) + series_probability(.3, n, kappa), 1)
                self.assertEqual(series_probability(0, n, kappa), 0)
                self.assertEqual(series_probability(1, n, kappa), 1)
        self.assertGreater(series_probability(.6, 11), .6)
        self.assertLess(series_probability(.6, 11, 20), series_probability(.6, 11))

    def test_series_rejects_even_format(self):
        with self.assertRaises(ValueError):
            series_probability(.6, 6)

    def test_football_normalization_swap_and_tail(self):
        p, tail = football_probabilities(1.6, 1.1, -.08)
        q, _ = football_probabilities(1.1, 1.6, -.08)
        np.testing.assert_allclose(p, q[::-1], atol=1e-12)
        self.assertAlmostEqual(sum(p), 1)
        self.assertTrue(np.all(p > 0))
        self.assertLess(tail, 3e-12)

    def test_draw_correction_direction(self):
        independent, _ = football_probabilities(1.6, 1.1)
        corrected, _ = football_probabilities(1.6, 1.1, -.08)
        self.assertGreater(corrected[1], independent[1])
        with self.assertRaises(ValueError):
            football_probabilities(1.6, 1.1, 2)

    def test_margin_recovers_slope(self):
        rng = np.random.default_rng(7)
        z = np.linspace(-2, 2, 200)
        x = np.column_stack([np.ones(200), z])
        b, sigma = fit_margin(x, 2 + 4*z + rng.normal(0,.1,200), np.arange(200), 201,
                              half_life=10000, penalty=.01)
        np.testing.assert_allclose(b, [2,4], atol=.03)
        self.assertLess(sigma, .2)

    def test_brier_convention(self):
        s = proper_scores([[.2,.8]], [1])
        self.assertAlmostEqual(s['brier'], .04)
        self.assertAlmostEqual(s['log_loss'], -math.log(.8))
        self.assertAlmostEqual(proper_scores([[1/3]*3], [0])['brier'], 1/3)

    def test_missing_market_is_excluded_from_both_paired_scores(self):
        p = [[.2,.8], [.9,.1], None]
        q = [[.4,.6], None, [.5,.5]]
        pair = paired_scores(p, q, [1,0,1])
        self.assertEqual(pair['n'], 1)
        self.assertAlmostEqual(pair['model']['brier'], .04)
        self.assertAlmostEqual(pair['market']['brier'], .16)
        self.assertIsNone(paired_scores([None], [None], [1])['model'])

    def test_invalid_probabilities_fail_and_zero_is_disclosed(self):
        with self.assertRaises(ValueError):
            proper_scores([[.6,.6]], [1])
        s = proper_scores([[1,0]], [1])
        self.assertEqual(s['zero_probability_outcomes'], 1)
        self.assertGreater(s['log_loss'], 30)

    def test_cluster_bootstrap_uses_observation_weighted_mean(self):
        s = cluster_bootstrap_difference([1,1,-1], [0,0,1], repeats=100)
        self.assertEqual(s['clusters'], 2)
        self.assertAlmostEqual(s['mean_difference'], 1/3)

    def test_zero_kelly_has_no_drawdown(self):
        s = simulate_drawdown([.5]*10, [2]*10, .5, paths=100)
        self.assertEqual(s['max_drawdown_p95'], 0)
        self.assertEqual(s['exceedance_rate'], 0)
        self.assertGreater(s['exceedance_upper95'], 0)

    def test_risk_repeatability(self):
        a = simulate_drawdown([.6]*20, [2]*20, .5, paths=100)
        self.assertEqual(a, simulate_drawdown([.6]*20, [2]*20, .5, paths=100))
        self.assertGreater(a['max_drawdown_p95'], 0)

    def test_demo_temporal_integrity_and_repeatability(self):
        a, svg = run()
        b, _ = run()
        self.assertEqual(a, b)
        self.assertEqual(a['data_kind'], 'SYNTHETIC_ONLY')
        self.assertEqual(a['binary_strength']['all_model']['n'], 600)
        self.assertLess(a['binary_strength']['paired']['n'], 600)
        for fold in a['walk_forward_folds']:
            self.assertEqual(fold['exclusive_training_cutoff_day'], fold['first_test_day'])
        self.assertTrue(svg.startswith('<svg'))


if __name__ == '__main__':
    unittest.main()
