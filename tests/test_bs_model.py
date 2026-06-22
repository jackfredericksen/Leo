"""Unit tests for BTC 5-min Black-Scholes convergence model."""

import math

from strategies.btc_5min import BTC5MinDetector, _norm_cdf


class TestNormCdf:
    def test_zero_is_half(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 0.001

    def test_large_positive_near_one(self):
        assert _norm_cdf(3.0) > 0.99

    def test_large_negative_near_zero(self):
        assert _norm_cdf(-3.0) < 0.01

    def test_symmetry(self):
        x = 1.23
        assert abs(_norm_cdf(x) + _norm_cdf(-x) - 1.0) < 0.001


class TestBSProbUp:
    def test_positive_delta_high_probability(self):
        # T=0.5 min, +0.10% move, vol=80%
        p = BTC5MinDetector._bs_prob_up(
            window_delta_frac=0.001,
            mins_to_close=0.5,
            annual_vol=0.80,
        )
        assert p > 0.85

    def test_zero_delta_near_fifty_with_time_left(self):
        p = BTC5MinDetector._bs_prob_up(
            window_delta_frac=0.0,
            mins_to_close=2.5,
            annual_vol=0.80,
        )
        assert 0.40 < p < 0.60

    def test_negative_delta_low_probability(self):
        p = BTC5MinDetector._bs_prob_up(
            window_delta_frac=-0.002,
            mins_to_close=0.5,
            annual_vol=0.80,
        )
        assert p < 0.15

    def test_less_time_increases_certainty(self):
        p_late = BTC5MinDetector._bs_prob_up(0.001, 0.3, 0.80)
        p_early = BTC5MinDetector._bs_prob_up(0.001, 2.0, 0.80)
        assert p_late > p_early