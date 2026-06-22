"""Unit tests for weather probability estimation."""

from strategies.weather_signal import (
    _prob_precip_exceeds,
    _prob_temp_exceeds,
    _temp_sigma,
)


class TestTempSigma:
    def test_same_day_tightest(self):
        assert _temp_sigma(0) == 2.0
        assert _temp_sigma(14) >= 8.0

    def test_increases_with_horizon(self):
        assert _temp_sigma(7) > _temp_sigma(1)


class TestTempProbability:
    def test_forecast_above_threshold_high_prob(self):
        p = _prob_temp_exceeds(forecast_f=80.0, threshold_f=70.0, days_out=0)
        assert p > 0.90

    def test_forecast_below_threshold_low_prob(self):
        p = _prob_temp_exceeds(forecast_f=65.0, threshold_f=75.0, days_out=0)
        assert p < 0.10

    def test_uncertainty_widens_distribution(self):
        p_near = _prob_temp_exceeds(72.0, 70.0, days_out=0)
        p_far = _prob_temp_exceeds(72.0, 70.0, days_out=14)
        assert abs(p_near - 0.5) > abs(p_far - 0.5) or p_far < p_near


class TestPrecipProbability:
    def test_dry_forecast_near_zero(self):
        p = _prob_precip_exceeds(0.0, 0.02, 0.5)
        assert p < 0.10

    def test_rain_with_low_amount_below_threshold(self):
        p = _prob_precip_exceeds(0.1, 0.6, 0.5)
        assert 0 < p < 0.6