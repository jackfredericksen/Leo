"""Unit tests for Kelly criterion position sizing."""

from strategies.kelly import KellySizer, implied_edge


class TestKellySizer:
    def test_positive_edge_returns_size(self):
        sizer = KellySizer(bankroll=1000, fraction=0.25)
        size = sizer.size(true_prob=0.65, market_price=0.50, max_size=100)
        assert size > 0
        assert size <= 100

    def test_no_edge_returns_zero(self):
        sizer = KellySizer(bankroll=1000, fraction=0.25)
        assert sizer.size(true_prob=0.40, market_price=0.50) == 0.0

    def test_invalid_price_returns_zero(self):
        sizer = KellySizer(bankroll=1000)
        assert sizer.size(true_prob=0.9, market_price=0.0) == 0.0
        assert sizer.size(true_prob=0.9, market_price=1.0) == 0.0

    def test_hard_cap_respected(self):
        sizer = KellySizer(bankroll=10_000, fraction=1.0, max_pct_of_bankroll=0.05)
        size = sizer.size(true_prob=0.80, market_price=0.40, max_size=10_000)
        assert size <= 500  # 5% of 10k

    def test_implied_edge_positive_with_model_edge(self):
        edge = implied_edge(true_prob=0.60, market_price=0.50, fee_pct=0.0)
        assert edge > 0


class TestKellyBankrollUpdate:
    def test_larger_bankroll_increases_size(self):
        small_sizer = KellySizer(bankroll=500, fraction=0.25)
        large_sizer = KellySizer(bankroll=2000, fraction=0.25)
        small = small_sizer.size(true_prob=0.70, market_price=0.50, max_size=5000)
        large = large_sizer.size(true_prob=0.70, market_price=0.50, max_size=5000)
        assert large > small