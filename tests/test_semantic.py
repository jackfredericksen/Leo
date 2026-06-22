"""Unit tests for semantic cross-platform matching helpers."""

from strategies.semantic_arb import (
    _direction,
    _extract_numbers,
    _jaccard,
    _keywords,
    _numbers_match,
)


class TestNumberExtraction:
    def test_dollar_amounts(self):
        nums = _extract_numbers("Will BTC exceed $100,000?")
        assert 100_000 in nums

    def test_k_suffix(self):
        nums = _extract_numbers("BTC above $90k")
        assert 90_000 in nums

    def test_multiple_numbers(self):
        nums = _extract_numbers("Between $90k and $100k")
        assert len(nums) == 2


class TestNumbersMatch:
    def test_matching_within_tolerance(self):
        assert _numbers_match([100_000], [100_500], tol=0.01)

    def test_mismatch_different_count(self):
        assert not _numbers_match([100_000], [100_000, 110_000], tol=0.01)

    def test_mismatch_beyond_tolerance(self):
        assert not _numbers_match([100_000], [110_000], tol=0.01)


class TestJaccard:
    def test_identical_keywords(self):
        a = _keywords("bitcoin price above threshold")
        b = _keywords("bitcoin price above threshold")
        assert _jaccard(a, b) == 1.0

    def test_no_overlap(self):
        a = _keywords("ethereum merge successful")
        b = _keywords("presidential election winner")
        assert _jaccard(a, b) == 0.0


class TestDirection:
    def test_above_keywords(self):
        assert _direction("Will BTC close above $90k?") == "above"

    def test_below_keywords(self):
        assert _direction("Temperature below 32 degrees") == "below"