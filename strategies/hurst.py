"""
Hurst Exponent — market regime classification via R/S analysis.

H > 0.55: trending (persistent, follow momentum)
H < 0.45: mean-reverting (anti-persistent, fade moves)
H ≈ 0.50: random walk (no structural edge)

Feed price observations into HurstTracker.update(); call regime() to gate trades.
Uses a rolling in-memory price buffer per market — no persistent storage needed.
"""

import math
from collections import deque
from enum import Enum
from typing import Optional


class MarketRegime(Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    RANDOM = "random"


def _rs_stat(log_returns: list[float]) -> Optional[float]:
    """R/S statistic for a single segment of log returns."""
    n = len(log_returns)
    if n < 4:
        return None
    mean_val = sum(log_returns) / n
    deviations = [x - mean_val for x in log_returns]
    cumsum, s = [], 0.0
    for d in deviations:
        s += d
        cumsum.append(s)
    r = max(cumsum) - min(cumsum)
    variance = sum(x * x for x in deviations) / n
    std = math.sqrt(variance) if variance > 0 else 0.0
    return r / std if std > 1e-12 else None


def hurst_exponent(prices: list[float]) -> Optional[float]:
    """
    Estimate Hurst exponent via R/S analysis at 3 lag scales.
    Returns None if price series is too short or degenerate.
    """
    if len(prices) < 20:
        return None
    log_returns = []
    for i in range(1, len(prices)):
        if prices[i] > 0 and prices[i - 1] > 0:
            log_returns.append(math.log(prices[i] / prices[i - 1]))
    if len(log_returns) < 10:
        return None

    lags, rs_vals = [], []
    for frac in (0.25, 0.50, 1.0):
        lag = max(4, int(len(log_returns) * frac))
        chunks = [
            log_returns[i: i + lag]
            for i in range(0, len(log_returns) - lag + 1, lag)
        ]
        chunk_rs = [_rs_stat(c) for c in chunks if len(c) >= lag]
        chunk_rs = [v for v in chunk_rs if v is not None and v > 0]
        if chunk_rs:
            lags.append(math.log(lag))
            rs_vals.append(math.log(sum(chunk_rs) / len(chunk_rs)))

    if len(lags) < 2:
        return None

    n = len(lags)
    mx = sum(lags) / n
    my = sum(rs_vals) / n
    num = sum((lags[i] - mx) * (rs_vals[i] - my) for i in range(n))
    den = sum((lags[i] - mx) ** 2 for i in range(n))
    return num / den if den > 1e-12 else None


def classify_regime(h: Optional[float]) -> MarketRegime:
    if h is None:
        return MarketRegime.RANDOM
    if h > 0.55:
        return MarketRegime.TRENDING
    if h < 0.45:
        return MarketRegime.MEAN_REVERTING
    return MarketRegime.RANDOM


class HurstTracker:
    """
    Per-market rolling price history and cached regime classification.
    update() is O(1). regime() returns immediately from the cache.
    Regime is recomputed from scratch each time a new price is added once
    the buffer is ≥ 20 observations.
    """

    def __init__(self, window: int = 50):
        self._window = window
        self._prices: dict[str, deque] = {}
        self._cache: dict[str, MarketRegime] = {}
        self._hurst: dict[str, Optional[float]] = {}

    def update(self, market_id: str, price: float) -> None:
        if market_id not in self._prices:
            self._prices[market_id] = deque(maxlen=self._window)
        self._prices[market_id].append(price)
        buf = self._prices[market_id]
        if len(buf) >= 20:
            h = hurst_exponent(list(buf))
            self._hurst[market_id] = h
            self._cache[market_id] = classify_regime(h)

    def regime(self, market_id: str) -> MarketRegime:
        return self._cache.get(market_id, MarketRegime.RANDOM)

    def hurst(self, market_id: str) -> Optional[float]:
        return self._hurst.get(market_id)

    def summary(self, limit: int = 20) -> list[dict]:
        items = [
            {
                "market_id": mid,
                "hurst": round(h, 3) if h is not None else None,
                "regime": self._cache.get(mid, MarketRegime.RANDOM).value,
            }
            for mid, h in self._hurst.items()
            if h is not None
        ]
        items.sort(key=lambda x: abs((x["hurst"] or 0.5) - 0.5), reverse=True)
        return items[:limit]
