"""
Coinbase Advanced API client for real-time crypto price signals.

Uses Coinbase public endpoints (no auth needed):
  - /products/{pair} for current price
  - /products/{pair}/candles for OHLCV history

Replaces Binance (geo-blocked in the US).
"""

import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

COINBASE_REST = "https://api.coinbase.com/api/v3/brokerage/market"

# Map internal symbol names to Coinbase product IDs
SYMBOL_TO_PAIR = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
}

# Fallback annualized volatility estimates (used when candle data is sparse)
ANNUAL_VOL = {"BTCUSDT": 0.85, "ETHUSDT": 1.05, "SOLUSDT": 1.30}


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime


@dataclass
class CryptoSnapshot:
    symbol: str          # e.g. "BTCUSDT"
    price: float
    bid: float
    ask: float
    volume_24h: float
    price_change_pct: float
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class BinanceClient:
    """
    Real-time crypto data from Coinbase Advanced API.
    Named BinanceClient for compatibility with existing strategy code.
    """

    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # 360 one-minute candles = 6 hours of history for vol blending
    _CANDLE_MAXLEN = 360

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._snapshots: dict[str, CryptoSnapshot] = {}
        self._candles: dict[str, deque[Candle]] = {
            s: deque(maxlen=self._CANDLE_MAXLEN) for s in self.SYMBOLS
        }

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    async def _fetch_one_snapshot(self, symbol: str):
        """Fetch price snapshot for a single symbol."""
        pair = SYMBOL_TO_PAIR[symbol]
        try:
            url = f"{COINBASE_REST}/products/{pair}"
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                price = float(data.get("price", 0) or 0)
                if price == 0:
                    return
                self._snapshots[symbol] = CryptoSnapshot(
                    symbol=symbol,
                    price=price,
                    bid=float(data.get("bid", price) or price),
                    ask=float(data.get("ask", price) or price),
                    volume_24h=float(data.get("volume", 0) or 0),
                    price_change_pct=float(
                        data.get("price_percentage_change_24h", 0) or 0
                    ),
                )
        except Exception as e:
            logger.warning(f"Coinbase snapshot fetch failed ({symbol}): {e}")

    async def fetch_snapshots(self):
        """Fetch latest price for all symbols in parallel."""
        if not self._session:
            return
        await asyncio.gather(
            *[self._fetch_one_snapshot(s) for s in self.SYMBOLS]
        )

    async def fetch_candles(self, symbol: str, limit: int = 60):
        """
        Fetch recent 1-minute OHLCV candles from Coinbase and accumulate
        them into the rolling buffer.  Only candles newer than the most
        recent buffered timestamp are appended, preventing duplicates.
        """
        if not self._session:
            return
        pair = SYMBOL_TO_PAIR.get(symbol)
        if not pair:
            return
        try:
            url = (
                f"{COINBASE_REST}/products/{pair}/candles"
                f"?granularity=ONE_MINUTE&limit={limit}"
            )
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                raw = data.get("candles", [])

            buf = self._candles[symbol]
            last_ts = buf[-1].timestamp if buf else None

            # Coinbase returns newest-first; reverse to oldest-first
            for c in reversed(raw):
                ts = datetime.fromtimestamp(int(c["start"]), tz=timezone.utc)
                if last_ts is not None and ts <= last_ts:
                    continue   # already have this candle
                buf.append(Candle(
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                    volume=float(c["volume"]),
                    timestamp=ts,
                ))
        except Exception as e:
            logger.warning(f"Coinbase candle fetch failed ({symbol}): {e}")

    async def refresh_all(self):
        """Refresh prices + candles for all symbols in parallel."""
        if not self._session:
            return
        await asyncio.gather(
            self.fetch_snapshots(),
            *[self.fetch_candles(s) for s in self.SYMBOLS],
        )

    def get_price(self, symbol: str) -> Optional[float]:
        snap = self._snapshots.get(symbol)
        return snap.price if snap else None

    def get_snapshot(self, symbol: str) -> Optional[CryptoSnapshot]:
        return self._snapshots.get(symbol)

    def compute_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        candles = list(self._candles.get(symbol, []))
        if len(candles) < period + 1:
            return None
        closes = [c.close for c in candles[-(period + 1):]]
        gains, losses = [], []
        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def compute_realized_vol(
        self, symbol: str, window: int = 30
    ) -> Optional[float]:
        """
        Annualized realized volatility from recent 1-min candles.
        Uses log close-to-close returns, annualized by sqrt(525_600 min/yr).
        Falls back to None when candle data is insufficient.
        """
        candles = list(self._candles.get(symbol, []))[-(window + 1):]
        if len(candles) < window + 1:
            return None
        closes = [c.close for c in candles]
        returns = [
            math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        n = len(returns)
        if n < 2:
            return None
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        return math.sqrt(variance * 525_600)

    def compute_realized_vol_blended(self, symbol: str) -> Optional[float]:
        """
        Blend short-window (30 min) and medium-window (up to 360 min / 6h)
        realized vol to reduce noise.

        Short-window vol reacts quickly to regime changes; medium-window vol
        smooths out single-candle spikes.  We weight medium more heavily
        (70/30) because a 30-minute sample has high estimation variance.

        Returns None if there isn't enough data for even the short window.
        """
        short = self.compute_realized_vol(symbol, window=30)
        if short is None:
            return None

        # Use however many candles we have for the medium window (cap at 360)
        buf_len = len(self._candles.get(symbol, []))
        med_window = min(buf_len - 1, 360)
        if med_window >= 60:
            medium = self.compute_realized_vol(symbol, window=med_window)
        else:
            medium = None

        if medium is None or not (0.10 < medium < 5.0):
            return short if 0.10 < short < 5.0 else None

        blended = 0.30 * short + 0.70 * medium
        return blended if 0.10 < blended < 5.0 else None

    def get_candle_close_at(self, symbol: str, target_dt: datetime) -> Optional[float]:
        """Return the close price of the 1-min candle closest to target_dt.
        Returns None if no candle is within 120 seconds of the target."""
        candles = list(self._candles.get(symbol, []))
        if not candles:
            return None
        best = min(candles, key=lambda c: abs((c.timestamp - target_dt).total_seconds()))
        if abs((best.timestamp - target_dt).total_seconds()) > 120:
            return None
        return best.close

    def compute_mtf_momentum(self, symbol: str) -> Optional[float]:
        """
        Multi-timeframe momentum: weighted average of 1/2/3/5-candle lookbacks.

        Weights are the v3 schedule from live-trading research (Jung-Hua Liu),
        biased toward the 2–3 minute lookback rather than the noisiest 30s horizon:
          1-candle (~1min): 0.20
          2-candle (~2min): 0.25
          3-candle (~3min): 0.30   ← primary signal
          5-candle (~5min): 0.25

        The original v2 schedule (0.40/0.30/0.20/0.10, short-biased) was found
        to overfit micro-noise and produced live win rates of ~25% despite strong
        backtests — the v3 schedule was the fix.

        Returns a score normalized to [-1, 1] where 1% BTC move = ±1.0.
        Missing lookbacks are skipped and remaining weights are renormalized.
        """
        lookbacks = [1, 2, 3, 5]
        weights   = [0.20, 0.25, 0.30, 0.25]
        total_w = 0.0
        score = 0.0
        for lb, w in zip(lookbacks, weights):
            m = self.compute_momentum(symbol, lookback=lb)
            if m is not None:
                score += w * max(-1.0, min(1.0, m / 0.01))
                total_w += w
        if total_w == 0:
            return None
        return score / total_w

    def compute_momentum(self, symbol: str, lookback: int = 10) -> Optional[float]:
        candles = list(self._candles.get(symbol, []))
        if len(candles) < lookback + 1:
            return None
        old = candles[-(lookback + 1)].close
        new = candles[-1].close
        if old == 0:
            return None
        return (new - old) / old

    def compute_vwap_deviation(self, symbol: str) -> Optional[float]:
        candles = list(self._candles.get(symbol, []))[-20:]
        if len(candles) < 5:
            return None
        total_vol = sum(c.volume for c in candles)
        if total_vol == 0:
            return None
        vwap = sum(
            ((c.high + c.low + c.close) / 3) * c.volume for c in candles
        ) / total_vol
        snap = self._snapshots.get(symbol)
        if not snap or vwap == 0:
            return None
        return (snap.price - vwap) / vwap

    async def compute_obi(self, symbol: str, levels: int = 10) -> Optional[float]:
        """
        Order Book Imbalance from Coinbase L2 book.
        Returns (bid_depth - ask_depth) / (bid_depth + ask_depth) in [-1, 1].
        Positive = more bids = buying pressure (bullish for 5-min prediction).
        """
        pair = SYMBOL_TO_PAIR.get(symbol)
        if not pair or not self._session:
            return None
        try:
            url = f"{COINBASE_REST}/products/{pair}/book?limit={levels}"
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            pricebook = data.get("pricebook", {})
            bids = pricebook.get("bids", [])
            asks = pricebook.get("asks", [])
            bid_depth = sum(float(b["size"]) for b in bids[:levels])
            ask_depth = sum(float(a["size"]) for a in asks[:levels])
            total = bid_depth + ask_depth
            if total == 0:
                return None
            return (bid_depth - ask_depth) / total
        except Exception as e:
            logger.warning(f"Coinbase OBI fetch failed ({symbol}): {e}")
            return None

    def compute_signals(self, symbol: str) -> dict:
        return {
            "price": self.get_price(symbol),
            "rsi": self.compute_rsi(symbol),
            "momentum_10": self.compute_momentum(symbol, 10),
            "momentum_30": self.compute_momentum(symbol, 30),
            "vwap_dev": self.compute_vwap_deviation(symbol),
            "price_change_24h": (
                self._snapshots[symbol].price_change_pct
                if symbol in self._snapshots else None
            ),
        }

    def estimate_prob_above(
        self,
        symbol: str,
        target: float,
        days_to_resolve: float,
        annual_vol: float = 0.80,
    ) -> Optional[float]:
        price = self.get_price(symbol)
        if not price or price <= 0 or target <= 0:
            return None
        t = max(days_to_resolve / 365.0, 1 / 8760)
        d2 = (
            math.log(price / target) - 0.5 * annual_vol ** 2 * t
        ) / (annual_vol * math.sqrt(t))
        return _norm_cdf(d2)


def _norm_cdf(x: float) -> float:
    a1, a2, a3, a4, a5 = (
        0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    )
    p = 0.2316419
    k = 1.0 / (1.0 + p * abs(x))
    poly = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
    cdf = 1.0 - (1.0 / (2 * math.pi) ** 0.5) * math.exp(-0.5 * x ** 2) * poly
    return float(cdf if x >= 0 else 1.0 - cdf)
