"""
Strategy 1: Real-time crypto price signal trading on Polymarket.

Polymarket crypto price markets use question text like:
  "Will BTC be above $100,000 on Dec 31?"
  "Will ETH close between $3,000 and $4,000 this week?"
  "Will SOL reach $200 before the end of March?"

Market structure (carried through the Market dataclass):
  - floor_strike: extracted lower bound (e.g. 100_000 for "above $100k")
  - cap_strike:   extracted upper bound (e.g. for range markets)
  - subtitle_yes: outcome label ("Yes" or "above $100k")

Signal: Use Coinbase real-time price + log-normal model to estimate
the true probability, compare to Polymarket's implied probability, trade edge.

Technical layer (RSI/momentum) adjusts base probability for short-term
momentum that the log-normal model misses.

Volatility: uses realized vol from recent candles when available,
falling back to the hardcoded ANNUAL_VOL baseline estimates.
"""

import logging
import math
import re
from datetime import datetime, timezone
from typing import Optional

from api_clients.binance_client import ANNUAL_VOL, BinanceClient, _norm_cdf
from api_clients.polymarket_client import Market
from strategies.signal_arb import AggregatedSignal, SignalArbConfig, ask_edge
from strategies.kelly import KellySizer

logger = logging.getLogger(__name__)

# Keywords in question text → Coinbase price symbol
_QUESTION_SYMBOL: list[tuple[list[str], str]] = [
    (["btc", "bitcoin"], "BTCUSDT"),
    (["eth", "ethereum", "ether"],  "ETHUSDT"),
    (["sol", "solana"],             "SOLUSDT"),
]


def _get_symbol(question: str) -> Optional[str]:
    """Match a crypto symbol from a market question string."""
    q = question.lower()
    for keywords, sym in _QUESTION_SYMBOL:
        if any(kw in q for kw in keywords):
            return sym
    return None


def _prob_in_range(spot, lo, hi, days, vol):
    """P(lo < price < hi) at expiry under log-normal."""
    t = max(days / 365.0, 1 / 8760)
    mu = -0.5 * vol ** 2 * t

    def d2(target):
        return (math.log(spot / target) + mu) / (vol * math.sqrt(t))

    return max(0.0, _norm_cdf(d2(lo)) - _norm_cdf(d2(hi)))


def _prob_above(spot, strike, days, vol):
    """P(price > strike) at expiry under log-normal."""
    t = max(days / 365.0, 1 / 8760)
    d2 = (
        math.log(spot / strike) - 0.5 * vol ** 2 * t
    ) / (vol * math.sqrt(t))
    return _norm_cdf(d2)


class CryptoSignalDetector:
    """
    Scans Kalshi BTC/ETH/SOL price markets and finds edge using
    Coinbase real-time prices + technical indicators.
    """

    def __init__(
        self,
        cfg: SignalArbConfig,
        binance: BinanceClient,
        bankroll: float,
    ):
        self.cfg = cfg
        self.binance = binance
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)

    def _vol(self, symbol: str) -> float:
        """Blended realized vol (30 min + 6h), falling back to baseline."""
        rv = self.binance.compute_realized_vol_blended(symbol)
        if rv:
            return rv
        return ANNUAL_VOL.get(symbol, 0.90)

    def _technical_adjustment(
        self, symbol: str, bullish_direction: bool
    ) -> tuple[float, int]:
        """
        Return (prob_adj, signal_score).
        bullish_direction=True means YES resolves if price goes UP.
        """
        sigs = self.binance.compute_signals(symbol)
        score = 0
        adj = 0.0

        rsi = sigs.get("rsi")
        if rsi is not None:
            if rsi < 30:    # oversold → bullish
                score += 1
                adj += 0.02
            elif rsi > 70:  # overbought → bearish
                score -= 1
                adj -= 0.02

        mom = sigs.get("momentum_10")
        if mom is not None:
            if mom > 0.01:
                score += 1
                adj += 0.015
            elif mom < -0.01:
                score -= 1
                adj -= 0.015

        mom30 = sigs.get("momentum_30")
        if mom30 is not None:
            if mom30 > 0.02:
                score += 1
                adj += 0.01
            elif mom30 < -0.02:
                score -= 1
                adj -= 0.01

        vwap = sigs.get("vwap_dev")
        if vwap is not None:
            if vwap > 0.005:   # price above VWAP → mean revert down
                score -= 1
                adj -= 0.01
            elif vwap < -0.005:
                score += 1
                adj += 0.01

        if not bullish_direction:
            adj = -adj
            score = -score

        return adj, score

    def scan(self, markets: list[Market]) -> list[AggregatedSignal]:
        results = []
        for market in markets:
            try:
                symbol = _get_symbol(market.question)
                if not symbol:
                    continue

                spot = self.binance.get_price(symbol)
                if not spot:
                    continue

                floor = market.floor_strike
                cap = market.cap_strike
                if floor is None and cap is None:
                    continue

                hours = (
                    market.close_time - datetime.now(timezone.utc)
                ).total_seconds() / 3600
                if hours <= 0:
                    continue
                days = hours / 24.0
                vol = self._vol(symbol)

                q_lower = market.question.lower()

                # Range market: both floor and cap present
                if floor is not None and cap is not None:
                    model_prob = _prob_in_range(
                        spot, floor, cap, days, vol
                    )
                    adj, score = 0.0, 0

                # Above/below market: only floor present
                elif floor is not None:
                    if any(
                        w in q_lower
                        for w in ("above", "over", "exceed", "higher", "more than")
                    ):
                        model_prob = _prob_above(spot, floor, days, vol)
                        adj, score = self._technical_adjustment(
                            symbol, bullish_direction=True
                        )
                    else:
                        model_prob = 1.0 - _prob_above(
                            spot, floor, days, vol
                        )
                        adj, score = self._technical_adjustment(
                            symbol, bullish_direction=False
                        )
                elif cap is not None:
                    model_prob = 1.0 - _prob_above(spot, cap, days, vol)
                    adj, score = self._technical_adjustment(
                        symbol, bullish_direction=False
                    )
                else:
                    continue

                adj_prob = max(0.01, min(0.99, model_prob + adj))

                result = ask_edge(
                    adj_prob,
                    market.yes_ask, market.no_ask, market.yes_bid,
                    self.cfg.fee_pct, self.cfg.min_edge,
                )
                if not result:
                    continue
                edge, side, entry_price = result

                trade_prob = adj_prob if side == "yes" else (1 - adj_prob)
                size = self.sizer.size(
                    trade_prob, entry_price, self.cfg.max_position_usd
                )

                rsi_val = self.binance.compute_rsi(symbol) or 0
                results.append(AggregatedSignal(
                    market_id=market.market_id,
                    question=market.question or market.market_id,
                    market_prob=market.yes_price,
                    model_prob=adj_prob,
                    edge=edge,
                    recommended_side=side,
                    source=f"coinbase:{symbol[:3]}",
                    confidence=min(1.0, abs(score) / 3.0),
                    reasoning=(
                        f"spot={spot:,.0f} "
                        f"floor={floor} cap={cap} "
                        f"base={model_prob:.2%} adj={adj:+.2%} "
                        f"RSI={rsi_val:.0f} vol={vol:.0%} "
                        f"ask={entry_price:.3f}"
                    ),
                    recommended_size_usd=size,
                ))
            except Exception as e:
                logger.debug(
                    f"CryptoSignal error {market.market_id}: {e}"
                )

        return sorted(results, key=lambda x: abs(x.edge), reverse=True)
