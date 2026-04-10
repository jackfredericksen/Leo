"""
Strategy 1: Real-time crypto price signal trading.

Kalshi crypto price markets use tickers like:
  KXBTCY-27JAN0100-B72500   → BTC price range (floor=72500) on Jan 1 2027
  KXBTCM-26APR-B65000       → BTC monthly above $65k
  KXETHMW-26APR18-B2000     → ETH weekly above $2000

Market structure:
  - floor_strike: lower bound (the strike price)
  - cap_strike:   upper bound (for range markets)
  - strike_type:  "between" (range) or "greater" / "less_than" (above/below)

Signal: Use Coinbase real-time price + log-normal model to estimate
the true probability, compare to Kalshi's implied probability, trade edge.

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
from api_clients.kalshi_client import Market
from strategies.signal_arb import AggregatedSignal, SignalArbConfig
from strategies.kelly import KellySizer

logger = logging.getLogger(__name__)

# Ticker prefixes → Coinbase symbol
_TICKER_SYMBOL = {
    "KXBTC": "BTCUSDT",
    "KXETH": "ETHUSDT",
    "KXSOL": "SOLUSDT",
}


def _get_symbol(market_id: str) -> Optional[str]:
    uid = market_id.upper()
    for prefix, sym in _TICKER_SYMBOL.items():
        if uid.startswith(prefix):
            # Must be followed immediately by a digit, hyphen, or
            # time-period letter (M/W/D/Y/Q) — not more letters.
            # Rules out KXSOLAR (SOL+AR), KXBTCVSGOLD (BTC+VS), etc.
            after = uid[len(prefix):]
            if after and (after[0].isdigit() or after[0] in "-MWDYQ"):
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
        """Realized vol if we have enough candles, else baseline estimate."""
        rv = self.binance.compute_realized_vol(symbol)
        if rv and 0.10 < rv < 5.0:   # sanity-check the estimate
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
                symbol = _get_symbol(market.market_id)
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

                sub_yes = market.subtitle_yes.lower()
                ticker_upper = market.market_id.upper()

                # Range market: both floor and cap present
                if floor is not None and cap is not None:
                    model_prob = _prob_in_range(
                        spot, floor, cap, days, vol
                    )
                    adj, score = 0.0, 0

                # Above/below market: only floor present
                elif floor is not None:
                    if (
                        "above" in sub_yes
                        or "over" in sub_yes
                        or re.search(r"-B\d", ticker_upper)
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
                market_prob = market.yes_price
                if market_prob <= 0:
                    continue

                edge = adj_prob - market_prob - self.cfg.fee_pct
                if abs(edge) < self.cfg.min_edge:
                    continue

                side = "yes" if edge > 0 else "no"
                trade_prob = adj_prob if side == "yes" else (1 - adj_prob)
                size = self.sizer.size(
                    trade_prob, market_prob, self.cfg.max_position_usd
                )

                rsi_val = self.binance.compute_rsi(symbol) or 0
                results.append(AggregatedSignal(
                    market_id=market.market_id,
                    question=market.question or market.market_id,
                    market_prob=market_prob,
                    model_prob=adj_prob,
                    edge=edge,
                    recommended_side=side,
                    source=f"coinbase:{symbol[:3]}",
                    confidence=min(1.0, abs(score) / 3.0),
                    reasoning=(
                        f"spot={spot:,.0f} "
                        f"floor={floor} cap={cap} "
                        f"base={model_prob:.2%} adj={adj:+.2%} "
                        f"RSI={rsi_val:.0f} vol={vol:.0%}"
                    ),
                    recommended_size_usd=size,
                ))
            except Exception as e:
                logger.debug(
                    f"CryptoSignal error {market.market_id}: {e}"
                )

        return sorted(results, key=lambda x: abs(x.edge), reverse=True)
