"""
Strategy 3: Kalshi crypto range straddle + logical arb.

Kalshi lists weekly/monthly BTC and ETH price RANGE contracts:
  "Will BTC close between $90,000-$100,000 this week?"

These use CF Benchmarks settlement prices (same as CME BTC futures).

Strategy:
  - Use Coinbase spot + log-normal model to estimate P(price in range)
  - Compare to Kalshi's implied probability, trade edge > min_edge

Also detects cross-range logical arb:
  If P("BTC above $90k") > P("BTC above $100k") is violated, flag it.
  The lower threshold must always have >= probability as the higher.

Volatility: uses realized vol from recent candles when available,
falling back to the hardcoded ANNUAL_VOL baseline estimates.
"""

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from api_clients.binance_client import ANNUAL_VOL, BinanceClient, _norm_cdf
from api_clients.polymarket_client import Market
from strategies.signal_arb import AggregatedSignal, SignalArbConfig, ask_edge
from strategies.kelly import KellySizer

logger = logging.getLogger(__name__)

# Match "between $90,000 and $100,000" or "$90k-$100k"
_RANGE_PATTERN = re.compile(
    r"\$?([\d,]+(?:\.\d+)?)\s*[kK]?"
    r"(?:\s*(?:and|to|–|-)\s*)"
    r"\$?([\d,]+(?:\.\d+)?)\s*[kK]?",
    re.IGNORECASE,
)

_ABOVE_PATTERN = re.compile(
    r"\b(BTC|ETH|SOL)\b.{0,60}\$?([\d,]+(?:\.\d+)?)[kK]?",
    re.IGNORECASE,
)

_SYMBOL_MAP = {
    "BTC": "BTCUSDT", "Bitcoin": "BTCUSDT",
    "ETH": "ETHUSDT", "Ethereum": "ETHUSDT",
    "SOL": "SOLUSDT", "Solana": "SOLUSDT",
}


def _prob_in_range(
    spot: float, lo: float, hi: float, days: float, vol: float
) -> float:
    """P(lo < price_at_T < hi) under log-normal."""
    t = max(days / 365.0, 1 / 8760)
    mu = -0.5 * vol ** 2 * t

    def d2(target):
        return (math.log(spot / target) + mu) / (vol * math.sqrt(t))

    return max(0.0, _norm_cdf(d2(lo)) - _norm_cdf(d2(hi)))


@dataclass
class RangeOpportunity:
    market_id: str
    question: str
    symbol: str
    lo: float
    hi: float
    spot: float
    days_to_resolve: float
    model_prob: float
    market_prob: float
    edge: float
    recommended_side: str


class RangeStraddleDetector:
    """Finds edge in Kalshi crypto range contracts."""

    def __init__(
        self,
        cfg: SignalArbConfig,
        binance: BinanceClient,
        bankroll: float,
    ):
        self.cfg = cfg
        self.binance = binance
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)

    def _get_symbol(self, question: str) -> Optional[str]:
        for kw, sym in _SYMBOL_MAP.items():
            if kw.lower() in question.lower():
                return sym
        return None

    def _vol(self, symbol: str) -> float:
        """Blended realized vol (30 min + 6h), falling back to baseline."""
        rv = self.binance.compute_realized_vol_blended(symbol)
        if rv:
            return rv
        return ANNUAL_VOL.get(symbol, 0.90)

    def _parse_range(
        self, question: str
    ) -> Optional[tuple[float, float]]:
        m = _RANGE_PATTERN.search(question)
        if not m:
            return None
        try:
            lo_str = m.group(1).replace(",", "")
            hi_str = m.group(2).replace(",", "")
            lo = float(lo_str)
            hi = float(hi_str)
            raw = m.group(0)
            prefix = raw.split("and")[0].split("–")[0].split("-")[0]
            suffix = raw.split("and")[-1].split("–")[-1].split("-")[-1]
            if re.search(r"\d[kK]", prefix):
                lo *= 1000
            if re.search(r"\d[kK]", suffix):
                hi *= 1000
            if hi <= lo:
                lo, hi = hi, lo
            return lo, hi
        except Exception:
            return None

    def scan(self, markets: list[Market]) -> list[AggregatedSignal]:
        results = []

        # --- Range markets ---
        for market in markets:
            try:
                if not any(
                    w in market.question.lower()
                    for w in ["between", "range", "–", " to "]
                ):
                    continue

                symbol = self._get_symbol(market.question)
                if not symbol:
                    continue

                parsed = self._parse_range(market.question)
                if not parsed:
                    continue
                lo, hi = parsed

                spot = self.binance.get_price(symbol)
                if not spot:
                    continue

                hours = (
                    market.close_time - datetime.now(timezone.utc)
                ).total_seconds() / 3600
                if hours <= 0:
                    continue
                days = hours / 24.0
                vol = self._vol(symbol)

                model_prob = _prob_in_range(spot, lo, hi, days, vol)

                result = ask_edge(
                    model_prob,
                    market.yes_ask, market.no_ask, market.yes_bid,
                    self.cfg.fee_pct, self.cfg.min_edge,
                )
                if not result:
                    continue
                edge, side, entry_price = result

                trade_prob = model_prob if side == "yes" else (1 - model_prob)
                size = self.sizer.size(
                    trade_prob, entry_price, self.cfg.max_position_usd
                )

                results.append(AggregatedSignal(
                    market_id=market.market_id,
                    question=market.question,
                    market_prob=market.yes_price,
                    model_prob=model_prob,
                    edge=edge,
                    recommended_side=side,
                    source="range-straddle",
                    reasoning=(
                        f"spot={spot:.0f} range=[{lo:.0f},{hi:.0f}] "
                        f"model={model_prob:.2%} mkt={market.yes_price:.2%} "
                        f"ask={entry_price:.3f} vol={vol:.0%}"
                    ),
                    recommended_size_usd=size,
                ))
            except Exception as e:
                logger.debug(
                    f"RangeStraddle error on {market.market_id}: {e}"
                )

        # --- Logical arb: P(above X) >= P(above Y) when X < Y ---
        above_markets: list[tuple[str, float, Market]] = []
        for market in markets:
            symbol = self._get_symbol(market.question)
            if not symbol:
                continue
            if not any(
                w in market.question.lower()
                for w in ["above", "over", "exceed"]
            ):
                continue
            m = _ABOVE_PATTERN.search(market.question)
            if not m:
                continue
            try:
                strike = float(m.group(2).replace(",", ""))
                snippet = market.question[m.start():m.start() + 20].lower()
                if "k" in snippet[-3:]:
                    strike *= 1000
                above_markets.append((symbol, strike, market))
            except Exception:
                pass

        # Group by symbol, sort by strike ascending
        by_symbol: dict[str, list] = {}
        for sym, strike, mkt in above_markets:
            by_symbol.setdefault(sym, []).append((strike, mkt))

        for sym, items in by_symbol.items():
            items.sort(key=lambda x: x[0])
            for i in range(len(items) - 1):
                lo_strike, lo_mkt = items[i]
                hi_strike, hi_mkt = items[i + 1]
                # Logical violation: P(above lower) < P(above higher)
                spread = lo_mkt.yes_ask - lo_mkt.yes_bid
                price_violation = lo_mkt.yes_price < hi_mkt.yes_price - 0.03
                if price_violation and spread <= 0.08:
                    edge = hi_mkt.yes_price - lo_mkt.yes_ask - self.cfg.fee_pct
                    if edge >= self.cfg.min_edge:
                        size = self.sizer.size(
                            hi_mkt.yes_price,
                            lo_mkt.yes_ask,
                            self.cfg.max_position_usd,
                        )
                        results.append(AggregatedSignal(
                            market_id=lo_mkt.market_id,
                            question=lo_mkt.question,
                            market_prob=lo_mkt.yes_price,
                            model_prob=hi_mkt.yes_price,
                            edge=edge,
                            recommended_side="yes",
                            source="range-logical-arb",
                            reasoning=(
                                f"P(>{lo_strike:.0f})"
                                f"={lo_mkt.yes_price:.2%} < "
                                f"P(>{hi_strike:.0f})"
                                f"={hi_mkt.yes_price:.2%} — impossible"
                            ),
                            recommended_size_usd=size,
                        ))

        return sorted(results, key=lambda x: abs(x.edge), reverse=True)
