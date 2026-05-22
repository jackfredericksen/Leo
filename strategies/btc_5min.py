"""
BTC Directional Strategy

Targets Polymarket's daily BTC up/down markets, e.g. "Bitcoin Up or Down on [date]?"

Signals (all sourced from Coinbase real-time data):
  1. Order Book Imbalance (OBI)  — weight 0.50  — Coinbase L2 top-10 levels
  2. 3-minute price momentum     — weight 0.30  — 1-min candle close-to-close
  3. RSI correction              — weight 0.20  — RSI < 40 = oversold (bullish),
                                                   RSI > 60 = overbought (bearish)

Model: prob_up = 0.5 + tanh(0.80 * composite) * 0.25
  → max ±25 percentage-point deviation from 50%, scaled smoothly by signal strength

Entry window: 1.5 ≤ minutes_to_close ≤ 4.0 (configurable via BTC5MIN_MIN/MAX_MINS)
  → targets the live 5-min window; pre-created future windows are filtered out
  → signals are updated every BTC5MIN_REFRESH_SEC seconds (default 20s)

Kelly fraction: 0.08 — conservative for daily binary, limits per-trade to 8%
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from strategies.kelly import KellySizer
from strategies.signal_arb import AggregatedSignal

logger = logging.getLogger(__name__)


@dataclass
class BTC5MinConfig:
    enabled: bool = True
    min_edge: float = 0.04          # 4% net after 2% fees
    max_position_usd: float = 50.0
    kelly_fraction: float = 0.08
    min_mins_to_close: float = 1.5
    max_mins_to_close: float = 4.0
    fee_pct: float = 0.02
    obi_levels: int = 10
    obi_weight: float = 0.50
    momentum_lookback: int = 3      # candles = 3 minutes
    momentum_weight: float = 0.30
    rsi_period: int = 14
    rsi_weight: float = 0.20
    refresh_interval_sec: int = 20


def _is_btc_5min(question: str, slug: str = "") -> bool:
    """
    Match Polymarket 5-minute BTC up/down rolling markets.

    Primary: event slug starts with 'btc-updown-5m-' (the reliable identifier;
    Polymarket uses 'btc-updown-4h-' for 4h, 'btc-updown-15m-' for 15m, etc.).
    Fallback (no slug): question contains 'bitcoin'/'btc' + '5 min' literal.
    """
    if slug:
        return slug.startswith("btc-updown-5m-")
    # slug not yet available — narrow question match to avoid catching 4h / daily variants
    q = question.lower()
    has_btc = "bitcoin" in q or " btc" in q or q.startswith("btc")
    return has_btc and ("5 min" in q or "5-min" in q or "5min" in q)


def _mins_to_close(close_time: datetime) -> float:
    return (close_time - datetime.now(timezone.utc)).total_seconds() / 60


class BTC5MinDetector:
    def __init__(self, cfg: BTC5MinConfig, binance, bankroll: float = 500.0):
        self._cfg = cfg
        self._binance = binance
        self.sizer = KellySizer(
            bankroll=bankroll,
            fraction=cfg.kelly_fraction,
        )

    async def scan(self, markets: list) -> list[AggregatedSignal]:
        cfg = self._cfg
        candidates = [
            m for m in markets
            if _is_btc_5min(m.question, getattr(m, "slug", ""))
            and cfg.min_mins_to_close
               <= _mins_to_close(m.close_time)
               <= cfg.max_mins_to_close
        ]
        if not candidates:
            return []

        # Fetch microstructure signals once for all candidates
        obi = await self._binance.compute_obi("BTCUSDT", levels=cfg.obi_levels)
        momentum = self._binance.compute_momentum(
            "BTCUSDT", lookback=cfg.momentum_lookback
        )
        rsi = self._binance.compute_rsi("BTCUSDT", period=cfg.rsi_period)

        results = []
        for market in candidates:
            sig = self._evaluate(market, obi, momentum, rsi)
            if sig:
                results.append(sig)

        results.sort(key=lambda s: s.edge, reverse=True)
        return results

    def _evaluate(
        self,
        market,
        obi: Optional[float],
        momentum: Optional[float],
        rsi: Optional[float],
    ) -> Optional[AggregatedSignal]:
        cfg = self._cfg

        if obi is None:
            return None

        # OBI in [-1, 1]: already normalised, no scaling needed
        obi_score = obi

        # Momentum: normalise 3-min return — clip at ±1% (typical 3-min BTC swing)
        mom_score = 0.0
        if momentum is not None:
            mom_score = max(-1.0, min(1.0, momentum / 0.01))

        # RSI correction: RSI > 60 → bearish signal, RSI < 40 → bullish signal
        rsi_score = 0.0
        if rsi is not None:
            if rsi > 60:
                rsi_score = -((rsi - 60) / 40)   # -1 at RSI = 100
            elif rsi < 40:
                rsi_score = (40 - rsi) / 40       # +1 at RSI = 0

        composite = (
            cfg.obi_weight * obi_score
            + cfg.momentum_weight * mom_score
            + cfg.rsi_weight * rsi_score
        )

        prob_up = 0.5 + math.tanh(0.80 * composite) * 0.25

        yes_ask = getattr(market, "yes_ask", 0.0)
        no_ask  = getattr(market, "no_ask",  0.0)
        if not yes_ask or not no_ask:
            return None

        if prob_up >= 0.5:
            side = "yes"
            market_prob = yes_ask
            true_prob = prob_up
            edge = prob_up - yes_ask - cfg.fee_pct
            entry_price = yes_ask
        else:
            side = "no"
            market_prob = 1.0 - no_ask  # market's implied P(up) via NO
            true_prob = 1.0 - prob_up
            edge = true_prob - no_ask - cfg.fee_pct
            entry_price = no_ask

        if edge < cfg.min_edge:
            return None

        size = self.sizer.size(true_prob, entry_price, cfg.max_position_usd)

        rsi_str = f" RSI={rsi:.0f}" if rsi is not None else ""
        mom_str = f" mom={momentum*100:+.2f}%" if momentum is not None else ""
        reasoning = f"OBI={obi:+.3f}{mom_str}{rsi_str} composite={composite:+.3f}"

        return AggregatedSignal(
            market_id=market.market_id,
            question=market.question,
            market_prob=market_prob,
            model_prob=prob_up,
            edge=edge,
            recommended_side=side,
            source="btc_5min",
            confidence=min(1.0, abs(composite) * 2),
            reasoning=reasoning,
            recommended_size_usd=size,
            slug=getattr(market, "slug", ""),
            detected_at=datetime.now(timezone.utc),
        )
