"""
Arbitrage detection engine for Coinbase Predictions.

Two arbitrage types:
  1. Over-round arb  — YES + NO prices sum < 1.0 (after fees), meaning you can
                       buy both sides for less than the guaranteed $1 payout.
  2. Mispricing arb  — A single side is priced well below its implied probability,
                       representing an edge even without locking both legs.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from api_clients.coinbase_predictions import Market
from config import ArbitrageConfig

logger = logging.getLogger(__name__)


@dataclass
class ArbOpportunity:
    market_id: str
    question: str
    arb_type: str           # "overround" | "mispricing"
    leg_yes: Optional[float]    # price to buy YES at (None if not part of trade)
    leg_no: Optional[float]     # price to buy NO at (None if not part of trade)
    gross_profit_pct: float     # before fees
    net_profit_pct: float       # after fees
    max_size_usd: float         # suggested max position size
    resolves_at: datetime
    detected_at: datetime


class ArbitrageDetector:
    def __init__(self, cfg: ArbitrageConfig):
        self.cfg = cfg

    def scan(self, markets: list[Market]) -> list[ArbOpportunity]:
        """Scan a list of markets and return detected opportunities."""
        opportunities = []
        for market in markets:
            if not self._is_eligible(market):
                continue
            opp = self._check_overround(market)
            if opp:
                opportunities.append(opp)
        opportunities.sort(key=lambda o: o.net_profit_pct, reverse=True)
        return opportunities

    def _is_eligible(self, market: Market) -> bool:
        if market.status != "open":
            return False
        if market.liquidity_usd < self.cfg.min_liquidity_usd:
            return False
        now = datetime.now(timezone.utc)
        hours_left = (market.resolves_at - now).total_seconds() / 3600
        if hours_left < self.cfg.min_hours_to_resolve:
            return False
        if hours_left > self.cfg.max_hours_to_resolve:
            return False
        return True

    def _check_overround(self, market: Market) -> Optional[ArbOpportunity]:
        """
        Over-round arbitrage: buy YES at ask + buy NO at ask.
        If YES_ask + NO_ask < 1.0 (minus fees), guaranteed profit.

        Total cost = yes_ask + no_ask
        Payout     = 1.00
        Gross profit = 1 - (yes_ask + no_ask)
        Net profit   = gross - 2 * fee_pct (one fee per leg)
        """
        yes_ask = market.yes_ask
        no_ask = market.no_ask
        total_cost = yes_ask + no_ask

        if total_cost >= self.cfg.overround_threshold:
            return None

        gross_pct = (1.0 - total_cost) / total_cost
        net_pct = gross_pct - 2 * self.cfg.fee_pct

        if net_pct < self.cfg.min_profit_pct:
            return None

        # Size: limited by liquidity and config cap
        max_size = min(self.cfg.max_position_usd, market.liquidity_usd * 0.05)

        logger.info(
            f"Overround arb found: {market.question!r} | "
            f"YES={yes_ask:.3f} NO={no_ask:.3f} sum={total_cost:.3f} "
            f"net={net_pct:.2%}"
        )

        return ArbOpportunity(
            market_id=market.market_id,
            question=market.question,
            arb_type="overround",
            leg_yes=yes_ask,
            leg_no=no_ask,
            gross_profit_pct=gross_pct,
            net_profit_pct=net_pct,
            max_size_usd=max_size,
            resolves_at=market.resolves_at,
            detected_at=datetime.now(timezone.utc),
        )
