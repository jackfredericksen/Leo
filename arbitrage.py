"""
Arbitrage detection engine for Kalshi / Coinbase Predictions.

Kalshi binary market identity:
  YES_ask = 1.00 - NO_bid
  NO_ask  = 1.00 - YES_bid

Over-round arb exists when YES_bid + NO_bid > 1.00 (plus fees).
That means you can sell YES and sell NO simultaneously (i.e. sell
both sides) and lock in a guaranteed profit regardless of outcome —
the opposite of the "buy both sides cheap" arb on traditional books.

Equivalently in terms of ask prices:
  YES_ask + NO_ask = (1 - NO_bid) + (1 - YES_bid)
                   = 2 - (YES_bid + NO_bid)
  If YES_bid + NO_bid > 1, then YES_ask + NO_ask < 1 — same arb.

Prices are in dollars (0.00–1.00). Contracts pay $1.00 at resolution.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from api_clients.kalshi_client import Market
from config import ArbitrageConfig

logger = logging.getLogger(__name__)


@dataclass
class ArbOpportunity:
    market_id: str
    question: str
    arb_type: str               # "overround"
    yes_bid: float              # price to sell YES at (= what market pays)
    no_bid: float               # price to sell NO at
    yes_ask: float              # derived: 1 - no_bid
    no_ask: float               # derived: 1 - yes_bid
    gross_profit_pct: float     # before fees
    net_profit_pct: float       # after fees
    max_size_usd: float
    close_time: datetime
    detected_at: datetime


class ArbitrageDetector:
    def __init__(self, cfg: ArbitrageConfig):
        self.cfg = cfg

    def scan(self, markets: list[Market]) -> list[ArbOpportunity]:
        opps = []
        for market in markets:
            if not self._is_eligible(market):
                continue
            opp = self._check_overround(market)
            if opp:
                opps.append(opp)
        return sorted(opps, key=lambda o: o.net_profit_pct, reverse=True)

    def _is_eligible(self, market: Market) -> bool:
        if market.status != "open":
            return False
        if market.liquidity_usd < self.cfg.min_liquidity_usd:
            return False
        now = datetime.now(timezone.utc)
        hours_left = (market.close_time - now).total_seconds() / 3600
        if hours_left < self.cfg.min_hours_to_resolve:
            return False
        if hours_left > self.cfg.max_hours_to_resolve:
            return False
        return True

    def _check_overround(self, market: Market) -> Optional[ArbOpportunity]:
        """
        Kalshi overround arb: YES_bid + NO_bid > 1.00

        Strategy: simultaneously post limit SELL YES at yes_bid price and
        SELL NO at no_bid price (or equivalently, buy the no/yes contract
        on the other side). When both fill, you collect more than $1.00
        for a guaranteed $1.00 payout — locking in the spread.

        Gross profit per $1 of notional = (yes_bid + no_bid) - 1.0
        Fee per contract: $0.07 × P × (1-P), charged on each leg.
        We estimate total fee as 2× the fee at midpoint.
        """
        yes_bid = market.yes_bid
        no_bid = market.no_bid
        total_bids = yes_bid + no_bid

        if total_bids <= self.cfg.overround_threshold:
            return None

        gross_pct = total_bids - 1.0

        # Kalshi fee: $0.07 × P × (1-P) per contract, ~2× for two legs
        fee_yes = 0.07 * yes_bid * (1 - yes_bid)
        fee_no = 0.07 * no_bid * (1 - no_bid)
        total_fee_pct = fee_yes + fee_no

        net_pct = gross_pct - total_fee_pct

        if net_pct < self.cfg.min_profit_pct:
            return None

        max_size = min(self.cfg.max_position_usd, market.liquidity_usd * 0.05)

        logger.info(
            f"Overround arb: {market.market_id!r} | "
            f"YES_bid={yes_bid:.3f} NO_bid={no_bid:.3f} "
            f"sum={total_bids:.4f} net={net_pct:.2%}"
        )

        return ArbOpportunity(
            market_id=market.market_id,
            question=market.question,
            arb_type="overround",
            yes_bid=yes_bid,
            no_bid=no_bid,
            yes_ask=round(1.0 - no_bid, 4),
            no_ask=round(1.0 - yes_bid, 4),
            gross_profit_pct=gross_pct,
            net_profit_pct=net_pct,
            max_size_usd=max_size,
            close_time=market.close_time,
            detected_at=datetime.now(timezone.utc),
        )
