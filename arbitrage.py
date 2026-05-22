"""
Arbitrage detection engine for Polymarket.

Polymarket binary market identity:
  YES and NO tokens are complementary — one always resolves to $1 USDC.

Over-round arb exists when YES_ask + NO_ask < 1.00 (minus fees).
You buy YES and NO simultaneously for less than $1.00 total, guaranteed
to collect $1.00 at resolution regardless of outcome.

Equivalently: YES_bid + NO_bid > 1.00 means the combined implied probability
exceeds 100%, which is the same condition viewed from the bid side.

Prices are in USDC (0.00–1.00). Each share pays $1.00 USDC at resolution.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from api_clients.polymarket_client import Market
from config import ArbitrageConfig

logger = logging.getLogger(__name__)


@dataclass
class ArbOpportunity:
    market_id: str
    question: str
    arb_type: str               # "overround"
    yes_bid: float              # YES bid price (for reference)
    no_bid: float               # NO bid price (for reference)
    yes_ask: float              # price to BUY YES at
    no_ask: float               # price to BUY NO at
    gross_profit_pct: float
    net_profit_pct: float
    max_size_usd: float
    close_time: datetime
    detected_at: datetime
    yes_token_id: str = ""      # needed by trader to place YES leg
    no_token_id: str = ""       # needed by trader to place NO leg


class ArbitrageDetector:
    def __init__(self, cfg: ArbitrageConfig):
        self.cfg = cfg

    async def scan_live(
        self,
        markets: list[Market],
        client,
        max_candidates: int = 20,
    ) -> list[ArbOpportunity]:
        """
        Like scan() but fetches real CLOB bid/ask for each market in parallel.
        Uses actual order-book best ask prices instead of Gamma API mid-prices,
        so overround conditions are detectable even when bid==ask==mid from Gamma.
        """
        eligible = [m for m in markets if self._is_eligible(m)][:max_candidates]
        if not eligible:
            return []

        # Fetch YES and NO orderbooks in parallel (2 calls per market)
        tasks = []
        for m in eligible:
            tasks.append(client.get_orderbook(m.yes_token_id))
            tasks.append(client.get_orderbook(m.no_token_id))

        books = await asyncio.gather(*tasks, return_exceptions=True)

        opps = []
        for i, market in enumerate(eligible):
            yes_book = books[i * 2]
            no_book  = books[i * 2 + 1]
            if isinstance(yes_book, Exception) or isinstance(no_book, Exception):
                continue
            if not yes_book.asks or not no_book.asks:
                continue

            yes_ask = float(yes_book.asks[0][0])
            no_ask  = float(no_book.asks[0][0])
            yes_bid = float(yes_book.bids[0][0]) if yes_book.bids else 0.0
            no_bid  = float(no_book.bids[0][0])  if no_book.bids  else 0.0

            total_asks = yes_ask + no_ask
            if total_asks >= (1.0 / self.cfg.overround_threshold):
                continue

            gross_pct = 1.0 - total_asks
            total_fee_pct = 0.01 * yes_ask + 0.01 * no_ask
            net_pct = gross_pct - total_fee_pct
            if net_pct < self.cfg.min_profit_pct:
                continue

            max_size = min(self.cfg.max_position_usd, market.liquidity_usd * 0.05)
            logger.info(
                f"Overround arb (live): {market.market_id!r} | "
                f"YES_ask={yes_ask:.3f} NO_ask={no_ask:.3f} net={net_pct:.2%}"
            )
            opps.append(ArbOpportunity(
                market_id=market.market_id,
                question=market.question,
                arb_type="overround",
                yes_bid=yes_bid,
                no_bid=no_bid,
                yes_ask=yes_ask,
                no_ask=no_ask,
                gross_profit_pct=gross_pct,
                net_profit_pct=net_pct,
                max_size_usd=max_size,
                close_time=market.close_time,
                detected_at=datetime.now(timezone.utc),
                yes_token_id=market.yes_token_id,
                no_token_id=market.no_token_id,
            ))

        return sorted(opps, key=lambda o: o.net_profit_pct, reverse=True)

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
        Polymarket overround arb: YES_ask + NO_ask < 1.00

        Strategy: simultaneously place limit BUY orders on YES and NO.
        When both fill, you hold both tokens and collect $1.00 at resolution —
        more than the combined purchase cost.

        Gross profit = 1.00 − (YES_ask + NO_ask)
        Fee estimate: ~1% of each leg (conservative)
        """
        yes_ask = market.yes_ask
        no_ask  = market.no_ask
        total_asks = yes_ask + no_ask

        # Arb exists when total cost < $1.00 guaranteed payout
        # Equivalently: YES_bid + NO_bid > 1 (since YES_ask ≈ 1 - NO_bid)
        if total_asks >= (1.0 / self.cfg.overround_threshold):
            return None

        gross_pct = 1.0 - total_asks

        # Polymarket fee: ~1% per leg (taker fee ≈ 2 bps per trade but we
        # conservatively model 1% total to account for spread slippage)
        total_fee_pct = 0.01 * yes_ask + 0.01 * no_ask

        net_pct = gross_pct - total_fee_pct

        if net_pct < self.cfg.min_profit_pct:
            return None

        max_size = min(self.cfg.max_position_usd, market.liquidity_usd * 0.05)

        logger.info(
            f"Overround arb: {market.market_id!r} | "
            f"YES_ask={yes_ask:.3f} NO_ask={no_ask:.3f} "
            f"sum={total_asks:.4f} net={net_pct:.2%}"
        )

        return ArbOpportunity(
            market_id=market.market_id,
            question=market.question,
            arb_type="overround",
            yes_bid=market.yes_bid,
            no_bid=market.no_bid,
            yes_ask=yes_ask,
            no_ask=no_ask,
            gross_profit_pct=gross_pct,
            net_profit_pct=net_pct,
            max_size_usd=max_size,
            close_time=market.close_time,
            detected_at=datetime.now(timezone.utc),
            yes_token_id=market.yes_token_id,
            no_token_id=market.no_token_id,
        )
