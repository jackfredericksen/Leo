"""
Market making strategy — passively earn the bid-ask spread by posting
limit orders on both sides of a prediction market.

Unlike arbitrage (which needs a mispriced market to exist), market making
generates income from other traders crossing the spread. Risk comes from
adverse selection — if the market moves strongly one way, one side fills
and you hold a directional position.

Safeguards:
  - Inventory skew: as one side fills more, widen the quote on that side
    to reduce further exposure in that direction.
  - Max inventory limits per market.
  - Cancel and requote on significant mid-price moves.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from api_clients.kalshi_client import Market

logger = logging.getLogger(__name__)


@dataclass
class MMConfig:
    # Half-spread: how far from mid to post each side (e.g. 0.01 = 1 cent)
    half_spread: float = 0.015
    # Order size per side in USD
    order_size_usd: float = 25.0
    # Max net inventory per market before pausing new quotes (USD)
    max_inventory_usd: float = 150.0
    # Cancel and requote if mid moves more than this since last quote
    requote_threshold: float = 0.02
    # Inventory skew factor: for every $1 of net long, widen YES ask by this
    skew_per_dollar: float = 0.001
    # Minimum spread in market before we bother quoting (not worth it if too thin)
    min_existing_spread: float = 0.005


@dataclass
class MMQuote:
    market_id: str
    yes_bid: float      # our limit buy on YES
    yes_ask: float      # our limit sell on YES (= limit buy on NO)
    size_usd: float
    mid_at_quote: float
    quoted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MarketMaker:
    def __init__(self, cfg: MMConfig):
        self.cfg = cfg
        # Track net inventory per market (positive = net long YES)
        self._inventory: dict[str, float] = {}
        self._active_quotes: dict[str, MMQuote] = {}

    def should_quote(self, market: Market) -> bool:
        if market.status != "open":
            return False
        # Only quote if the existing spread is wide enough to be worth stepping into
        existing_spread = market.yes_ask - market.yes_bid
        if existing_spread < self.cfg.min_existing_spread:
            return False
        # Don't quote if we're at inventory limit
        net_inv = self._inventory.get(market.market_id, 0.0)
        if abs(net_inv) >= self.cfg.max_inventory_usd:
            logger.info(f"MM: inventory limit hit for {market.market_id}")
            return False
        return True

    def should_requote(self, market: Market) -> bool:
        prev = self._active_quotes.get(market.market_id)
        if not prev:
            return False
        mid_now = (market.yes_bid + market.yes_ask) / 2
        return abs(mid_now - prev.mid_at_quote) >= self.cfg.requote_threshold

    def build_quote(self, market: Market) -> Optional[MMQuote]:
        if not self.should_quote(market):
            return None

        mid = (market.yes_bid + market.yes_ask) / 2
        net_inv = self._inventory.get(market.market_id, 0.0)

        # Inventory skew: if we're long YES, push ask up and bid down
        skew = net_inv * self.cfg.skew_per_dollar
        our_bid = max(0.01, mid - self.cfg.half_spread - skew)
        our_ask = min(0.99, mid + self.cfg.half_spread - skew)

        if our_ask <= our_bid:
            return None

        quote = MMQuote(
            market_id=market.market_id,
            yes_bid=round(our_bid, 3),
            yes_ask=round(our_ask, 3),
            size_usd=self.cfg.order_size_usd,
            mid_at_quote=mid,
        )
        self._active_quotes[market.market_id] = quote
        return quote

    def record_fill(self, market_id: str, side: str, size_usd: float):
        """Call this when an order fills to update inventory tracking."""
        delta = size_usd if side == "YES" else -size_usd
        self._inventory[market_id] = self._inventory.get(market_id, 0.0) + delta
        logger.info(
            f"MM fill: {market_id} {side} ${size_usd:.2f} | "
            f"net_inv=${self._inventory[market_id]:.2f}"
        )

    def get_inventory(self, market_id: str) -> float:
        return self._inventory.get(market_id, 0.0)
