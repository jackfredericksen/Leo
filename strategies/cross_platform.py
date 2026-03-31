"""
Cross-platform arbitrage — exploit price differences for the same event
across Coinbase Predictions and other prediction markets (Kalshi, Manifold, etc.).

Example:
  - Coinbase: "BTC above $100k by Dec 31?" YES @ 0.42
  - Kalshi:   same event                  YES @ 0.51
  → Buy YES on Coinbase, sell YES (buy NO) on Kalshi
  → Locked-in profit = 0.51 - 0.42 = $0.09 per $1 of exposure (minus fees)

Requirements:
  - Accounts on each platform
  - Ability to match markets across platforms (by slug/keyword/event)
  - Each platform client must expose the same Market interface

Currently supported external platforms:
  - Kalshi (via KalshiClient)
  - Manifold Markets (via ManifoldClient) — play money only, useful for signal
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from api_clients.kalshi_client import Market

logger = logging.getLogger(__name__)


@dataclass
class CrossPlatformConfig:
    min_profit_pct: float = 0.03      # Higher bar due to withdrawal delays
    min_liquidity_usd: float = 500.0
    max_position_usd: float = 200.0
    fee_pct_coinbase: float = 0.02
    fee_pct_external: float = 0.02    # Adjust per platform


@dataclass
class ExternalMarket:
    """Normalised view of a market on any external platform."""
    platform: str          # "kalshi" | "manifold" | etc.
    market_id: str
    question: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    liquidity_usd: float
    resolves_at: Optional[datetime]


@dataclass
class CrossPlatformOpportunity:
    buy_platform: str
    sell_platform: str
    coinbase_market_id: str
    external_market_id: str
    question: str
    buy_price: float       # YES ask on cheaper platform
    sell_price: float      # YES bid on more expensive platform
    gross_profit_pct: float
    net_profit_pct: float
    max_size_usd: float
    detected_at: datetime


class CrossPlatformDetector:
    def __init__(self, cfg: CrossPlatformConfig):
        self.cfg = cfg
        # Map of normalised question slug → ExternalMarket
        # Populated by calling load_external_markets()
        self._external: dict[str, ExternalMarket] = {}

    def load_external_markets(self, markets: list[ExternalMarket]):
        """Update the external market cache. Call periodically."""
        self._external = {self._slug(m.question): m for m in markets}

    def scan(self, coinbase_markets: list[Market]) -> list[CrossPlatformOpportunity]:
        opps = []
        for cb_market in coinbase_markets:
            if cb_market.status != "open":
                continue
            slug = self._slug(cb_market.question)
            ext = self._external.get(slug)
            if not ext:
                continue
            opp = self._check_cross(cb_market, ext)
            if opp:
                opps.append(opp)
        return sorted(opps, key=lambda o: o.net_profit_pct, reverse=True)

    def _check_cross(
        self, cb: Market, ext: ExternalMarket
    ) -> Optional[CrossPlatformOpportunity]:
        """
        Check both directions:
          A) Buy YES on Coinbase, sell YES (= buy NO) on external
          B) Buy YES on external, sell YES on Coinbase
        """
        total_fee = self.cfg.fee_pct_coinbase + self.cfg.fee_pct_external

        # Direction A: cheap YES on Coinbase, expensive YES on external
        if ext.yes_bid > cb.yes_ask:
            gross = ext.yes_bid - cb.yes_ask
            gross_pct = gross / cb.yes_ask
            net_pct = gross_pct - total_fee
            if net_pct >= self.cfg.min_profit_pct:
                liq = min(cb.liquidity_usd, ext.liquidity_usd)
                if liq >= self.cfg.min_liquidity_usd:
                    return CrossPlatformOpportunity(
                        buy_platform="coinbase",
                        sell_platform=ext.platform,
                        coinbase_market_id=cb.market_id,
                        external_market_id=ext.market_id,
                        question=cb.question,
                        buy_price=cb.yes_ask,
                        sell_price=ext.yes_bid,
                        gross_profit_pct=gross_pct,
                        net_profit_pct=net_pct,
                        max_size_usd=min(self.cfg.max_position_usd, liq * 0.05),
                        detected_at=datetime.now(timezone.utc),
                    )

        # Direction B: cheap YES on external, expensive YES on Coinbase
        if cb.yes_bid > ext.yes_ask:
            gross = cb.yes_bid - ext.yes_ask
            gross_pct = gross / ext.yes_ask
            net_pct = gross_pct - total_fee
            if net_pct >= self.cfg.min_profit_pct:
                liq = min(cb.liquidity_usd, ext.liquidity_usd)
                if liq >= self.cfg.min_liquidity_usd:
                    return CrossPlatformOpportunity(
                        buy_platform=ext.platform,
                        sell_platform="coinbase",
                        coinbase_market_id=cb.market_id,
                        external_market_id=ext.market_id,
                        question=cb.question,
                        buy_price=ext.yes_ask,
                        sell_price=cb.yes_bid,
                        gross_profit_pct=gross_pct,
                        net_profit_pct=net_pct,
                        max_size_usd=min(self.cfg.max_position_usd, liq * 0.05),
                        detected_at=datetime.now(timezone.utc),
                    )

        return None

    @staticmethod
    def _slug(question: str) -> str:
        """Normalise a question string for fuzzy matching across platforms."""
        import re
        return re.sub(r"[^a-z0-9]", "", question.lower())
