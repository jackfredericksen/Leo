"""
Cross-platform price signal — detect when Kalshi is mis-priced relative
to Polymarket for the same event.

Example:
  - Kalshi:     "Will BTC be above $100k by Dec 31?" YES @ 0.42
  - Polymarket: same event                           YES @ 0.51
  → Kalshi YES is cheap relative to Polymarket consensus.
  → Signal: buy YES on Kalshi.

All trading executes on Kalshi only. Polymarket prices are used
purely as a directional signal to identify mispricing. Set
CROSS_ARB_SIGNAL_ONLY=false only if you also have Polymarket
execution wired up (requires Polygon USDC wallet setup).
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from api_clients.kalshi_client import Market

logger = logging.getLogger(__name__)


@dataclass
class CrossPlatformConfig:
    min_profit_pct: float = 0.03
    min_liquidity_usd: float = 500.0
    max_position_usd: float = 200.0
    fee_pct_kalshi: float = 0.02
    fee_pct_polymarket: float = 0.01
    # If True, only use Polymarket as a signal; don't attempt Poly execution
    signal_only: bool = True


@dataclass
class ExternalMarket:
    """Normalised view of a market on any external platform."""
    platform: str
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
    buy_platform: str      # where to buy (usually "kalshi")
    sell_platform: str     # where the price is higher (signal source)
    kalshi_market_id: str
    external_market_id: str
    question: str
    buy_price: float
    sell_price: float
    gross_profit_pct: float
    net_profit_pct: float
    max_size_usd: float
    signal_only: bool
    detected_at: datetime


class CrossPlatformDetector:
    def __init__(self, cfg: CrossPlatformConfig):
        self.cfg = cfg
        self._external: dict[str, ExternalMarket] = {}

    def load_external_markets(self, markets: list[ExternalMarket]):
        """Update the external market cache. Call periodically."""
        self._external = {_slug(m.question): m for m in markets}
        logger.info(
            f"CrossPlatform: loaded {len(self._external)} "
            f"external markets"
        )

    def scan(
        self, kalshi_markets: list[Market]
    ) -> list[CrossPlatformOpportunity]:
        opps = []
        for km in kalshi_markets:
            if km.status != "open":
                continue
            slug = _slug(km.question)
            ext = self._external.get(slug) or self._fuzzy_match(km.question)
            if not ext:
                continue
            opp = self._check_cross(km, ext)
            if opp:
                opps.append(opp)
        return sorted(opps, key=lambda o: o.net_profit_pct, reverse=True)

    def _fuzzy_match(
        self, question: str, min_score: float = 0.30
    ) -> Optional[ExternalMarket]:
        """Jaccard similarity on 3+ char word sets."""
        q_words = set(_words(question))
        best_score = min_score
        best = None
        for ext in self._external.values():
            ext_words = set(_words(ext.question))
            if not q_words or not ext_words:
                continue
            score = len(q_words & ext_words) / len(q_words | ext_words)
            if score > best_score:
                best_score = score
                best = ext
        return best

    def _check_cross(
        self, km: Market, ext: ExternalMarket
    ) -> Optional[CrossPlatformOpportunity]:
        total_fee = self.cfg.fee_pct_kalshi + self.cfg.fee_pct_polymarket
        now = datetime.now(timezone.utc)

        # Direction A: Kalshi YES cheap vs external YES
        if ext.yes_bid > km.yes_ask:
            gross_pct = ext.yes_bid - km.yes_ask
            net_pct = gross_pct - total_fee
            if net_pct >= self.cfg.min_profit_pct:
                liq = min(km.liquidity_usd, ext.liquidity_usd)
                if liq >= self.cfg.min_liquidity_usd:
                    return CrossPlatformOpportunity(
                        buy_platform="kalshi",
                        sell_platform=ext.platform,
                        kalshi_market_id=km.market_id,
                        external_market_id=ext.market_id,
                        question=km.question,
                        buy_price=km.yes_ask,
                        sell_price=ext.yes_bid,
                        gross_profit_pct=gross_pct,
                        net_profit_pct=net_pct,
                        max_size_usd=min(
                            self.cfg.max_position_usd, liq * 0.05
                        ),
                        signal_only=self.cfg.signal_only,
                        detected_at=now,
                    )

        # Direction B: External YES cheap vs Kalshi YES
        if km.yes_bid > ext.yes_ask:
            gross_pct = km.yes_bid - ext.yes_ask
            net_pct = gross_pct - total_fee
            if net_pct >= self.cfg.min_profit_pct:
                liq = min(km.liquidity_usd, ext.liquidity_usd)
                if liq >= self.cfg.min_liquidity_usd:
                    return CrossPlatformOpportunity(
                        buy_platform=ext.platform,
                        sell_platform="kalshi",
                        kalshi_market_id=km.market_id,
                        external_market_id=ext.market_id,
                        question=km.question,
                        buy_price=ext.yes_ask,
                        sell_price=km.yes_bid,
                        gross_profit_pct=gross_pct,
                        net_profit_pct=net_pct,
                        max_size_usd=min(
                            self.cfg.max_position_usd, liq * 0.05
                        ),
                        signal_only=self.cfg.signal_only,
                        detected_at=now,
                    )

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(question: str) -> str:
    return re.sub(r"[^a-z0-9]", "", question.lower())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())
