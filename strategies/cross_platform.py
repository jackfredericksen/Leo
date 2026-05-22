"""
Cross-platform price signal — detect when Polymarket is mis-priced relative
to Kalshi (used as a read-only oracle) for the same event.

Example:
  - Polymarket: "Will BTC be above $100k by Dec 31?" YES @ 0.42
  - Kalshi:      same event                           YES @ 0.51
  → Polymarket YES is cheap relative to Kalshi consensus.
  → Signal: buy YES on Polymarket.

Kalshi is used purely as a read-only price oracle (no auth required for
public market data). All execution happens on Polymarket.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from api_clients.polymarket_client import Market

logger = logging.getLogger(__name__)

KALSHI_PUBLIC_BASE = "https://api.elections.kalshi.com/trade-api/v2"


# ---------------------------------------------------------------------------
# Kalshi read-only signal client (no authentication required)
# ---------------------------------------------------------------------------

class KalshiSignalClient:
    """
    Minimal read-only Kalshi client for fetching public market prices.
    Used as a signal source for the Polymarket cross-platform strategy.
    No API key or private key needed — Kalshi market data is public.
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        )
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    async def get_external_markets(
        self, max_pages: int = 5
    ) -> list["ExternalMarket"]:
        """
        Fetch active Kalshi binary markets and return them as ExternalMarket objects.
        Paginates up to max_pages × 200 markets.
        """
        results: list[ExternalMarket] = []
        cursor = None

        for _ in range(max_pages):
            try:
                params: dict = {
                    "limit": 200,
                    "with_nested_markets": "true",
                }
                if cursor:
                    params["cursor"] = cursor

                async with self._session.get(
                    f"{KALSHI_PUBLIC_BASE}/events", params=params
                ) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()

                for event in data.get("events", []):
                    for m in (event.get("markets") or []):
                        ext = self._parse_external(m)
                        if ext:
                            results.append(ext)

                cursor = data.get("cursor")
                if not cursor:
                    break
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Kalshi signal fetch: {e}")
                break

        logger.info(f"KalshiSignal: loaded {len(results)} markets")
        return results

    @staticmethod
    def _parse_external(m: dict) -> Optional["ExternalMarket"]:
        try:
            def _p(new_key: str, old_key: str) -> float:
                v = m.get(new_key) or m.get(old_key) or 0
                return float(v) if v else 0.0

            yes_bid = _p("yes_bid_dollars", "yes_bid")
            no_bid  = _p("no_bid_dollars",  "no_bid")
            yes_ask = float(m.get("yes_ask_dollars") or m.get("yes_ask") or 0) or round(1.0 - no_bid, 4)
            no_ask  = float(m.get("no_ask_dollars")  or m.get("no_ask")  or 0) or round(1.0 - yes_bid, 4)

            if yes_bid <= 0 and yes_ask <= 0:
                return None

            end_raw = m.get("close_time") or m.get("latest_expiration_time", "")
            resolves_at = None
            if end_raw:
                try:
                    if isinstance(end_raw, (int, float)):
                        from datetime import timezone
                        resolves_at = datetime.fromtimestamp(end_raw, tz=timezone.utc)
                    else:
                        resolves_at = datetime.fromisoformat(
                            str(end_raw).replace("Z", "+00:00")
                        )
                except Exception:
                    pass

            question = str(m.get("title") or m.get("yes_sub_title") or m.get("ticker", ""))
            if not question:
                return None

            liq = float(m.get("liquidity_dollars") or m.get("liquidity") or 0)

            return ExternalMarket(
                platform="kalshi",
                market_id=str(m.get("ticker", "")),
                question=question,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                liquidity_usd=liq,
                resolves_at=resolves_at,
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CrossPlatformConfig:
    min_profit_pct: float = 0.03
    min_liquidity_usd: float = 500.0
    max_position_usd: float = 200.0
    fee_pct_polymarket: float = 0.01   # ~1% conservative
    fee_pct_kalshi: float = 0.02       # Kalshi oracle side (no fee — signal only)


@dataclass
class ExternalMarket:
    """Normalised view of a market on an external platform (signal source)."""
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
    buy_platform: str       # "polymarket" (always — this is where we execute)
    sell_platform: str      # "kalshi" (signal source)
    polymarket_market_id: str
    external_market_id: str
    question: str
    buy_price: float        # ask price on Polymarket
    sell_price: float       # bid price on Kalshi (the "fair" reference)
    gross_profit_pct: float
    net_profit_pct: float
    max_size_usd: float
    detected_at: datetime


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class CrossPlatformDetector:
    """
    Compares Polymarket prices against Kalshi prices (read-only signal).
    When Polymarket is cheaper than Kalshi by more than min_profit_pct + fees,
    flag the Polymarket market as mispriced and signal a buy.
    """

    def __init__(self, cfg: CrossPlatformConfig):
        self.cfg = cfg
        self._external: dict[str, ExternalMarket] = {}

    def load_external_markets(self, markets: list[ExternalMarket]):
        """Update the Kalshi signal cache. Call periodically."""
        self._external = {_slug(m.question): m for m in markets}
        logger.info(
            f"CrossPlatform: loaded {len(self._external)} Kalshi signal markets"
        )

    def scan(
        self, polymarket_markets: list[Market]
    ) -> list[CrossPlatformOpportunity]:
        opps = []
        for pm in polymarket_markets:
            if pm.status != "open":
                continue
            slug = _slug(pm.question)
            ext = self._external.get(slug) or self._fuzzy_match(pm.question)
            if not ext:
                continue
            opp = self._check_cross(pm, ext)
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
        self, pm: Market, ext: ExternalMarket
    ) -> Optional[CrossPlatformOpportunity]:
        total_fee = self.cfg.fee_pct_polymarket + self.cfg.fee_pct_kalshi
        now = datetime.now(timezone.utc)

        # Direction A: Polymarket YES is cheap vs Kalshi YES
        if ext.yes_bid > pm.yes_ask:
            gross_pct = ext.yes_bid - pm.yes_ask
            net_pct = gross_pct - total_fee
            if net_pct >= self.cfg.min_profit_pct:
                liq = min(pm.liquidity_usd, ext.liquidity_usd)
                if liq >= self.cfg.min_liquidity_usd:
                    return CrossPlatformOpportunity(
                        buy_platform="polymarket",
                        sell_platform="kalshi",
                        polymarket_market_id=pm.market_id,
                        external_market_id=ext.market_id,
                        question=pm.question,
                        buy_price=pm.yes_ask,
                        sell_price=ext.yes_bid,
                        gross_profit_pct=gross_pct,
                        net_profit_pct=net_pct,
                        max_size_usd=min(self.cfg.max_position_usd, liq * 0.05),
                        detected_at=now,
                    )

        # Direction B: Polymarket NO is cheap vs Kalshi NO
        if ext.no_bid > pm.no_ask:
            gross_pct = ext.no_bid - pm.no_ask
            net_pct = gross_pct - total_fee
            if net_pct >= self.cfg.min_profit_pct:
                liq = min(pm.liquidity_usd, ext.liquidity_usd)
                if liq >= self.cfg.min_liquidity_usd:
                    return CrossPlatformOpportunity(
                        buy_platform="polymarket",
                        sell_platform="kalshi",
                        polymarket_market_id=pm.market_id,
                        external_market_id=ext.market_id,
                        question=pm.question,
                        buy_price=pm.no_ask,
                        sell_price=ext.no_bid,
                        gross_profit_pct=gross_pct,
                        net_profit_pct=net_pct,
                        max_size_usd=min(self.cfg.max_position_usd, liq * 0.05),
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
