"""
Polymarket Gamma API client — used as a price signal source only.

All trading executes on Kalshi. Polymarket prices are used purely
to detect when Kalshi is mis-priced relative to the broader market.

Polymarket is a decentralized prediction market on Polygon (USDC).
Their Gamma API is public and requires no authentication.

Docs: https://docs.polymarket.com
Gamma API base: https://gamma-api.polymarket.com
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"


@dataclass
class PolyMarket:
    """Raw Polymarket market, normalised from the Gamma API response."""
    market_id: str
    question: str
    yes_price: float        # probability of YES (0-1)
    no_price: float         # probability of NO (0-1)
    volume_usd: float
    liquidity_usd: float
    end_date: Optional[datetime]
    active: bool


class PolymarketClient:
    """
    Async client for the Polymarket Gamma API.

    Usage:
        async with PolymarketClient() as poly:
            markets = await poly.get_all_active_markets(max_pages=3)
            ext = poly.to_external_markets(markets)
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    async def get_all_active_markets(
        self, max_pages: int = 3
    ) -> list[PolyMarket]:
        """
        Fetch active binary markets from Polymarket Gamma API.
        Paginates up to max_pages × 100 markets.
        """
        all_markets: list[PolyMarket] = []
        offset = 0
        limit = 100

        for _ in range(max_pages):
            try:
                params = {
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "offset": offset,
                }
                async with self._session.get(
                    f"{GAMMA_BASE}/markets", params=params
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"Polymarket API status {resp.status}"
                        )
                        break
                    data = await resp.json()

                batch = data if isinstance(data, list) else data.get("markets", [])
                if not batch:
                    break

                for raw in batch:
                    m = self._parse_market(raw)
                    if m:
                        all_markets.append(m)

                if len(batch) < limit:
                    break
                offset += limit
                await asyncio.sleep(0.3)

            except Exception as e:
                logger.warning(f"Polymarket fetch error: {e}")
                break

        logger.info(
            f"Polymarket: loaded {len(all_markets)} active markets"
        )
        return all_markets

    def _parse_market(self, raw: dict) -> Optional[PolyMarket]:
        try:
            # Polymarket binary markets have exactly 2 outcomes
            outcomes = raw.get("outcomes", [])
            prices_raw = raw.get("outcomePrices", [])

            if isinstance(outcomes, str):
                import json
                try:
                    outcomes = json.loads(outcomes)
                except Exception:
                    outcomes = []
            if isinstance(prices_raw, str):
                import json
                try:
                    prices_raw = json.loads(prices_raw)
                except Exception:
                    prices_raw = []

            if len(outcomes) != 2 or len(prices_raw) != 2:
                return None

            # Find which index is YES
            yes_idx = 0
            for i, o in enumerate(outcomes):
                if str(o).lower() in ("yes", "true"):
                    yes_idx = i
                    break

            no_idx = 1 - yes_idx
            yes_price = float(prices_raw[yes_idx])
            no_price = float(prices_raw[no_idx])

            # Parse end date
            end_date = None
            end_raw = raw.get("endDate") or raw.get("end_date_iso")
            if end_raw:
                try:
                    end_date = datetime.fromisoformat(
                        str(end_raw).replace("Z", "+00:00")
                    )
                except Exception:
                    pass

            return PolyMarket(
                market_id=str(raw.get("id", "")),
                question=str(
                    raw.get("question")
                    or raw.get("title")
                    or ""
                ),
                yes_price=yes_price,
                no_price=no_price,
                volume_usd=float(raw.get("volume", 0) or 0),
                liquidity_usd=float(raw.get("liquidity", 0) or 0),
                end_date=end_date,
                active=bool(raw.get("active", True)),
            )
        except Exception:
            return None

    def to_external_markets(
        self, poly_markets: list[PolyMarket]
    ) -> list:
        """
        Convert PolyMarket objects to the ExternalMarket format expected
        by CrossPlatformDetector.
        """
        from strategies.cross_platform import ExternalMarket
        result = []
        for m in poly_markets:
            if not m.active or not m.question:
                continue
            result.append(ExternalMarket(
                platform="polymarket",
                market_id=m.market_id,
                question=m.question,
                yes_bid=m.yes_price,
                yes_ask=m.yes_price,   # no spread info from Gamma API
                no_bid=m.no_price,
                no_ask=m.no_price,
                liquidity_usd=m.liquidity_usd,
                resolves_at=m.end_date,
            ))
        return result
