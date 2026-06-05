"""
Pyth Network Hermes client — real-time BTC/USD oracle prices.

Polymarket resolves BTC prediction markets against the Pyth BTC/USD feed.
Using this source eliminates the Coinbase/oracle basis risk (~0.02–0.20%)
that would otherwise corrupt window-delta calculations near the threshold.

Feed: BTC/USD
  ID: e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43
API: https://hermes.pyth.network/v2/...

Two modes:
  poll()   — one-shot REST fetch, used by the 30s scan loop as a fallback
  stream() — long-lived SSE task (~400ms ticks), used by the 2s fast loop
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

HERMES_BASE = "https://hermes.pyth.network"
BTC_USD_ID = "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"

# Historical cache prunes to this many minute-buckets (~8h at 1-min resolution)
_HIST_CACHE_MAX = 500


def _decode(item: dict) -> Optional[float]:
    """Parse a Pyth parsed price item → float USD."""
    try:
        p = item["price"]
        return int(p["price"]) * (10 ** int(p["expo"]))
    except (KeyError, ValueError, TypeError):
        return None


class PythClient:
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._price: Optional[float] = None
        self._publish_time: Optional[int] = None
        # Historical prices keyed by minute bucket (unix_ts // 60 * 60)
        self._hist: dict[int, float] = {}

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    # ── Accessors ─────────────────────────────────────────────────────────

    def get_price(self) -> Optional[float]:
        """Return the most recently received BTC/USD price (stream or poll)."""
        return self._price

    # ── One-shot REST poll ─────────────────────────────────────────────────

    async def poll(self) -> Optional[float]:
        """Fetch current BTC/USD price via REST. Fallback when stream is down."""
        if not self._session:
            return None
        try:
            url = (
                f"{HERMES_BASE}/v2/updates/price/latest"
                f"?ids[]={BTC_USD_ID}&parsed=true"
            )
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Pyth poll: HTTP {resp.status}")
                    return None
                data = await resp.json()
                items = data.get("parsed", [])
                if not items:
                    return None
                price = _decode(items[0])
                if price and price > 0:
                    self._price = price
                    self._publish_time = items[0]["price"].get("publish_time")
                return price
        except Exception as e:
            logger.warning(f"Pyth poll: {e}")
            return None

    # ── Historical price lookup ────────────────────────────────────────────

    async def get_price_at(self, target_dt: datetime) -> Optional[float]:
        """
        Return the Pyth BTC/USD price closest to target_dt.

        Rounds to the nearest minute and caches results so repeated lookups
        for the same window-open time don't make redundant API calls.
        """
        if not self._session:
            return None
        bucket = int(target_dt.timestamp()) // 60 * 60
        if bucket in self._hist:
            return self._hist[bucket]
        try:
            url = (
                f"{HERMES_BASE}/v2/updates/price/{bucket}"
                f"?ids[]={BTC_USD_ID}&parsed=true"
            )
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=8)
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        f"Pyth historical: HTTP {resp.status} "
                        f"for ts={bucket} ({target_dt.strftime('%H:%M:%S')} UTC)"
                    )
                    return None
                data = await resp.json()
                items = data.get("parsed", [])
                if not items:
                    return None
                price = _decode(items[0])
                if price and price > 0:
                    self._hist[bucket] = price
                    if len(self._hist) > _HIST_CACHE_MAX:
                        del self._hist[min(self._hist)]
                return price
        except Exception as e:
            logger.warning(f"Pyth get_price_at({bucket}): {e}")
            return None

    # ── SSE stream ────────────────────────────────────────────────────────

    async def stream(self):
        """
        Long-running SSE task. Reconnects automatically on disconnect.
        Spawn as an asyncio task; it runs until cancelled.

        Pyth Hermes streams price updates at ~400ms intervals.
        Updates self._price in place so fast_scan reads it without any
        additional I/O.
        """
        url = (
            f"{HERMES_BASE}/v2/updates/price/stream"
            f"?ids[]={BTC_USD_ID}&parsed=true&allow_unordered=true"
        )
        backoff = 1.0
        while True:
            try:
                if not self._session:
                    await asyncio.sleep(5)
                    continue
                logger.info("Pyth SSE: connecting…")
                async with self._session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=None, connect=10),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"Pyth SSE: HTTP {resp.status}, "
                            f"retrying in {backoff:.0f}s"
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                        continue

                    logger.info("Pyth SSE: stream live")
                    backoff = 1.0

                    async for raw in resp.content:
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            payload = json.loads(line[5:])
                            items = payload.get("parsed", [])
                            if items:
                                price = _decode(items[0])
                                if price and price > 0:
                                    self._price = price
                                    self._publish_time = items[0]["price"].get(
                                        "publish_time"
                                    )
                        except Exception:
                            pass  # malformed line — skip silently

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"Pyth SSE disconnected: {e}, "
                    f"retrying in {backoff:.0f}s"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
