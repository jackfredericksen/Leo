"""
Strategy: Automated Market Making with Maker Rebates.

Posts resting limit buy orders on both YES and NO sides of selected markets.
Earns the bid-ask spread passively when takers fill against resting orders.

Polymarket CLOB v2 (live April 28, 2026) redistributes 100% of taker fees to
makers daily in USDC, proportional to order size, mid proximity, and quoting
consistency. Orders must remain on-book >= 3.5 seconds to qualify for rebates.

The taker fee peaks at p=0.50 (~1.56%) and drops near 0 or 1 (~0.20%), so
the highest rebate-per-fill is captured on near-certain markets. This strategy
targets markets with real spread > min_spread (from live CLOB data, not Gamma),
so it requires get_orderbook() calls to see actual depth.

Two modes:
  Liquid  (volume >= min_volume_usd): tight quotes, high fill rate
  Thin    (OI < thin_oi_usd, wide spread): wide quotes, high per-fill profit

Order lifecycle:
  1. Select markets by checking real CLOB spread via get_orderbook()
  2. Post buy YES at (mid - half_spread) and buy NO at ((1-mid) - half_spread)
  3. Track order IDs and quote age
  4. Requote if mid drifts > requote_threshold (cancel + replace)
  5. Cancel all on shutdown

This strategy is NOT latency-sensitive — resting orders have no race condition.
A home Mac mini is fully adequate.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from api_clients.polymarket_client import Market, PolymarketClient

logger = logging.getLogger(__name__)


@dataclass
class MarketMakerConfig:
    enabled: bool = True
    quote_half_spread: float = 0.03      # 3¢ each side of mid for liquid markets
    thin_quote_half_spread: float = 0.05 # 5¢ each side for thin markets
    max_inventory_usd: float = 200.0     # max per-market USDC commitment (both legs)
    min_volume_usd: float = 5000.0       # liquid market threshold
    thin_oi_usd: float = 500.0           # thin market threshold (OI)
    min_spread_cents: float = 0.06       # only quote if real CLOB spread > this
    max_markets: int = 8                 # max simultaneous markets to quote
    requote_threshold: float = 0.025     # requote if mid moves > 2.5¢
    requote_interval_sec: int = 30
    min_order_usd: float = 10.0          # minimum per-leg order size
    min_onbook_sec: float = 3.5          # don't cancel before this (rebate qualification)
    refresh_interval_sec: int = 30


@dataclass
class _ActiveQuote:
    order_id: str
    token_id: str
    side: str           # "yes" or "no"
    price: float
    size_usd: float
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def age_sec(self) -> float:
        return (datetime.now(timezone.utc) - self.placed_at).total_seconds()


@dataclass
class _MarketQuotes:
    market_id: str
    question: str
    fair_value: float
    yes_quote: Optional[_ActiveQuote] = None
    no_quote: Optional[_ActiveQuote] = None
    yes_inventory_usd: float = 0.0   # cumulative fills on YES side
    no_inventory_usd: float = 0.0    # cumulative fills on NO side

    def stale(self, current_mid: float, threshold: float) -> bool:
        return abs(current_mid - self.fair_value) > threshold

    def is_quoted(self) -> bool:
        return self.yes_quote is not None or self.no_quote is not None

    def inventory_skew(self) -> float:
        """
        Signed inventory imbalance in [-1, +1].
        Positive = long YES → skew quotes to sell more YES (lower YES bid).
        Zero when balanced or no fills yet.
        """
        total = self.yes_inventory_usd + self.no_inventory_usd
        if total < 1.0:
            return 0.0
        return (self.yes_inventory_usd - self.no_inventory_usd) / total


class MarketMakerStrategy:
    """
    Stateful market maker: maintains resting orders across multiple markets,
    requoting when fair value drifts.

    Call run_once(markets) periodically from the main mm_loop.
    Call cancel_all() on shutdown.
    """

    def __init__(self, cfg: MarketMakerConfig, client: PolymarketClient):
        self.cfg = cfg
        self.client = client
        self._quotes: dict[str, _MarketQuotes] = {}

    async def run_once(self, markets: list[Market]) -> int:
        """
        One iteration of the MM loop. Returns count of actively quoted markets.
        """
        market_map = {m.market_id: m for m in markets}

        # Requote or remove stale positions
        requote_candidates: list[tuple[Market, _MarketQuotes]] = []
        for mid_id in list(self._quotes.keys()):
            mq = self._quotes[mid_id]
            market = market_map.get(mid_id)
            if not market or market.status != "open":
                await self._cancel_quotes(mq)
                del self._quotes[mid_id]
                continue

            book = await self.client.get_orderbook(market.yes_token_id)
            current_mid = _book_mid(book)
            if current_mid > 0 and mq.stale(current_mid, self.cfg.requote_threshold):
                logger.debug(
                    f"MM: requoting {market.question[:40]} "
                    f"(mid moved {abs(current_mid - mq.fair_value):.3f})"
                )
                await self._cancel_quotes(mq)
                # Keep inventory state when requoting the same market
                requote_candidates.append((market, mq))
                del self._quotes[mid_id]

        # Requote markets that need refreshing (preserves inventory state)
        for market, old_mq in requote_candidates:
            if len(self._quotes) < self.cfg.max_markets:
                await self._quote_market(market, existing_mq=old_mq)

        # Add new markets up to max
        if len(self._quotes) < self.cfg.max_markets:
            candidates = await self._find_candidates(markets)
            for market in candidates:
                if len(self._quotes) >= self.cfg.max_markets:
                    break
                if market.market_id in self._quotes:
                    continue
                await self._quote_market(market)

        return len(self._quotes)

    async def cancel_all(self):
        """Cancel all outstanding orders. Call on shutdown."""
        for mq in self._quotes.values():
            await self._cancel_quotes(mq)
        self._quotes.clear()

    def summary(self) -> list[dict]:
        return [
            {
                "market_id": mq.market_id,
                "question": mq.question,
                "fair_value": round(mq.fair_value, 3),
                "yes_bid": round(mq.yes_quote.price, 3) if mq.yes_quote else None,
                "no_bid": round(mq.no_quote.price, 3) if mq.no_quote else None,
            }
            for mq in self._quotes.values()
        ]

    @property
    def active_count(self) -> int:
        return len(self._quotes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _find_candidates(self, markets: list[Market]) -> list[Market]:
        """Fetch real CLOB spreads and return markets worth quoting."""
        candidates = [
            m for m in markets
            if m.status == "open"
            and m.market_id not in self._quotes
            and m.yes_token_id
        ]
        # Sort by potential profitability: liquid first, then thin
        candidates.sort(key=lambda m: m.volume_24h + m.open_interest, reverse=True)
        candidates = candidates[:self.cfg.max_markets * 3]  # over-fetch, filter below

        good = []
        for market in candidates:
            try:
                book = await self.client.get_orderbook(market.yes_token_id)
                if not book.bids or not book.asks:
                    continue
                spread = float(book.asks[0][0]) - float(book.bids[0][0])
                if spread < self.cfg.min_spread_cents:
                    continue
                is_liquid = market.volume >= self.cfg.min_volume_usd
                is_thin = market.open_interest < self.cfg.thin_oi_usd and spread >= 0.08
                if not (is_liquid or is_thin):
                    continue
                good.append(market)
                await asyncio.sleep(0.1)  # gentle on CLOB API
            except Exception as e:
                logger.debug(f"MM candidate check {market.market_id}: {e}")

        good.sort(key=lambda m: m.volume >= self.cfg.min_volume_usd, reverse=True)
        return good

    async def _quote_market(self, market: Market, existing_mq: Optional[_MarketQuotes] = None) -> bool:
        book = await self.client.get_orderbook(market.yes_token_id)
        if not book.bids or not book.asks:
            return False

        mid = _book_mid(book)
        if not (0.02 <= mid <= 0.98):
            return False

        is_thin = market.open_interest < self.cfg.thin_oi_usd
        half = self.cfg.thin_quote_half_spread if is_thin else self.cfg.quote_half_spread

        # Inventory skew: if long YES, lower YES bid to attract fewer YES fills
        # and widen NO bid to attract more NO fills (and vice-versa)
        skew = existing_mq.inventory_skew() if existing_mq else 0.0
        skew_adj = skew * 0.02   # ±2¢ max adjustment at full skew

        yes_bid = round(max(0.01, mid - half - skew_adj), 2)
        no_bid  = round(max(0.01, (1.0 - mid) - half + skew_adj), 2)

        per_leg = min(
            self.cfg.max_inventory_usd / 2,
            max(self.cfg.min_order_usd, market.liquidity_usd * 0.01),
        )
        if per_leg < self.cfg.min_order_usd:
            return False

        yes_shares = PolymarketClient.usdc_to_shares(per_leg, yes_bid)
        no_shares = PolymarketClient.usdc_to_shares(per_leg, no_bid)

        yes_oid = await _place_limit(self.client, market.yes_token_id, "buy", yes_bid, yes_shares)
        no_oid = await _place_limit(self.client, market.no_token_id, "buy", no_bid, no_shares)

        if not yes_oid and not no_oid:
            return False

        mq = _MarketQuotes(
            market_id=market.market_id,
            question=market.question,
            fair_value=mid,
            yes_inventory_usd=existing_mq.yes_inventory_usd if existing_mq else 0.0,
            no_inventory_usd=existing_mq.no_inventory_usd if existing_mq else 0.0,
        )
        if yes_oid:
            mq.yes_quote = _ActiveQuote(
                order_id=yes_oid, token_id=market.yes_token_id,
                side="yes", price=yes_bid, size_usd=per_leg,
            )
        if no_oid:
            mq.no_quote = _ActiveQuote(
                order_id=no_oid, token_id=market.no_token_id,
                side="no", price=no_bid, size_usd=per_leg,
            )
        self._quotes[market.market_id] = mq
        logger.info(
            f"MM: quoting {market.question[:40]} | "
            f"YES@{yes_bid:.2f} NO@{no_bid:.2f} "
            f"mid={mid:.2f} {'[thin]' if is_thin else '[liquid]'}"
        )
        return True

    async def _cancel_quotes(self, mq: _MarketQuotes):
        for quote in [mq.yes_quote, mq.no_quote]:
            if quote:
                if quote.age_sec() < self.cfg.min_onbook_sec:
                    wait = self.cfg.min_onbook_sec - quote.age_sec()
                    if wait > 0:
                        await asyncio.sleep(wait)
                try:
                    matched_shares = await self.client.cancel_order(quote.order_id)
                    if matched_shares > 0:
                        fill_usd = matched_shares * quote.price
                        if quote.side == "yes":
                            mq.yes_inventory_usd += fill_usd
                        else:
                            mq.no_inventory_usd += fill_usd
                        logger.debug(
                            f"MM fill detected: {quote.side} "
                            f"{matched_shares:.2f} shares @ {quote.price:.3f} "
                            f"(${fill_usd:.2f}) on {mq.market_id[:12]}"
                        )
                except Exception as e:
                    logger.debug(f"MM cancel {quote.order_id}: {e}")


# ------------------------------------------------------------------
# Module helpers
# ------------------------------------------------------------------

def _book_mid(book) -> float:
    if book.bids and book.asks:
        return (float(book.bids[0][0]) + float(book.asks[0][0])) / 2
    return 0.0


async def _place_limit(
    client: PolymarketClient,
    token_id: str,
    action: str,
    price: float,
    shares: float,
) -> Optional[str]:
    try:
        result = await client.place_order(
            token_id=token_id, action=action, price=price, size=shares
        )
        return result.order_id if result else None
    except Exception as e:
        logger.debug(f"MM place_limit: {e}")
        return None
