"""
Strategy: Order Book Imbalance (OFI) Momentum.

Uses real-time CLOB order book depth to compute Order Flow Imbalance (OFI),
a 5–15 minute directional signal documented to be predictive of price moves
in Polymarket's microstructure (arXiv:2604.24366).

OFI = (bid_volume_L5 - ask_volume_L5) / total_volume_L5
  +1 = all depth on bid side (strong buy pressure)
  -1 = all depth on ask side (strong sell pressure)

Regime detection (no historical price feed needed):
  Activity ratio = volume_24h / max(liquidity_usd, 1)
  High ratio (> high_activity_ratio): info-dense regime → follow OFI direction
  Low ratio (< low_activity_ratio):   quiet regime       → fade OFI extremes

In quiet markets, extreme OFI is a transient imbalance that mean-reverts.
In active markets, it reflects informed order flow and predicts continuation.

Requires live CLOB order book data — makes one async get_orderbook() call
per candidate market per scan. Candidate selection limits API calls.
"""

import asyncio
import logging
from dataclasses import dataclass

from api_clients.polymarket_client import Market, Orderbook, PolymarketClient
from strategies.kelly import KellySizer
from strategies.signal_arb import AggregatedSignal

logger = logging.getLogger(__name__)


@dataclass
class OrderbookMomentumConfig:
    enabled: bool = True
    ofi_threshold: float = 0.45         # abs(OFI) must exceed this to trade
    orderbook_levels: int = 5           # top N price levels for OFI calc
    high_activity_ratio: float = 2.0    # volume_24h / liquidity above this → follow
    low_activity_ratio: float = 0.3     # below this → fade
    min_liquidity_usd: float = 500.0    # skip illiquid markets
    max_spread_cents: float = 0.12      # skip very wide-spread markets
    max_candidate_markets: int = 20     # cap API calls per scan
    fee_pct: float = 0.02
    min_edge: float = 0.04
    max_position_usd: float = 75.0
    kelly_fraction: float = 0.10
    refresh_interval_sec: int = 120


def _ofi(book: Orderbook, levels: int) -> float:
    """Compute order flow imbalance from top-N levels of the order book."""
    bid_vol = sum(float(b[1]) for b in book.bids[:levels])
    ask_vol = sum(float(a[1]) for a in book.asks[:levels])
    total = bid_vol + ask_vol
    if total < 1.0:
        return 0.0
    return (bid_vol - ask_vol) / total


def _activity_ratio(market: Market) -> float:
    liq = max(market.liquidity_usd, 1.0)
    return market.volume_24h / liq


class OrderbookMomentumDetector:
    """
    Fetches live order books for candidate markets and generates momentum
    signals based on order flow imbalance.
    """

    def __init__(self, cfg: OrderbookMomentumConfig, client: PolymarketClient, bankroll: float):
        self.cfg = cfg
        self.client = client
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)

    async def scan(self, markets: list[Market]) -> list[AggregatedSignal]:
        candidates = self._select_candidates(markets)
        if not candidates:
            return []

        # Fetch all orderbooks in parallel — one call per candidate
        books = await asyncio.gather(
            *[self.client.get_orderbook(m.yes_token_id) for m in candidates],
            return_exceptions=True,
        )

        results = []
        for market, book in zip(candidates, books):
            try:
                if isinstance(book, Exception) or not book:
                    continue
                if not book.bids or not book.asks:
                    continue

                spread = float(book.asks[0][0]) - float(book.bids[0][0])
                if spread > self.cfg.max_spread_cents:
                    continue

                ofi = _ofi(book, self.cfg.orderbook_levels)
                if abs(ofi) < self.cfg.ofi_threshold:
                    continue

                ratio = _activity_ratio(market)
                signal = self._regime_signal(ofi, ratio)
                if signal is None:
                    continue

                side, direction = signal
                mid = (float(book.bids[0][0]) + float(book.asks[0][0])) / 2
                entry = market.yes_ask if side == "yes" else market.no_ask

                shift = abs(ofi) * 0.05
                if side == "yes":
                    model_prob = min(0.95, mid + shift)
                else:
                    model_prob = max(0.05, mid - shift)
                    model_prob = 1.0 - model_prob

                edge = model_prob - entry - self.cfg.fee_pct
                if edge < self.cfg.min_edge:
                    continue

                size = self.sizer.size(model_prob, entry, self.cfg.max_position_usd)

                regime_label = "follow" if direction == "follow" else "fade"
                results.append(AggregatedSignal(
                    market_id=market.market_id,
                    question=market.question,
                    market_prob=market.yes_price,
                    model_prob=model_prob,
                    edge=edge,
                    recommended_side=side,
                    source=f"ofi:{regime_label}",
                    confidence=min(0.9, 0.5 + abs(ofi) * 0.5),
                    reasoning=(
                        f"ofi={ofi:+.2f} "
                        f"ratio={ratio:.1f} "
                        f"spread={spread:.3f} "
                        f"regime={regime_label}"
                    ),
                    recommended_size_usd=size,
                ))

            except Exception as e:
                logger.debug(f"OFI error on {market.market_id}: {e}")

        return sorted(results, key=lambda x: x.edge, reverse=True)

    def _select_candidates(self, markets: list[Market]) -> list[Market]:
        candidates = [
            m for m in markets
            if m.status == "open"
            and m.liquidity_usd >= self.cfg.min_liquidity_usd
            and m.yes_token_id
        ]
        # Sort by volume_24h descending — most active markets first
        candidates.sort(key=lambda m: m.volume_24h, reverse=True)
        return candidates[: self.cfg.max_candidate_markets]

    def _regime_signal(
        self, ofi: float, ratio: float
    ):
        """
        Returns (side, 'follow'|'fade') or None if no tradeable signal.

        High-activity regime: follow OFI (momentum)
        Low-activity regime:  fade OFI extremes (mean reversion)
        Mid-activity: no signal — too ambiguous
        """
        if ratio >= self.cfg.high_activity_ratio:
            # Follow: positive OFI → buy YES
            if ofi >= self.cfg.ofi_threshold:
                return ("yes", "follow")
            if ofi <= -self.cfg.ofi_threshold:
                return ("no", "follow")

        elif ratio <= self.cfg.low_activity_ratio:
            # Fade: positive OFI (bid-heavy) → likely to revert → buy NO
            if ofi >= self.cfg.ofi_threshold:
                return ("no", "fade")
            if ofi <= -self.cfg.ofi_threshold:
                return ("yes", "fade")

        return None
