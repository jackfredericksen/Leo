"""
Strategy: Favorite Mispricing (Short Overconfident Favorites).

The favorite-longshot bias in prediction markets causes heavy favorites to be
systematically overpriced. Markets above ~88¢ in thin, low-volume events resolve
YES less often than their price implies — retail traders overpay for certainty.

Signal: markets where:
  1. YES price > threshold (default 0.88)
  2. Low volume / open interest (thin market = less efficient pricing)
  3. Resolution is not imminent (otherwise the outcome really is near-certain)
  4. Our fair-value estimate (baseline - discount_factor) shows NO-side edge

Trade: post a NO limit order (short the favorite). Profit from mean reversion
back toward fair value or from the persistent ~5-6% over-pricing of favorites.

Uses existing execute_signal() pipeline in trader.py.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from api_clients.polymarket_client import Market
from strategies.kelly import KellySizer
from strategies.signal_arb import AggregatedSignal, ask_edge

logger = logging.getLogger(__name__)


@dataclass
class FavoriteShortConfig:
    enabled: bool = True
    min_yes_price: float = 0.88
    max_yes_price: float = 0.97         # skip near-certain markets
    max_volume_usd: float = 5000.0      # skip efficiently-priced liquid markets
    max_open_interest_usd: float = 3000.0
    min_days_to_resolve: float = 7.0    # don't short near resolution
    discount_factor: float = 0.06       # favorites are ~6% overpriced on average
    fee_pct: float = 0.02
    min_edge: float = 0.03
    max_position_usd: float = 75.0
    kelly_fraction: float = 0.08        # smaller Kelly — structural edge, not sharp signal
    refresh_interval_sec: int = 300


class FavoriteShortDetector:
    """
    Screens for overpriced favorites and signals short (NO) positions.

    Fair-value model: apply a fixed discount_factor to the market YES price.
    This captures the empirical favorite-longshot bias documented across
    prediction markets. Works best in thin, long-horizon political/entertainment
    markets where the crowd hasn't corrected the bias.
    """

    def __init__(self, cfg: FavoriteShortConfig, bankroll: float):
        self.cfg = cfg
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)

    def scan(self, markets: list[Market]) -> list[AggregatedSignal]:
        results = []
        now = datetime.now(timezone.utc)

        for market in markets:
            try:
                if market.status != "open":
                    continue

                yes_price = market.yes_price
                if not (self.cfg.min_yes_price <= yes_price <= self.cfg.max_yes_price):
                    continue

                if market.volume > self.cfg.max_volume_usd:
                    continue
                if market.open_interest > self.cfg.max_open_interest_usd:
                    continue

                hours_left = (market.close_time - now).total_seconds() / 3600
                if hours_left < self.cfg.min_days_to_resolve * 24:
                    continue

                model_prob = max(0.05, min(0.95, yes_price - self.cfg.discount_factor))

                result = ask_edge(
                    model_prob,
                    market.yes_ask, market.no_ask, market.yes_bid,
                    self.cfg.fee_pct, self.cfg.min_edge,
                )
                if not result:
                    continue
                edge, side, entry_price = result

                if side != "no":
                    continue

                size = self.sizer.size(
                    1.0 - model_prob, entry_price, self.cfg.max_position_usd
                )

                results.append(AggregatedSignal(
                    market_id=market.market_id,
                    question=market.question,
                    market_prob=yes_price,
                    model_prob=model_prob,
                    edge=edge,
                    recommended_side="no",
                    source="favorite_short",
                    confidence=0.65,
                    reasoning=(
                        f"mkt={yes_price:.0%} "
                        f"fair≈{model_prob:.0%} "
                        f"vol=${market.volume:.0f} "
                        f"oi=${market.open_interest:.0f} "
                        f"{hours_left/24:.0f}d left"
                    ),
                    recommended_size_usd=size,
                ))

            except Exception as e:
                logger.debug(f"FavoriteShort error on {market.market_id}: {e}")

        return sorted(results, key=lambda x: x.edge, reverse=True)
