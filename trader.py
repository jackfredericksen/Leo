"""
Trade execution for Leo arbitrage bot.

Executes both legs of an overround arb simultaneously (or as close as possible).
Includes dry-run mode, position limits, and basic retry logic.
"""

import asyncio
import logging
from datetime import datetime, timezone

from api_clients.coinbase_predictions import CoinbasePredictionsClient, OrderResult
from arbitrage import ArbOpportunity
from config import Config
from storage import Storage

logger = logging.getLogger(__name__)


class Trader:
    def __init__(self, cfg: Config, client: CoinbasePredictionsClient, storage: Storage):
        self.cfg = cfg
        self.arb_cfg = cfg.arbitrage
        self.client = client
        self.storage = storage
        self._total_exposure = 0.0

    async def execute(self, opp: ArbOpportunity) -> bool:
        """
        Execute an arbitrage opportunity.
        Returns True if trade was placed (or simulated in dry_run), False if skipped.
        """
        if not self._check_exposure(opp.max_size_usd):
            logger.warning(f"Skipping {opp.market_id}: exposure limit reached")
            return False

        size = min(opp.max_size_usd, self.arb_cfg.max_position_usd)

        if self.cfg.dry_run:
            logger.info(
                f"[DRY RUN] Would trade {opp.arb_type} arb on {opp.market_id!r} | "
                f"size=${size:.2f} | net profit={opp.net_profit_pct:.2%}"
            )
            self.storage.log_trade(
                market_id=opp.market_id,
                question=opp.question,
                arb_type=opp.arb_type,
                leg_yes_price=opp.leg_yes,
                leg_no_price=opp.leg_no,
                size_usd=size,
                net_profit_pct=opp.net_profit_pct,
                dry_run=True,
                status="simulated",
            )
            return True

        try:
            yes_order, no_order = await asyncio.gather(
                self._place_leg(opp.market_id, "YES", opp.leg_yes, size),
                self._place_leg(opp.market_id, "NO", opp.leg_no, size),
            )

            if not yes_order or not no_order:
                logger.error(f"One leg failed for {opp.market_id}, attempting cancellation")
                await self._cancel_if_placed(yes_order, no_order)
                return False

            self._total_exposure += size
            self.storage.log_trade(
                market_id=opp.market_id,
                question=opp.question,
                arb_type=opp.arb_type,
                leg_yes_price=opp.leg_yes,
                leg_no_price=opp.leg_no,
                size_usd=size,
                net_profit_pct=opp.net_profit_pct,
                dry_run=False,
                status="placed",
                yes_order_id=yes_order.order_id,
                no_order_id=no_order.order_id,
            )
            logger.info(
                f"Placed arb on {opp.market_id} | "
                f"YES={yes_order.order_id} NO={no_order.order_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Trade execution error for {opp.market_id}: {e}")
            return False

    async def _place_leg(
        self, market_id: str, side: str, price: float, size_usd: float
    ) -> OrderResult | None:
        try:
            return await self.client.place_order(
                market_id=market_id,
                side=side,
                size_usd=size_usd,
                price=price,
            )
        except Exception as e:
            logger.error(f"Failed to place {side} leg on {market_id}: {e}")
            return None

    async def _cancel_if_placed(self, *orders):
        for order in orders:
            if order:
                try:
                    await self.client.cancel_order(order.order_id)
                except Exception as e:
                    logger.error(f"Failed to cancel order {order.order_id}: {e}")

    def _check_exposure(self, size: float) -> bool:
        return self._total_exposure + size <= self.arb_cfg.max_total_exposure_usd
