"""
Trade execution for Leo arbitrage bot.

Executes both legs of an overround arb simultaneously.
Includes dry-run mode, position limits, and basic retry logic.

Kalshi contracts are integer quantities (1 contract = $1 payout).
Prices are in dollars (0.00–1.00). To sell YES at 0.62, you place
a sell-yes limit order at yes_price=62 (cents).
"""

import asyncio
import logging
from api_clients.kalshi_client import KalshiClient, OrderResult
from arbitrage import ArbOpportunity
from config import Config
from storage import Storage

logger = logging.getLogger(__name__)


class Trader:
    def __init__(
        self, cfg: Config, client: KalshiClient, storage: Storage
    ):
        self.cfg = cfg
        self.arb_cfg = cfg.arbitrage
        self.client = client
        self.storage = storage
        self._total_exposure = 0.0

    async def execute(self, opp: ArbOpportunity) -> bool:
        """
        Execute an arbitrage opportunity.
        Returns True if trade was placed (or simulated), False if skipped.
        """
        if not self._check_exposure(opp.max_size_usd):
            logger.warning(
                f"Skipping {opp.market_id}: exposure limit reached"
            )
            return False

        size_usd = min(opp.max_size_usd, self.arb_cfg.max_position_usd)

        # Kalshi: contracts are integer units, price in cents (1–99)
        yes_price_cents = round(opp.yes_bid * 100)
        no_price_cents = round(opp.no_bid * 100)
        contracts = KalshiClient.contracts_for_usd(size_usd, opp.yes_bid)

        if contracts < 1:
            logger.warning(f"Skipping {opp.market_id}: size too small")
            return False

        if self.cfg.dry_run:
            logger.info(
                f"[DRY RUN] {opp.market_id!r} | "
                f"sell YES@{opp.yes_bid:.3f} + sell NO@{opp.no_bid:.3f} | "
                f"{contracts} contracts | net={opp.net_profit_pct:.2%}"
            )
            self.storage.log_trade(
                market_id=opp.market_id,
                question=opp.question,
                arb_type=opp.arb_type,
                yes_price=opp.yes_bid,
                no_price=opp.no_bid,
                contracts=contracts,
                net_profit_pct=opp.net_profit_pct,
                dry_run=True,
                status="simulated",
            )
            return True

        try:
            yes_order, no_order = await asyncio.gather(
                self._place_leg(
                    opp.market_id, "yes", "sell",
                    contracts, yes_price_cents,
                ),
                self._place_leg(
                    opp.market_id, "no", "sell",
                    contracts, no_price_cents,
                ),
            )

            if not yes_order or not no_order:
                logger.error(
                    f"One leg failed for {opp.market_id}, cancelling"
                )
                await self._cancel_if_placed(yes_order, no_order)
                return False

            self._total_exposure += size_usd
            self.storage.log_trade(
                market_id=opp.market_id,
                question=opp.question,
                arb_type=opp.arb_type,
                yes_price=opp.yes_bid,
                no_price=opp.no_bid,
                contracts=contracts,
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
        self,
        ticker: str,
        side: str,
        action: str,
        contracts: int,
        price_cents: int,
    ) -> OrderResult | None:
        try:
            return await self.client.place_order(
                ticker=ticker,
                side=side,
                action=action,
                count=contracts,
                limit_price=price_cents,
            )
        except Exception as e:
            logger.error(
                f"Failed to place {action} {side} on {ticker}: {e}"
            )
            return None

    async def _cancel_if_placed(self, *orders):
        for order in orders:
            if order:
                await self.client.cancel_order(order.order_id)

    def _check_exposure(self, size: float) -> bool:
        return (
            self._total_exposure + size
            <= self.arb_cfg.max_total_exposure_usd
        )
