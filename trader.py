"""
Trade execution for Leo — Kalshi-only trading bot.

Handles three trade types:
  1. Overround arb  (ArbOpportunity)    — sell both YES and NO legs
  2. Signal trade   (AggregatedSignal)  — single-leg buy on one side
  3. Correlated arb (CorrelatedOpportunity) — single-leg buy on mispriced side

All live orders go through KalshiClient.place_order().
Dry-run mode logs every intended trade without placing real orders.

A per-market cooldown (default 30 min) prevents the same market from
being traded by multiple strategies simultaneously.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from api_clients.kalshi_client import KalshiClient, OrderResult
from arbitrage import ArbOpportunity
from config import Config
from position_manager import PositionManager
from storage import Storage
from strategies.correlated import CorrelatedOpportunity
from strategies.signal_arb import AggregatedSignal

logger = logging.getLogger(__name__)

_COOLDOWN_MINUTES = 30


class Trader:
    def __init__(
        self,
        cfg: Config,
        client: KalshiClient,
        storage: Storage,
        pos_manager: PositionManager | None = None,
    ):
        self.cfg = cfg
        self.arb_cfg = cfg.arbitrage
        self.client = client
        self.storage = storage
        self.pos_manager = pos_manager
        self._total_exposure = 0.0
        # market_id → time of last trade (prevents multi-strategy double-fires)
        self._cooldowns: dict[str, datetime] = {}

    # ------------------------------------------------------------------ #
    #  Cooldown helpers                                                    #
    # ------------------------------------------------------------------ #

    def _in_cooldown(self, market_id: str) -> bool:
        last = self._cooldowns.get(market_id)
        if not last:
            return False
        return (
            datetime.now(timezone.utc) - last
            < timedelta(minutes=_COOLDOWN_MINUTES)
        )

    def _mark_cooldown(self, market_id: str):
        self._cooldowns[market_id] = datetime.now(timezone.utc)

    # ------------------------------------------------------------------ #
    #  Overround arb (two-legged sell)                                     #
    # ------------------------------------------------------------------ #

    async def execute(self, opp: ArbOpportunity) -> bool:
        """Sell both YES and NO legs simultaneously."""
        if self._in_cooldown(opp.market_id):
            return False
        if not self._check_exposure(opp.max_size_usd):
            logger.warning(
                f"Skipping {opp.market_id}: exposure limit reached"
            )
            return False

        size_usd = min(opp.max_size_usd, self.arb_cfg.max_position_usd)
        yes_price_cents = round(opp.yes_bid * 100)
        no_price_cents = round(opp.no_bid * 100)
        contracts = KalshiClient.contracts_for_usd(size_usd, opp.yes_bid)

        if contracts < 1:
            return False

        if self.cfg.dry_run:
            logger.info(
                f"[DRY RUN] overround {opp.market_id!r} | "
                f"sell YES@{opp.yes_bid:.3f} + sell NO@{opp.no_bid:.3f}"
                f" | {contracts} contracts | net={opp.net_profit_pct:.2%}"
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
            self._mark_cooldown(opp.market_id)
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
            self._mark_cooldown(opp.market_id)
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
            logger.error(
                f"Arb execution error for {opp.market_id}: {e}"
            )
            return False

    # ------------------------------------------------------------------ #
    #  Signal trade (single-leg buy)                                       #
    # ------------------------------------------------------------------ #

    async def execute_signal(
        self,
        sig: AggregatedSignal,
        market_map: dict,
    ) -> bool:
        """Buy one side on a signal opportunity."""
        if self._in_cooldown(sig.market_id):
            return False
        if not self._check_exposure(sig.recommended_size_usd):
            return False
        if self._has_position(sig.market_id):
            logger.debug(
                f"Skipping {sig.market_id}: already holding a position"
            )
            return False

        market = market_map.get(sig.market_id)
        if not market:
            return False

        side = sig.recommended_side.lower()
        price_dollars = market.yes_ask if side == "yes" else market.no_ask
        if price_dollars <= 0:
            return False

        size_usd = min(
            sig.recommended_size_usd, self.arb_cfg.max_position_usd
        )
        contracts = KalshiClient.contracts_for_usd(size_usd, price_dollars)
        if contracts < 1:
            return False

        price_cents = round(price_dollars * 100)
        yes_p = price_dollars if side == "yes" else (1 - price_dollars)
        no_p = price_dollars if side == "no" else (1 - price_dollars)

        if self.cfg.dry_run:
            logger.info(
                f"[DRY RUN] signal {sig.source} {sig.market_id!r} | "
                f"buy {side}@{price_dollars:.3f} x{contracts} | "
                f"edge={sig.edge:+.1%}"
            )
            self.storage.log_trade(
                market_id=sig.market_id,
                question=sig.question,
                arb_type=f"signal:{sig.source}",
                yes_price=yes_p,
                no_price=no_p,
                contracts=contracts,
                net_profit_pct=sig.edge,
                dry_run=True,
                status="simulated",
            )
            self._mark_cooldown(sig.market_id)
            return True

        try:
            order = await self._place_leg(
                sig.market_id, side, "buy", contracts, price_cents
            )
            if not order:
                return False
            self._total_exposure += size_usd
            self._mark_cooldown(sig.market_id)
            self.storage.log_trade(
                market_id=sig.market_id,
                question=sig.question,
                arb_type=f"signal:{sig.source}",
                yes_price=yes_p,
                no_price=no_p,
                contracts=contracts,
                net_profit_pct=sig.edge,
                dry_run=False,
                status="placed",
                yes_order_id=order.order_id if side == "yes" else None,
                no_order_id=order.order_id if side == "no" else None,
            )
            logger.info(
                f"Placed signal trade on {sig.market_id} | "
                f"{side} order={order.order_id}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Signal execution error for {sig.market_id}: {e}"
            )
            return False

    # ------------------------------------------------------------------ #
    #  Correlated arb (single-leg buy on mispriced market)                 #
    # ------------------------------------------------------------------ #

    async def execute_correlated(
        self,
        opp: CorrelatedOpportunity,
        market_map: dict,
    ) -> bool:
        """Buy the mispriced side on a correlated-arb opportunity."""
        if self._in_cooldown(opp.market_id_mispriced):
            return False
        if not self._check_exposure(opp.max_size_usd):
            return False
        if self._has_position(opp.market_id_mispriced):
            logger.debug(
                f"Skipping {opp.market_id_mispriced}: already holding"
            )
            return False

        market = market_map.get(opp.market_id_mispriced)
        if not market:
            return False

        side = opp.recommended_side.lower()
        price_dollars = market.yes_ask if side == "yes" else market.no_ask
        if price_dollars <= 0:
            return False

        size_usd = min(opp.max_size_usd, self.arb_cfg.max_position_usd)
        contracts = KalshiClient.contracts_for_usd(size_usd, price_dollars)
        if contracts < 1:
            return False

        price_cents = round(price_dollars * 100)
        yes_p = price_dollars if side == "yes" else (1 - price_dollars)
        no_p = price_dollars if side == "no" else (1 - price_dollars)

        if self.cfg.dry_run:
            logger.info(
                f"[DRY RUN] corr-arb {opp.market_id_mispriced!r} | "
                f"buy {side}@{price_dollars:.3f} x{contracts} | "
                f"edge={opp.edge:+.1%} "
                f"anchor={opp.market_id_anchor}"
            )
            self.storage.log_trade(
                market_id=opp.market_id_mispriced,
                question=opp.question_mispriced,
                arb_type=f"corr:{opp.relation.value}",
                yes_price=yes_p,
                no_price=no_p,
                contracts=contracts,
                net_profit_pct=opp.edge,
                dry_run=True,
                status="simulated",
            )
            self._mark_cooldown(opp.market_id_mispriced)
            return True

        try:
            order = await self._place_leg(
                opp.market_id_mispriced, side, "buy",
                contracts, price_cents,
            )
            if not order:
                return False
            self._total_exposure += size_usd
            self._mark_cooldown(opp.market_id_mispriced)
            self.storage.log_trade(
                market_id=opp.market_id_mispriced,
                question=opp.question_mispriced,
                arb_type=f"corr:{opp.relation.value}",
                yes_price=yes_p,
                no_price=no_p,
                contracts=contracts,
                net_profit_pct=opp.edge,
                dry_run=False,
                status="placed",
                yes_order_id=order.order_id if side == "yes" else None,
                no_order_id=order.order_id if side == "no" else None,
            )
            logger.info(
                f"Placed corr-arb on {opp.market_id_mispriced} | "
                f"order={order.order_id}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Corr execution error for {opp.market_id_mispriced}: {e}"
            )
            return False

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

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

    def _has_position(self, market_id: str) -> bool:
        if not self.pos_manager:
            return False
        pos = self.pos_manager.get_position(market_id)
        return pos is not None and pos.contracts > 0

    def _check_exposure(self, size: float) -> bool:
        return (
            self._total_exposure + size
            <= self.arb_cfg.max_total_exposure_usd
        )
