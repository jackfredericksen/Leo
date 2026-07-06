"""
Trade execution for Leo — Polymarket trading bot.

Handles three trade types:
  1. Overround arb  (ArbOpportunity)        — buy both YES and NO legs cheap
  2. Signal trade   (AggregatedSignal)      — single-leg buy on one side
  3. Correlated arb (CorrelatedOpportunity) — single-leg buy on mispriced side

All live orders go through PolymarketClient.place_order().
Dry-run mode logs every intended trade without placing real orders.

On Polymarket, overround arb is executed by BUYING both YES and NO shares
when their total ask price < $1.00 (guaranteed $1.00 payout at resolution).
This is equivalent to the sell-both-sides arb on Kalshi.

A per-market cooldown (default 30 min) prevents the same market from
being traded by multiple strategies simultaneously.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from api_clients.polymarket_client import PolymarketClient, OrderResult
from arbitrage import ArbOpportunity
from config import Config
from position_manager import PositionManager
from storage import Storage
from strategies.correlated import CorrelatedOpportunity
from strategies.signal_arb import AggregatedSignal
from order_gate import live_orders_allowed, trading_allowed

logger = logging.getLogger(__name__)

_COOLDOWN_MINUTES = 30


class Trader:
    def __init__(
        self,
        cfg: Config,
        client: PolymarketClient,
        storage: Storage,
        pos_manager: PositionManager | None = None,
        alerter=None,
        confluence=None,
    ):
        self.cfg = cfg
        self.arb_cfg = cfg.arbitrage
        self.risk_cfg = cfg.risk
        self.alert_cfg = cfg.alerting
        self.client = client
        self.storage = storage
        self.pos_manager = pos_manager
        self.alerter = alerter
        self.confluence = confluence
        self._total_exposure = 0.0
        self._cooldowns: dict[str, datetime] = {}
        self._today_date: date = datetime.now(timezone.utc).date()
        self._today_usd_deployed: float = 0.0
        self._strategy_daily: dict[str, float] = {}

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
        # Prune expired cooldowns every ~100 marks to prevent unbounded growth
        if len(self._cooldowns) > 200:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=_COOLDOWN_MINUTES)
            self._cooldowns = {k: v for k, v in self._cooldowns.items() if v > cutoff}

    # ------------------------------------------------------------------ #
    #  Daily limit helpers                                                 #
    # ------------------------------------------------------------------ #

    def _reset_daily_if_needed(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._today_date:
            self._today_date = today
            self._today_usd_deployed = 0.0
            self._strategy_daily.clear()
            # Do not zero _total_exposure at midnight — open risk persists

    def _check_strategy_cap(self, strategy_key: str, size_usd: float) -> bool:
        limit = self.risk_cfg.per_strategy_max_usd
        if limit <= 0:
            return True
        self._reset_daily_if_needed()
        used = self._strategy_daily.get(strategy_key, 0.0)
        if used + size_usd > limit:
            logger.warning(
                f"Per-strategy cap hit for {strategy_key}: "
                f"${used:.0f}+${size_usd:.0f} > ${limit:.0f}"
            )
            return False
        return True

    def _reserve_strategy_cap(self, strategy_key: str, size_usd: float) -> None:
        self._strategy_daily[strategy_key] = (
            self._strategy_daily.get(strategy_key, 0.0) + size_usd
        )

    def _release_strategy_cap(self, strategy_key: str, size_usd: float) -> None:
        self._strategy_daily[strategy_key] = max(
            0.0, self._strategy_daily.get(strategy_key, 0.0) - size_usd
        )

    def _check_and_reserve_exposure(
        self, size_usd: float, strategy_key: str = "unknown"
    ) -> bool:
        """Atomically check AND reserve exposure before any await point."""
        self._reset_daily_if_needed()
        if not self._check_strategy_cap(strategy_key, size_usd):
            return False
        if self._today_usd_deployed + size_usd > self.risk_cfg.max_daily_usd_deployed:
            logger.warning(
                f"Daily deployment limit reached: "
                f"${self._today_usd_deployed:.0f}/${self.risk_cfg.max_daily_usd_deployed:.0f}"
            )
            if self.alerter:
                self._alert_task(self.alerter.daily_limit_hit(
                    self._today_usd_deployed, self.risk_cfg.max_daily_usd_deployed
                ))
            return False
        if self._total_exposure + size_usd > self.arb_cfg.max_total_exposure_usd:
            logger.warning(
                f"Total exposure limit reached: "
                f"${self._total_exposure:.0f}/${self.arb_cfg.max_total_exposure_usd:.0f}"
            )
            return False
        self._today_usd_deployed += size_usd
        self._total_exposure += size_usd
        self._reserve_strategy_cap(strategy_key, size_usd)
        return True

    def _release_exposure(self, size_usd: float, strategy_key: str = "unknown") -> None:
        self._today_usd_deployed = max(0.0, self._today_usd_deployed - size_usd)
        self._total_exposure = max(0.0, self._total_exposure - size_usd)
        self._release_strategy_cap(strategy_key, size_usd)

    def _alert_task(self, coro) -> None:
        """Fire-and-forget an alerter coroutine (only if coroutine is not None)."""
        if coro is not None:
            asyncio.create_task(coro)

    # ------------------------------------------------------------------ #
    #  Overround arb (buy both legs cheap)                                 #
    # ------------------------------------------------------------------ #

    async def execute(self, opp: ArbOpportunity) -> bool:
        """
        Buy both YES and NO shares when total ask < $1.00.
        Profit = $1.00 payout - (YES_ask + NO_ask) per share.
        """
        ok, reason = trading_allowed()
        if not ok:
            logger.debug(f"Arb blocked ({reason}): {opp.market_id[:16]}")
            return False
        sk = opp.arb_type or "arb"
        if self._in_cooldown(opp.market_id):
            return False
        if self.confluence:
            self.confluence.update(opp.market_id, "arb", "both")
        if self.confluence and not self.confluence.passes(opp.market_id, "both"):
            logger.debug(f"Confluence gate: {opp.market_id[:16]} skipped (arb)")
            return False

        size_usd = min(opp.max_size_usd, self.arb_cfg.max_position_usd)
        if not self._check_and_reserve_exposure(size_usd, sk):
            return False

        # We buy YES at yes_ask and NO at no_ask
        yes_ask = opp.yes_ask
        no_ask  = opp.no_ask
        # shares to buy on the YES side (NO side will match)
        shares = PolymarketClient.usdc_to_shares(size_usd / 2, yes_ask)

        if shares < 1.0:
            self._release_exposure(size_usd, sk)
            return False

        if self.cfg.dry_run:
            logger.info(
                f"[DRY RUN] overround {opp.market_id!r} | "
                f"buy YES@{yes_ask:.3f} + buy NO@{no_ask:.3f} | "
                f"{shares:.1f} shares | net={opp.net_profit_pct:.2%}"
            )
            self.storage.log_trade(
                market_id=opp.market_id,
                question=opp.question,
                arb_type=opp.arb_type,
                side="both",
                yes_price=yes_ask,
                no_price=no_ask,
                contracts=int(shares),
                net_profit_pct=opp.net_profit_pct,
                dry_run=True,
                status="simulated",
                size_usd=size_usd,
            )
            self._release_exposure(size_usd, sk)  # dry-run doesn't consume real limits
            self._mark_cooldown(opp.market_id)
            return True

        # Mark cooldown before any await so concurrent tasks respect it immediately
        self._mark_cooldown(opp.market_id)
        try:
            yes_order, no_order = await asyncio.gather(
                self._place_leg(
                    opp.yes_token_id, "yes", "buy", yes_ask, shares
                ),
                self._place_leg(
                    opp.no_token_id, "no", "buy", no_ask, shares
                ),
            )

            if not yes_order or not no_order:
                logger.error(f"One leg failed for {opp.market_id}, cancelling")
                self._release_exposure(size_usd, sk)
                await self._cancel_if_placed(yes_order, no_order)
                return False

            self.storage.log_trade(
                market_id=opp.market_id,
                question=opp.question,
                arb_type=opp.arb_type,
                side="both",
                yes_price=yes_ask,
                no_price=no_ask,
                contracts=int(shares),
                net_profit_pct=opp.net_profit_pct,
                dry_run=False,
                status="placed",
                yes_order_id=yes_order.order_id,
                no_order_id=no_order.order_id,
                size_usd=size_usd,
            )
            logger.info(
                f"Placed arb on {opp.market_id} | "
                f"YES={yes_order.order_id} NO={no_order.order_id}"
            )
            if self.alerter and size_usd >= self.alert_cfg.large_fill_threshold_usd:
                self._alert_task(self.alerter.large_fill(
                    opp.market_id, "both", size_usd, "overround"
                ))
            return True

        except Exception as e:
            self._release_exposure(size_usd)
            logger.error(f"Arb execution error for {opp.market_id}: {e}")
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
        ok, reason = trading_allowed()
        if not ok:
            logger.debug(f"Signal blocked ({reason}): {sig.market_id[:16]}")
            return False
        sk = f"signal:{sig.source or 'signal'}"
        if self._in_cooldown(sig.market_id):
            return False
        if self.confluence:
            self.confluence.update(sig.market_id, sig.source or "signal", sig.recommended_side)
        if self.confluence and not self.confluence.passes(sig.market_id, sig.recommended_side):
            logger.debug(f"Confluence gate: {sig.market_id[:16]} {sig.recommended_side} skipped")
            return False
        if self._has_position(sig.market_id):
            logger.debug(f"Skipping {sig.market_id}: already holding a position")
            return False

        market = market_map.get(sig.market_id)
        if not market:
            return False

        side = sig.recommended_side.lower()
        if side == "yes":
            token_id      = market.yes_token_id
            price_dollars = market.yes_ask
        else:
            token_id      = market.no_token_id
            price_dollars = market.no_ask

        if price_dollars <= 0 or not token_id:
            return False

        size_usd = min(sig.recommended_size_usd, self.arb_cfg.max_position_usd)
        if not self._check_and_reserve_exposure(size_usd, sk):
            return False

        shares = PolymarketClient.usdc_to_shares(size_usd, price_dollars)
        if shares < 1.0:
            self._release_exposure(size_usd, sk)
            if size_usd <= 0:
                logger.debug(
                    f"Signal skip $0 size ({sig.source}): "
                    f"{sig.market_id[:16]} edge={sig.edge:+.1%}"
                )
            return False

        # Use actual market prices for storage (not a complement approximation)
        yes_p = market.yes_ask
        no_p  = market.no_ask

        if self.cfg.dry_run:
            logger.info(
                f"[DRY RUN] signal {sig.source} {sig.market_id!r} | "
                f"buy {side}@{price_dollars:.3f} x{shares:.1f} | "
                f"edge={sig.edge:+.1%}"
            )
            self.storage.log_trade(
                market_id=sig.market_id,
                question=sig.question,
                arb_type=f"signal:{sig.source}",
                side=side,
                yes_price=yes_p,
                no_price=no_p,
                contracts=int(shares),
                net_profit_pct=sig.edge,
                dry_run=True,
                status="simulated",
                size_usd=size_usd,
            )
            self._release_exposure(size_usd, sk)  # dry-run doesn't consume real limits
            self._mark_cooldown(sig.market_id)
            if self.alerter and sig.edge * 100 >= self.alert_cfg.big_edge_threshold_pct:
                self._alert_task(self.alerter.big_edge(
                    sig.question or sig.market_id, sig.edge * 100,
                    sig.source or "signal", side, size_usd,
                ))
            return True

        # Mark cooldown before any await so concurrent tasks respect it immediately
        self._mark_cooldown(sig.market_id)
        try:
            order = await self._place_leg(
                token_id, side, "buy", price_dollars, shares
            )
            if not order:
                self._release_exposure(size_usd, sk)
                return False
            self.storage.log_trade(
                market_id=sig.market_id,
                question=sig.question,
                arb_type=f"signal:{sig.source}",
                side=side,
                yes_price=yes_p,
                no_price=no_p,
                contracts=int(shares),
                net_profit_pct=sig.edge,
                dry_run=False,
                status="placed",
                yes_order_id=order.order_id if side == "yes" else None,
                no_order_id=order.order_id if side == "no" else None,
                size_usd=size_usd,
            )
            logger.info(
                f"Placed signal trade on {sig.market_id} | "
                f"{side} order={order.order_id}"
            )
            if self.alerter and size_usd >= self.alert_cfg.large_fill_threshold_usd:
                self._alert_task(self.alerter.large_fill(
                    sig.market_id, side, size_usd, sig.source or "signal"
                ))
            return True
        except Exception as e:
            self._release_exposure(size_usd, sk)
            logger.error(f"Signal execution error for {sig.market_id}: {e}")
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
        ok, reason = trading_allowed()
        if not ok:
            logger.debug(f"Corr blocked ({reason}): {opp.market_id_mispriced[:16]}")
            return False
        sk = f"corr:{opp.relation.value}"
        if self._in_cooldown(opp.market_id_mispriced):
            return False
        if self.confluence:
            self.confluence.update(
                opp.market_id_mispriced,
                f"corr:{opp.relation.value}",
                opp.recommended_side.lower(),
            )
        if self.confluence and not self.confluence.passes(opp.market_id_mispriced, opp.recommended_side):
            logger.debug(f"Confluence gate: {opp.market_id_mispriced[:16]} corr skipped")
            return False
        if self._has_position(opp.market_id_mispriced):
            logger.debug(f"Skipping {opp.market_id_mispriced}: already holding")
            return False

        market = market_map.get(opp.market_id_mispriced)
        if not market:
            return False

        side = opp.recommended_side.lower()
        if side == "yes":
            token_id      = market.yes_token_id
            price_dollars = market.yes_ask
        else:
            token_id      = market.no_token_id
            price_dollars = market.no_ask

        if price_dollars <= 0 or not token_id:
            return False

        size_usd = min(opp.max_size_usd, self.arb_cfg.max_position_usd)
        if not self._check_and_reserve_exposure(size_usd, sk):
            return False

        shares = PolymarketClient.usdc_to_shares(size_usd, price_dollars)
        if shares < 1.0:
            self._release_exposure(size_usd, sk)
            return False

        # Use actual market prices for storage (not a complement approximation)
        yes_p = market.yes_ask
        no_p  = market.no_ask

        if self.cfg.dry_run:
            logger.info(
                f"[DRY RUN] corr-arb {opp.market_id_mispriced!r} | "
                f"buy {side}@{price_dollars:.3f} x{shares:.1f} | "
                f"edge={opp.edge:+.1%} "
                f"anchor={opp.market_id_anchor}"
            )
            self.storage.log_trade(
                market_id=opp.market_id_mispriced,
                question=opp.question_mispriced,
                arb_type=f"corr:{opp.relation.value}",
                side=side,
                yes_price=yes_p,
                no_price=no_p,
                contracts=int(shares),
                net_profit_pct=opp.edge,
                dry_run=True,
                status="simulated",
                size_usd=size_usd,
            )
            self._release_exposure(size_usd, sk)  # dry-run doesn't consume real limits
            self._mark_cooldown(opp.market_id_mispriced)
            return True

        # Mark cooldown before any await so concurrent tasks respect it immediately
        self._mark_cooldown(opp.market_id_mispriced)
        try:
            order = await self._place_leg(
                token_id, side, "buy", price_dollars, shares
            )
            if not order:
                self._release_exposure(size_usd, sk)
                return False
            self.storage.log_trade(
                market_id=opp.market_id_mispriced,
                question=opp.question_mispriced,
                arb_type=f"corr:{opp.relation.value}",
                side=side,
                yes_price=yes_p,
                no_price=no_p,
                contracts=int(shares),
                net_profit_pct=opp.edge,
                dry_run=False,
                status="placed",
                yes_order_id=order.order_id if side == "yes" else None,
                no_order_id=order.order_id if side == "no" else None,
                size_usd=size_usd,
            )
            logger.info(
                f"Placed corr-arb on {opp.market_id_mispriced} | "
                f"order={order.order_id}"
            )
            if self.alerter and size_usd >= self.alert_cfg.large_fill_threshold_usd:
                self._alert_task(self.alerter.large_fill(
                    opp.market_id_mispriced, side, size_usd,
                    f"corr:{opp.relation.value}"
                ))
            return True
        except Exception as e:
            self._release_exposure(size_usd, sk)
            logger.error(f"Corr execution error for {opp.market_id_mispriced}: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    async def _place_leg(
        self,
        token_id: str,
        side: str,
        action: str,
        price: float,
        shares: float,
    ) -> OrderResult | None:
        ok, reason = live_orders_allowed()
        if not ok:
            logger.warning(f"Live order blocked ({reason}): {action} {side}")
            return None
        try:
            result = await self.client.place_order(
                token_id=token_id,
                action=action,
                price=price,
                size=shares,
            )
            if result:
                result.side = side
            return result
        except Exception as e:
            logger.error(f"Failed to place {action} {side} ({token_id[:8]}): {e}")
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

