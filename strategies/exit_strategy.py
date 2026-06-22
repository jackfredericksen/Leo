"""
Exit strategy — sell near-certain positions to lock profit.

Scans open positions where current mid price > exit_threshold (default 0.90).
Places a live sell order at the best bid; skips if the bid is too low.
Uses a session-level set to avoid repeatedly trying to exit the same position.
"""

import logging
from dataclasses import dataclass

from api_clients.polymarket_client import PolymarketClient
from order_gate import live_orders_allowed

logger = logging.getLogger(__name__)


@dataclass
class ExitConfig:
    enabled: bool = True
    exit_threshold: float = 0.90    # sell when mid > this
    min_contracts: float = 1.0      # skip positions smaller than this
    min_bid_ratio: float = 0.94     # don't sell if best_bid < threshold * this
    refresh_interval_sec: int = 60


class ExitStrategy:
    """
    Periodic position exit scanner.
    Call scan_and_exit() from exit_loop in main.py.
    """

    def __init__(self, cfg: ExitConfig, client: PolymarketClient):
        self.cfg = cfg
        self.client = client
        self._exited: set[str] = set()   # "market_id:side" keys already exited this session

    async def scan_and_exit(
        self,
        positions: list,
        market_map: dict,
        dry_run: bool = True,
    ) -> list[dict]:
        """
        positions : list of open position objects from PositionManager.
        Returns list of exit records for logging/UI.
        """
        if not self.cfg.enabled:
            return []

        exits = []
        for pos in positions:
            key = f"{pos.market_id}:{pos.side}"
            if key in self._exited:
                continue
            if pos.contracts < self.cfg.min_contracts:
                continue

            market = market_map.get(pos.market_id)
            if not market:
                continue

            side = pos.side.lower()
            current_price = market.yes_price if side == "yes" else market.no_price
            token_id = market.yes_token_id if side == "yes" else market.no_token_id

            if current_price < self.cfg.exit_threshold:
                continue

            if dry_run:
                logger.info(
                    f"[DRY RUN] Exit: would sell {pos.contracts:.1f} {side.upper()}"
                    f" @ {current_price:.3f} on {pos.market_id[:16]}"
                )
                self._exited.add(key)
                exits.append({
                    "market_id": pos.market_id,
                    "question": getattr(pos, "question", "")[:60],
                    "side": side,
                    "price": round(current_price, 3),
                    "contracts": pos.contracts,
                })
                continue

            try:
                book = await self.client.get_orderbook(token_id)
                sell_price = book.best_bid if book.bids else current_price * 0.97
                min_acceptable = self.cfg.exit_threshold * self.cfg.min_bid_ratio
                if sell_price < min_acceptable:
                    logger.debug(
                        f"Exit: best bid {sell_price:.3f} < min {min_acceptable:.3f}, "
                        f"skipping {pos.market_id[:12]}"
                    )
                    continue

                ok, reason = live_orders_allowed()
                if not ok:
                    logger.debug(f"Exit sell blocked ({reason}): {pos.market_id[:12]}")
                    continue

                result = await self.client.place_order(
                    token_id=token_id,
                    action="sell",
                    price=round(sell_price, 3),
                    size=pos.contracts,
                )
                if result:
                    logger.info(
                        f"Exit: sold {pos.contracts:.1f} {side.upper()}"
                        f" @ {sell_price:.3f} on {pos.market_id[:16]}"
                    )
                    self._exited.add(key)
                    exits.append({
                        "market_id": pos.market_id,
                        "question": getattr(pos, "question", "")[:60],
                        "side": side,
                        "price": round(sell_price, 3),
                        "contracts": pos.contracts,
                    })
            except Exception as e:
                logger.debug(f"Exit scan error {pos.market_id}: {e}")

        return exits
