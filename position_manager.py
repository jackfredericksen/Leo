"""
Position manager — tracks open positions and computes real-time P&L.

Polls Kalshi's /portfolio/positions and /portfolio/balance endpoints to
maintain a live view of:
  - All open positions (market, side, size, avg entry price)
  - Unrealized P&L (current mid vs avg entry)
  - Realized P&L from settled/closed positions
  - Account balance

Positions are refreshed every `refresh_interval_sec` seconds (default 30s).
For real-time fill tracking, the fill loop should call `record_fill()`
immediately when an order completes.

P&L model:
  For a YES position of N contracts entered at avg_price:
    unrealized_pnl = N * (current_mid - avg_price)

  On resolution:
    realized_pnl = N * (1.0 - avg_price)  if YES resolves
    realized_pnl = N * (0.0 - avg_price)  = -N * avg_price  if NO resolves
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from api_clients.kalshi_client import KalshiClient, Market

logger = logging.getLogger(__name__)


@dataclass
class Position:
    market_id: str
    question: str
    side: str               # "yes" | "no"
    contracts: int          # total open contracts
    avg_price: float        # average entry price (0–1)
    current_mid: float      # latest mid price from market data
    unrealized_pnl: float   # current mark-to-market
    close_time: datetime
    status: str             # "open" | "closed" | "settled"
    result: Optional[str]   # "yes" | "no" | None — resolution result


@dataclass
class PortfolioSnapshot:
    balance_usd: float
    positions: list[Position]
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    taken_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status == "open"]

    @property
    def position_count(self) -> int:
        return len(self.open_positions)


class PositionManager:
    def __init__(self, client: KalshiClient, refresh_interval_sec: int = 30):
        self.client = client
        self.refresh_interval_sec = refresh_interval_sec
        self._snapshot: Optional[PortfolioSnapshot] = None
        self._last_refresh: Optional[datetime] = None
        self._realized_pnl: float = 0.0  # accumulated from settled positions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def refresh(self, market_map: dict[str, Market]):
        """
        Fetch positions + balance from Kalshi and build a portfolio snapshot.
        `market_map` is a dict of ticker → Market with current prices.
        """
        try:
            balance = await self.client.get_balance()
            raw_positions = await self.client.get_positions()
        except Exception as e:
            logger.error(f"PositionManager refresh failed: {e}")
            return

        positions = []
        unrealized = 0.0

        for rp in raw_positions:
            market_id = rp.get("ticker", "")
            market = market_map.get(market_id)

            # Kalshi position fields
            yes_position = int(rp.get("position", 0) or 0)
            # Positive = net long YES, negative = net long NO
            if yes_position >= 0:
                side = "yes"
                contracts = yes_position
            else:
                side = "no"
                contracts = abs(yes_position)

            if contracts == 0:
                continue

            avg_price = float(rp.get("average_price", 0.5) or 0.5) / 100

            if market:
                current_mid = (market.yes_bid + market.yes_ask) / 2
                if side == "no":
                    current_mid = (market.no_bid + market.no_ask) / 2
                question = market.question
                close_time = market.close_time
                status = market.status
                result = market.result
            else:
                current_mid = avg_price
                question = market_id
                close_time = datetime.now(timezone.utc)
                status = "unknown"
                result = None

            unreal = contracts * (current_mid - avg_price)
            unrealized += unreal

            # Track realized P&L from settled positions
            if status == "settled" and result:
                payout = 1.0 if (result == side) else 0.0
                realized = contracts * (payout - avg_price)
                self._realized_pnl += realized
                logger.info(
                    f"Settled {market_id}: side={side} result={result} "
                    f"realized=${realized:.2f}"
                )

            positions.append(Position(
                market_id=market_id,
                question=question,
                side=side,
                contracts=contracts,
                avg_price=avg_price,
                current_mid=current_mid,
                unrealized_pnl=unreal,
                close_time=close_time,
                status=status,
                result=result,
            ))

        self._snapshot = PortfolioSnapshot(
            balance_usd=balance,
            positions=positions,
            unrealized_pnl=unrealized,
            realized_pnl=self._realized_pnl,
            total_pnl=unrealized + self._realized_pnl,
        )
        self._last_refresh = datetime.now(timezone.utc)

    async def run_refresh_loop(self, get_market_map):
        """
        Continuously refresh positions in the background.
        `get_market_map` is a callable returning dict[str, Market].
        """
        while True:
            await self.refresh(get_market_map())
            await asyncio.sleep(self.refresh_interval_sec)

    def needs_refresh(self) -> bool:
        if self._last_refresh is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_refresh).total_seconds()
        return elapsed >= self.refresh_interval_sec

    @property
    def snapshot(self) -> Optional[PortfolioSnapshot]:
        return self._snapshot

    def get_position(self, market_id: str) -> Optional[Position]:
        if not self._snapshot:
            return None
        for p in self._snapshot.positions:
            if p.market_id == market_id:
                return p
        return None

    def summary(self) -> dict:
        if not self._snapshot:
            return {}
        s = self._snapshot
        return {
            "balance_usd": s.balance_usd,
            "open_positions": s.position_count,
            "unrealized_pnl": s.unrealized_pnl,
            "realized_pnl": s.realized_pnl,
            "total_pnl": s.total_pnl,
        }
