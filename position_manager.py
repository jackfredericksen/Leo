"""
Position manager — tracks open Polymarket positions and computes real-time P&L.

Polls the Polymarket CLOB /positions endpoint and /balance to maintain
a live view of:
  - All open positions (market, side, shares, avg entry price)
  - Unrealized P&L (current mid vs avg entry)
  - Realized P&L from settled positions
  - USDC account balance

Positions are refreshed every `refresh_interval_sec` seconds (default 30s).

P&L model:
  For a YES position of N shares entered at avg_price:
    unrealized_pnl = N × (current_mid − avg_price)

  On resolution:
    realized_pnl = N × (1.0 − avg_price)  if YES resolves
    realized_pnl = N × (0.0 − avg_price)  = −N × avg_price  if NO resolves
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from api_clients.polymarket_client import PolymarketClient, Market

logger = logging.getLogger(__name__)


@dataclass
class Position:
    market_id: str          # condition_id
    question: str
    side: str               # "yes" | "no"
    contracts: float        # shares held (float on Polymarket)
    avg_price: float        # average entry price (0–1)
    current_mid: float      # latest mid price
    unrealized_pnl: float
    close_time: datetime
    status: str             # "open" | "closed" | "settled"
    result: Optional[str]   # "yes" | "no" | None


@dataclass
class PortfolioSnapshot:
    balance_usd: float
    positions: list[Position]
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    taken_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status == "open"]

    @property
    def position_count(self) -> int:
        return len(self.open_positions)


class PositionManager:
    def __init__(
        self, client: PolymarketClient, refresh_interval_sec: int = 30
    ):
        self.client = client
        self.refresh_interval_sec = refresh_interval_sec
        self._snapshot: Optional[PortfolioSnapshot] = None
        self._last_refresh: Optional[datetime] = None
        self._realized_pnl: float = 0.0
        self._settled_ids: set[str] = set()  # market_ids already counted in realized

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def refresh(self, market_map: dict[str, Market]):
        """
        Fetch positions + balance from Polymarket and build a portfolio snapshot.
        `market_map` maps condition_id → Market with current prices.
        """
        try:
            balance     = await self.client.get_balance()
            raw_positions = await self.client.get_positions()
        except Exception as e:
            logger.error(f"PositionManager refresh failed: {e}")
            return

        positions  = []
        unrealized = 0.0

        # Build a reverse lookup: token_id → (market, side)
        token_to_market: dict[str, tuple[Market, str]] = {}
        for m in market_map.values():
            if m.yes_token_id:
                token_to_market[m.yes_token_id] = (m, "yes")
            if m.no_token_id:
                token_to_market[m.no_token_id] = (m, "no")

        for rp in raw_positions:
            # Data API returns: conditionId, asset (token_id), size, avgPrice, outcome
            # Old CLOB API returned: asset_id, conditionId, size, avgPrice
            token_id   = str(rp.get("asset", rp.get("asset_id", "")))
            condition_id = str(rp.get("conditionId", rp.get("condition_id", "")))
            shares     = float(rp.get("size", 0) or 0)
            avg_price  = float(rp.get("avgPrice", rp.get("avg_price", 0.5)) or 0.5)

            if shares <= 0:
                continue

            # Determine which market and side from the token_id
            if token_id in token_to_market:
                market, side = token_to_market[token_id]
                market_id   = market.market_id
                question    = market.question
                close_time  = market.close_time
                status      = market.status
                result      = market.result
                current_mid = (
                    (market.yes_bid + market.yes_ask) / 2
                    if side == "yes"
                    else (market.no_bid + market.no_ask) / 2
                )
            elif condition_id and condition_id in market_map:
                # Fallback: use condition_id + guess side from price
                market = market_map[condition_id]
                # Guess side: if avg_price closer to yes_price → YES
                side = (
                    "yes"
                    if abs(avg_price - market.yes_price) <= abs(avg_price - market.no_price)
                    else "no"
                )
                market_id   = condition_id
                question    = market.question
                close_time  = market.close_time
                status      = market.status
                result      = market.result
                current_mid = market.yes_price if side == "yes" else market.no_price
            else:
                market_id   = condition_id or token_id[:16]
                question    = market_id
                close_time  = datetime.now(timezone.utc)
                status      = "unknown"
                result      = None
                current_mid = avg_price
                side        = "yes"

            if status == "settled" and result:
                # Count realized P&L exactly once per settled position
                if market_id not in self._settled_ids:
                    payout = 1.0 if result == side else 0.0
                    realized = shares * (payout - avg_price)
                    self._realized_pnl += realized
                    self._settled_ids.add(market_id)
                    logger.info(
                        f"Settled {market_id}: side={side} result={result} "
                        f"realized=${realized:.2f}"
                    )
                # Settled positions have no unrealized P&L
                unreal = 0.0
            else:
                unreal = shares * (current_mid - avg_price)
                unrealized += unreal

            positions.append(Position(
                market_id=market_id,
                question=question,
                side=side,
                contracts=shares,
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

    def needs_refresh(self) -> bool:
        if self._last_refresh is None:
            return True
        elapsed = (
            datetime.now(timezone.utc) - self._last_refresh
        ).total_seconds()
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
