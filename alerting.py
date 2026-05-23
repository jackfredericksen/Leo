"""Discord webhook alerting for Leo."""

import logging
from datetime import datetime, timezone

import aiohttp

logger = logging.getLogger(__name__)


class Alerter:
    def __init__(self, webhook_url: str):
        self._url = webhook_url

    async def send(self, message: str) -> None:
        if not self._url:
            return
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    self._url,
                    json={"content": message},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
        except Exception as e:
            logger.debug(f"Discord alert failed: {e}")

    async def large_fill(
        self, market_id: str, side: str, size_usd: float, source: str
    ) -> None:
        await self.send(
            f"🟢 **Leo Fill** `{source}` {side.upper()} **${size_usd:.0f}** "
            f"on `{market_id[:16]}`"
        )

    async def daily_limit_hit(self, total: float, limit: float) -> None:
        await self.send(
            f"🔴 **Daily deployment limit hit**: ${total:.0f} / ${limit:.0f}"
        )

    async def circuit_breaker_open(self, error_count: int, pause_sec: int) -> None:
        await self.send(
            f"⚠️ **CLOB circuit breaker OPEN** after {error_count} errors "
            f"— pausing {pause_sec}s"
        )

    async def big_edge(
        self,
        question: str,
        edge_pct: float,
        strategy: str,
        side: str,
        size_usd: float,
    ) -> None:
        sign = "+" if edge_pct >= 0 else ""
        await self.send(
            f"🎯 **High-edge signal** `{strategy}` → {side.upper()} "
            f"**{sign}{edge_pct:.1f}%** | ${size_usd:.0f} deployed\n"
            f"_{question[:80]}_"
        )

    async def outcome_resolved(
        self,
        trade_id: int,
        question: str,
        outcome: str,
        strategy: str,
    ) -> None:
        icon = "✅" if outcome == "won" else "❌"
        await self.send(
            f"{icon} **Trade #{trade_id} resolved** `{strategy}` → **{outcome.upper()}**\n"
            f"_{question[:80]}_"
        )

    async def daily_summary(
        self,
        balance: float,
        total_pnl: float,
        unrealized: float,
        top_strategies: list[dict],
    ) -> None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p_sign = "+" if total_pnl >= 0 else ""
        u_sign = "+" if unrealized >= 0 else ""
        lines = [
            f"📊 **Leo Daily Summary — {date_str}**",
            f"Balance: **${balance:.2f}** | P&L: **{p_sign}${total_pnl:.2f}**"
            f" | Unrealized: **{u_sign}${unrealized:.2f}**",
            "",
            "**Top Strategies:**",
        ]
        for s in top_strategies[:5]:
            avg_edge = (s.get("avg_edge") or 0) * 100
            est_pnl = s.get("est_pnl") or 0
            lines.append(
                f"  • `{s['arb_type']}`: {s['trades']} trades,"
                f" avg edge {avg_edge:.1f}%, est P&L ${est_pnl:.2f}"
            )
        await self.send("\n".join(lines))
