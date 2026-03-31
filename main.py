"""
Leo — Kalshi / Coinbase Predictions Arbitrage Bot

Entry point. Runs the main scan/trade loop with a Rich terminal UI.

Coinbase Predictions is powered by Kalshi (launched Jan 28, 2026,
all 50 US states). We connect to the Kalshi API directly.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box

from api_clients.kalshi_client import KalshiClient
from arbitrage import ArbitrageDetector, ArbOpportunity
from config import config
from storage import Storage
from trader import Trader

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("data/leo.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("leo.main")
console = Console()


def build_ui(
    opportunities: list[ArbOpportunity],
    stats: dict,
    scan_count: int,
) -> Panel:
    mode = (
        "[bold red]LIVE[/]"
        if not config.dry_run
        else "[bold yellow]DRY RUN[/]"
    )

    opp_table = Table(
        "Market", "YES bid", "NO bid", "Sum", "Net Profit", "Closes",
        box=box.SIMPLE_HEAD,
        title="Active Opportunities",
        title_style="bold cyan",
    )
    for opp in opportunities[:10]:
        total = opp.yes_bid + opp.no_bid
        hours = (
            (opp.close_time - datetime.now(timezone.utc)).total_seconds()
            / 3600
        )
        opp_table.add_row(
            opp.market_id[:35] + ("…" if len(opp.market_id) > 35 else ""),
            f"{opp.yes_bid:.3f}",
            f"{opp.no_bid:.3f}",
            f"[green]{total:.4f}[/]",
            f"[bold green]{opp.net_profit_pct:.2%}[/]",
            f"{hours:.1f}h",
        )

    stats_table = Table(box=box.SIMPLE, show_header=False)
    stats_table.add_column(style="dim")
    stats_table.add_column()
    stats_table.add_row("Mode", mode)
    stats_table.add_row("Scans", str(scan_count))
    stats_table.add_row("Total Trades", str(stats.get("total_trades", 0)))
    stats_table.add_row(
        "Contracts", str(stats.get("total_contracts") or 0)
    )
    stats_table.add_row(
        "Est. P&L", f"${stats.get('estimated_pnl_usd') or 0:.2f}"
    )

    return Panel(
        Columns([opp_table, stats_table]),
        title=(
            "[bold]Leo — Kalshi / Coinbase Predictions Arb[/bold]  "
            + datetime.now().strftime("%H:%M:%S")
        ),
        border_style="cyan",
    )


async def run():
    storage = Storage(config.storage)
    detector = ArbitrageDetector(config.arbitrage)

    if not config.kalshi.key_id:
        console.print(
            "[bold yellow]KALSHI_KEY_ID not set — "
            "running in scan-only / dry-run mode[/]"
        )

    scan_count = 0
    current_opps: list[ArbOpportunity] = []

    async with KalshiClient(config.kalshi) as client:
        trader = Trader(config, client, storage)

        async def scan_loop():
            nonlocal scan_count, current_opps
            while True:
                try:
                    markets = await client.get_all_markets()
                    opps = detector.scan(markets)
                    current_opps = opps
                    scan_count += 1

                    for opp in opps:
                        storage.log_opportunity(
                            market_id=opp.market_id,
                            question=opp.question,
                            arb_type=opp.arb_type,
                            gross_pct=opp.gross_profit_pct,
                            net_pct=opp.net_profit_pct,
                            acted_on=False,
                            detected_at=opp.detected_at,
                        )
                        await trader.execute(opp)

                except Exception as e:
                    logger.error(f"Scan error: {e}")

                await asyncio.sleep(config.arbitrage.poll_interval_sec)

        async def ui_loop(live: Live):
            while True:
                stats = storage.get_pnl_summary()
                live.update(
                    build_ui(current_opps, stats, scan_count)
                )
                await asyncio.sleep(1)

        with Live(console=console, refresh_per_second=1) as live:
            await asyncio.gather(scan_loop(), ui_loop(live))


def main():
    def _shutdown(sig, frame):
        console.print("\n[yellow]Shutting down Leo...[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    console.print(
        "[bold cyan]Starting Leo — "
        "Kalshi / Coinbase Predictions Arbitrage Bot[/]"
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()
