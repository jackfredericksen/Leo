"""
Leo — Coinbase Predictions Arbitrage Bot

Entry point. Runs the main scan/trade loop with a Rich terminal UI.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

from api_clients.coinbase_predictions import CoinbasePredictionsClient
from arbitrage import ArbitrageDetector, ArbOpportunity
from config import config
from storage import Storage
from trader import Trader

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("data/leo.log"), logging.StreamHandler()],
)
logger = logging.getLogger("leo.main")
console = Console()


def build_ui(opportunities: list[ArbOpportunity], stats: dict, scan_count: int) -> Panel:
    mode = "[bold red]LIVE[/]" if not config.dry_run else "[bold yellow]DRY RUN[/]"

    # Opportunities table
    opp_table = Table(
        "Market", "Type", "YES Ask", "NO Ask", "Sum", "Net Profit", "Resolves",
        box=box.SIMPLE_HEAD,
        title="Active Opportunities",
        title_style="bold cyan",
    )
    for opp in opportunities[:10]:
        total = (opp.leg_yes or 0) + (opp.leg_no or 0)
        hours = (opp.resolves_at - datetime.now(timezone.utc)).total_seconds() / 3600
        opp_table.add_row(
            opp.question[:40] + ("…" if len(opp.question) > 40 else ""),
            opp.arb_type,
            f"{opp.leg_yes:.3f}" if opp.leg_yes else "—",
            f"{opp.leg_no:.3f}" if opp.leg_no else "—",
            f"[green]{total:.3f}[/]",
            f"[bold green]{opp.net_profit_pct:.2%}[/]",
            f"{hours:.1f}h",
        )

    # Stats panel
    stats_table = Table(box=box.SIMPLE, show_header=False)
    stats_table.add_column(style="dim")
    stats_table.add_column()
    stats_table.add_row("Mode", mode)
    stats_table.add_row("Scans", str(scan_count))
    stats_table.add_row("Total Trades", str(stats.get("total_trades", 0)))
    stats_table.add_row("Deployed", f"${stats.get('total_deployed_usd') or 0:.2f}")
    stats_table.add_row("Est. P&L", f"${stats.get('estimated_pnl_usd') or 0:.2f}")

    return Panel(
        Columns([opp_table, stats_table]),
        title=f"[bold]Leo — Coinbase Predictions Arb[/bold]  {datetime.now().strftime('%H:%M:%S')}",
        border_style="cyan",
    )


async def run():
    storage = Storage(config.storage)
    detector = ArbitrageDetector(config.arbitrage)

    if not config.coinbase.api_key:
        console.print("[bold red]COINBASE_API_KEY not set — running in scan-only mode[/]")

    scan_count = 0
    current_opps: list[ArbOpportunity] = []

    async with CoinbasePredictionsClient(config.coinbase) as client:
        trader = Trader(config, client, storage)

        async def scan_loop():
            nonlocal scan_count, current_opps
            while True:
                try:
                    markets = await client.get_markets()
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
                live.update(build_ui(current_opps, stats, scan_count))
                await asyncio.sleep(1)

        with Live(console=console, refresh_per_second=1) as live:
            await asyncio.gather(scan_loop(), ui_loop(live))


def main():
    def _shutdown(sig, frame):
        console.print("\n[yellow]Shutting down Leo...[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    console.print("[bold cyan]Starting Leo — Coinbase Predictions Arbitrage Bot[/]")
    asyncio.run(run())


if __name__ == "__main__":
    main()
