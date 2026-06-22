"""
Leo — Polymarket Prediction Market Trading Bot

Entry point: logging setup, orchestration, and Rich terminal UI.
Strategy loops live in loops.py; terminal tables in terminal_ui.py.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.live import Live

import bot_state as _bot_state
from alerting import Alerter
from api_clients.binance_client import BinanceClient
from api_clients.llm_client import LLMClient
from api_clients.news_client import NewsClient
from api_clients.polymarket_client import PolymarketClient
from api_clients.pyth_client import PythClient
from arbitrage import ArbitrageDetector
from config import config
from loops import create_bot_tasks, ui_loop  # noqa: F401 — ui_loop used below
from position_manager import PositionManager
from storage import Storage
from strategies.btc_5min import BTC5MinDetector
from strategies.confluence import ConfluenceTracker
from strategies.hurst import HurstTracker
from strategies.kyle_lambda import KyleLambdaTracker
from trader import Trader
from runtime_config import load_runtime
from strategy_audit import audit_scheduled_strategies

from logging.handlers import RotatingFileHandler as _RotFileHandler

os.makedirs("data", exist_ok=True)

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_IGNORE_PREFIXES = ("uvicorn", "fastapi", "starlette", "asyncio",
                    "aiohttp", "httpx", "websockets", "h11", "hpack")


class _UILogHandler(logging.Handler):
    """Copies INFO+ records for Leo-owned loggers into the web UI feed buffer."""

    def emit(self, record):
        if record.name.startswith(_IGNORE_PREFIXES):
            return
        try:
            _bot_state._log_buffer.append({
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "name": record.name,
                "msg": record.getMessage(),
            })
        except Exception:
            pass


_main_handler = _RotFileHandler("data/leo.log", maxBytes=10 * 1024 * 1024, backupCount=5)
_error_handler = _RotFileHandler("data/errors.log", maxBytes=5 * 1024 * 1024, backupCount=3)
_error_handler.setLevel(logging.WARNING)
_ui_handler = _UILogHandler()
_ui_handler.setLevel(logging.INFO)

for _h in (_main_handler, _error_handler, _ui_handler):
    _h.setFormatter(logging.Formatter(_LOG_FMT))

logging.basicConfig(
    level=config.log_level,
    format=_LOG_FMT,
    handlers=[_main_handler, logging.StreamHandler(sys.stderr)],
)
logging.getLogger().addHandler(_error_handler)
logging.getLogger().addHandler(_ui_handler)

logger = logging.getLogger("leo.main")
console = Console()

_bot_state.config = config


async def run():
    storage = Storage(config.storage)
    _bot_state._storage_ref = storage
    _bot_state.config = config
    load_runtime(config)

    _resume_event = asyncio.Event()
    _resume_event.set()
    _bot_state._resume_event = _resume_event
    stop_event_main = asyncio.Event()
    _bot_state._stop_event = stop_event_main

    if not config.polymarket.private_key:
        console.print(
            "[bold yellow]POLY_PRIVATE_KEY not set — "
            "running in read-only / dry-run mode[/]"
        )

    alerter = Alerter(config.alerting.discord_webhook_url)
    _bot_state._alerter_ref = alerter
    news_client = NewsClient(config.news_fade.news_api_key)

    async with PolymarketClient(config.polymarket) as client:
        _bot_state._client_ref = client
        arb_detector = ArbitrageDetector(config.arbitrage)
        pos_manager = PositionManager(
            client, config.positions.refresh_interval_sec
        )

        confluence = ConfluenceTracker(config.confluence)
        _bot_state._confluence_ref = confluence
        hurst_tracker = HurstTracker()
        _bot_state._hurst_ref = hurst_tracker

        trader = Trader(config, client, storage, pos_manager, alerter, confluence)
        _bot_state._pos_manager_ref = pos_manager
        _bot_state._trader_ref = trader

        llm_for_evolution = None
        if config.llm.enabled and config.llm.api_key:
            llm_for_evolution = LLMClient(
                api_key=config.llm.api_key,
                model=config.llm.model,
                max_concurrent=1,
                cache_ttl_min=0,
            )

        async with BinanceClient() as binance, PythClient() as pyth:
            try:
                btc_bankroll = await trader.client.get_balance()
                if btc_bankroll <= 0:
                    btc_bankroll = config.risk.paper_bankroll
            except Exception:
                btc_bankroll = config.risk.paper_bankroll
            btc_detector = BTC5MinDetector(
                config.btc_5min, binance, pyth, btc_bankroll
            )

            tasks = create_bot_tasks(
                client=client,
                trader=trader,
                storage=storage,
                pos_manager=pos_manager,
                arb_detector=arb_detector,
                binance=binance,
                pyth=pyth,
                btc_detector=btc_detector,
                news_client=news_client,
                alerter=alerter,
                llm_for_evolution=llm_for_evolution,
            )
            task_names = {t.get_name() for t in tasks}
            _bot_state.strategy_audit = audit_scheduled_strategies(
                config, task_names
            )

            stop_event = asyncio.Event()

            def _handle_signal():
                console.print("\n[yellow]Shutting down Leo...[/yellow]")
                stop_event.set()
                stop_event_main.set()

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _handle_signal)

            ui_task = None
            try:
                with Live(console=console, refresh_per_second=1) as live:
                    ui_task = asyncio.create_task(
                        ui_loop(live, storage, pos_manager), name="ui"
                    )
                    await asyncio.wait(
                        [
                            asyncio.ensure_future(stop_event.wait()),
                            asyncio.ensure_future(stop_event_main.wait()),
                        ],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
            finally:
                for t in tasks:
                    t.cancel()
                if ui_task is not None:
                    ui_task.cancel()
                await asyncio.gather(
                    *tasks,
                    *([ui_task] if ui_task is not None else []),
                    return_exceptions=True,
                )
                console.print("[green]Leo stopped cleanly.[/]")


def main():
    console.print("[bold cyan]Starting Leo — Polymarket Trading Bot[/]")
    asyncio.run(run())


if __name__ == "__main__":
    main()