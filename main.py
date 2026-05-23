"""
Leo — Polymarket Prediction Market Trading Bot

Strategies (all execute on Polymarket):
  1.  Overround arb          — YES_ask + NO_ask < $1.00 (buy both cheap)
  2.  Crypto price signal    — Binance spot + log-normal model vs Polymarket price
  3.  Range straddle arb     — BTC/ETH/SOL range bracket pricing
  4.  Correlation / logical arb — mathematically impossible pricing
  5.  News overreaction fade — fade spikes 1.5-4h after a move
  6.  Forecast aggregation   — Metaculus + Manifold as signal oracle
  7.  LLM fundamental analysis — Claude Haiku qualitative assessment
  8.  Weather signal         — Open-Meteo model vs Polymarket weather markets
  9.  Cross-platform signal  — Kalshi public prices as signal for Polymarket
  10. Favorite mispricing    — short overpriced favorites (favorite-longshot bias)
  11. Oracle squeeze         — buy near-certain outcomes past their close_time
  12. Semantic arb           — exact cross-platform match + price gap
  13. OFI momentum           — order book imbalance directional signal
  14. Market making          — resting limit orders + CLOB v2 maker rebates
  +   Position tracking      — live P&L
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timedelta, timezone

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from alerting import Alerter
from api_clients.binance_client import BinanceClient
from api_clients.forecast_client import ForecastClient
from api_clients.news_client import NewsClient
from api_clients.polymarket_client import PolymarketClient, Market
from api_clients.llm_client import LLMClient
from api_clients.weather_client import WeatherClient
from strategies.confluence import ConfluenceConfig, ConfluenceTracker
from strategies.exit_strategy import ExitConfig, ExitStrategy
from strategies.evolution import EvolutionConfig, EvolutionAgent
from strategies.hurst import HurstTracker
from strategies.kyle_lambda import KyleLambdaTracker
from arbitrage import ArbitrageDetector, ArbOpportunity
from strategies.cross_platform import (
    CrossPlatformConfig,
    CrossPlatformDetector,
    CrossPlatformOpportunity,
    KalshiSignalClient,
)
from config import config
from position_manager import PositionManager
from storage import Storage
from strategies.correlated import CorrelatedOpportunity
from strategies.crypto_signal import CryptoSignalDetector
from strategies.forecast_signal import ForecastSignalDetector
from strategies.llm_signal import LLMSignalDetector
from strategies.news_fade import NewsFadeDetector, NewsFadeConfig
from strategies.range_straddle import RangeStraddleDetector
from strategies.signal_arb import AggregatedSignal, SignalArbConfig
from strategies.weather_signal import WeatherSignalDetector
from strategies.favorite_short import FavoriteShortDetector, FavoriteShortConfig
from strategies.oracle_squeeze import OracleSqueezeDetector, OracleSqueezeConfig
from strategies.semantic_arb import SemanticArbDetector, SemanticArbConfig
from strategies.orderbook_momentum import OrderbookMomentumDetector, OrderbookMomentumConfig
from strategies.market_maker import MarketMakerStrategy, MarketMakerConfig
from strategies.btc_5min import BTC5MinConfig, BTC5MinDetector
from trader import Trader

from logging.handlers import RotatingFileHandler as _RotFileHandler

os.makedirs("data", exist_ok=True)

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LEO_PREFIXES = ("leo", "arbitrage", "strategies", "trader", "position",
                 "storage", "alerting", "auto_correlator", "api_clients",
                 "web_gui", "__main__")
_IGNORE_PREFIXES = ("uvicorn", "fastapi", "starlette", "asyncio",
                    "aiohttp", "httpx", "websockets", "h11", "hpack")


class _UILogHandler(logging.Handler):
    """Copies INFO+ records for Leo-owned loggers into the web UI feed buffer."""
    def emit(self, record):
        if record.name.startswith(_IGNORE_PREFIXES):
            return
        try:
            import bot_state as _bs
            _bs._log_buffer.append({
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "name": record.name,
                "msg": record.getMessage(),
            })
        except Exception:
            pass


_main_handler  = _RotFileHandler("data/leo.log",   maxBytes=10*1024*1024, backupCount=5)
_error_handler = _RotFileHandler("data/errors.log", maxBytes=5*1024*1024,  backupCount=3)
_error_handler.setLevel(logging.WARNING)
_ui_handler    = _UILogHandler()
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


# ---------------------------------------------------------------------------
# Shared bot state
# ---------------------------------------------------------------------------

class BotState:
    def __init__(self):
        self.markets: list[Market] = []
        self.market_map: dict[str, Market] = {}

        self.extended_market_map: dict[str, Market] = {}  # market_map + recent BTC 5-min markets

        self.arb_opps: list[ArbOpportunity] = []
        self.crypto_opps: list[AggregatedSignal] = []
        self.cross_opps: list[CrossPlatformOpportunity] = []
        self.range_opps: list[AggregatedSignal] = []
        self.corr_opps: list[CorrelatedOpportunity] = []
        self.fade_opps: list[AggregatedSignal] = []
        self.forecast_opps: list[AggregatedSignal] = []
        self.llm_opps: list[AggregatedSignal] = []
        self.weather_opps: list[AggregatedSignal] = []

        self.fav_opps: list[AggregatedSignal] = []
        self.squeeze_opps: list[AggregatedSignal] = []
        self.semarg_opps: list[AggregatedSignal] = []
        self.ofi_opps: list[AggregatedSignal] = []
        self.btc5min_opps: list[AggregatedSignal] = []
        self.mm_quotes: list[dict] = []

        self.arb_scans = 0
        self.crypto_scans = 0
        self.cross_scans = 0
        self.range_scans = 0
        self.corr_scans = 0
        self.fade_scans = 0
        self.forecast_scans = 0
        self.llm_scans = 0
        self.weather_scans = 0
        self.fav_scans = 0
        self.squeeze_scans = 0
        self.semarg_scans = 0
        self.ofi_scans = 0
        self.btc5min_scans = 0
        self.mm_active = 0

        # Web-gui compat aliases
        self.signal_opps: list[AggregatedSignal] = []
        self.signal_scans = 0
        self.mm_scans = 0
        self.whale_signals = 0

        # Per-strategy timestamp of last non-empty scan result
        self.last_signal_at: dict[str, str] = {}


state = BotState()

import bot_state as _bot_state  # noqa: E402
_bot_state.state = state
_bot_state.config = config


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _mode() -> str:
    return (
        "[bold red]LIVE[/]"
        if not config.dry_run
        else "[bold yellow]DRY RUN[/]"
    )


def _hrs(close_time: datetime) -> str:
    h = (close_time - datetime.now(timezone.utc)).total_seconds() / 3600
    return f"{h:.1f}h"


def _arb_table() -> Table:
    t = Table(
        "Market", "Sum", "Net%", "Closes",
        box=box.SIMPLE_HEAD, title="[cyan]Overround Arb[/]",
        title_style="bold", min_width=50,
    )
    for o in state.arb_opps[:5]:
        q = o.question[:26] + ("…" if len(o.question) > 26 else "")
        t.add_row(
            q,
            f"[green]{o.yes_ask + o.no_ask:.4f}[/]",
            f"[bold green]{o.net_profit_pct:.1%}[/]",
            _hrs(o.close_time),
        )
    if not state.arb_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _crypto_table() -> Table:
    t = Table(
        "Market", "Spot→Model", "Edge", "Side",
        box=box.SIMPLE_HEAD, title="[yellow]Crypto Signal[/]",
        title_style="bold", min_width=54,
    )
    for o in state.crypto_opps[:5]:
        clr = "green" if o.edge > 0 else "red"
        q = o.question[:24] + ("…" if len(o.question) > 24 else "")
        t.add_row(
            q,
            f"{o.market_prob:.0%}→{o.model_prob:.0%}",
            f"[{clr}]{o.edge:+.1%}[/]",
            o.recommended_side,
        )
    if not state.crypto_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _cross_table() -> Table:
    t = Table(
        "Market", "Buy@", "Ref@", "Net%",
        box=box.SIMPLE_HEAD, title="[blue]Cross-Platform (Kalshi→Poly)[/]",
        title_style="bold", min_width=54,
    )
    for o in state.cross_opps[:5]:
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        t.add_row(
            q,
            f"PM {o.buy_price:.3f}",
            f"KL {o.sell_price:.3f}",
            f"[bold blue]{o.net_profit_pct:.1%}[/]",
        )
    if not state.cross_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _range_table() -> Table:
    t = Table(
        "Market", "Model", "Mkt", "Edge",
        box=box.SIMPLE_HEAD, title="[magenta]Range/Logic Arb[/]",
        title_style="bold", min_width=50,
    )
    for o in (state.range_opps + state.corr_opps)[:5]:
        if isinstance(o, AggregatedSignal):
            clr = "green" if o.edge > 0 else "red"
            q = o.question[:24] + ("…" if len(o.question) > 24 else "")
            t.add_row(q, f"{o.model_prob:.0%}", f"{o.market_prob:.0%}", f"[{clr}]{o.edge:+.1%}[/]")
        else:
            q = o.question_mispriced[:24] + ("…" if len(o.question_mispriced) > 24 else "")
            t.add_row(q, o.relation.value, f"{o.edge:.1%}", o.recommended_side)
    if not state.range_opps and not state.corr_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _fade_table() -> Table:
    t = Table(
        "Market", "Dir", "Edge",
        box=box.SIMPLE_HEAD, title="[orange3]News Fade[/]",
        title_style="bold", min_width=46,
    )
    for o in state.fade_opps[:5]:
        clr = "green" if o.edge > 0 else "red"
        q = o.question[:30] + ("…" if len(o.question) > 30 else "")
        t.add_row(q, o.recommended_side, f"[{clr}]{o.edge:+.1%}[/]")
    if not state.fade_opps:
        t.add_row("[dim]none[/]", "", "")
    return t


def _forecast_table() -> Table:
    t = Table(
        "Market", "Src", "Model", "Mkt", "Edge",
        box=box.SIMPLE_HEAD, title="[bright_cyan]Forecast Aggregator[/]",
        title_style="bold", min_width=58,
    )
    for o in state.forecast_opps[:5]:
        clr = "green" if o.edge > 0 else "red"
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        t.add_row(q, (o.source or "")[:10], f"{o.model_prob:.0%}", f"{o.market_prob:.0%}", f"[{clr}]{o.edge:+.1%}[/]")
    if not state.forecast_opps:
        t.add_row("[dim]none[/]", "", "", "", "")
    return t


def _llm_table() -> Table:
    t = Table(
        "Market", "Model", "Mkt", "Edge", "Conf",
        box=box.SIMPLE_HEAD, title="[bright_green]LLM Analysis[/]",
        title_style="bold", min_width=58,
    )
    for o in state.llm_opps[:5]:
        clr = "green" if o.edge > 0 else "red"
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        t.add_row(q, f"{o.model_prob:.0%}", f"{o.market_prob:.0%}", f"[{clr}]{o.edge:+.1%}[/]", f"{o.confidence:.0%}" if o.confidence else "—")
    if not state.llm_opps:
        t.add_row("[dim]none[/]", "", "", "", "")
    return t


def _weather_table() -> Table:
    t = Table(
        "Market", "Forecast", "Mkt", "Edge",
        box=box.SIMPLE_HEAD, title="[bright_blue]Weather Signal[/]",
        title_style="bold", min_width=54,
    )
    for o in state.weather_opps[:5]:
        clr = "green" if o.edge > 0 else "red"
        q = o.question[:24] + ("…" if len(o.question) > 24 else "")
        t.add_row(q, f"{o.model_prob:.0%}", f"{o.market_prob:.0%}", f"[{clr}]{o.edge:+.1%}[/]")
    if not state.weather_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _fav_table() -> Table:
    t = Table(
        "Market", "Mkt", "Fair", "Edge",
        box=box.SIMPLE_HEAD, title="[bold red]Favorite Short[/]",
        title_style="bold", min_width=52,
    )
    for o in state.fav_opps[:5]:
        q = o.question[:24] + ("…" if len(o.question) > 24 else "")
        t.add_row(q, f"{o.market_prob:.0%}", f"{o.model_prob:.0%}", f"[green]{o.edge:+.1%}[/]")
    if not state.fav_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _squeeze_table() -> Table:
    t = Table(
        "Market", "Price", "Target", "Edge",
        box=box.SIMPLE_HEAD, title="[bold yellow]Oracle Squeeze[/]",
        title_style="bold", min_width=52,
    )
    for o in state.squeeze_opps[:5]:
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        tgt = "1.00" if o.recommended_side == "yes" else "0.00"
        t.add_row(q, f"{o.market_prob:.2f}", tgt, f"[bold green]{o.edge:+.1%}[/]")
    if not state.squeeze_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _semarg_table() -> Table:
    t = Table(
        "Market", "Poly", "Ext", "Gap",
        box=box.SIMPLE_HEAD, title="[bold blue]Semantic Arb[/]",
        title_style="bold", min_width=52,
    )
    for o in state.semarg_opps[:5]:
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        t.add_row(q, f"{o.market_prob:.0%}", f"{o.model_prob:.0%}", f"[bold blue]{o.edge:+.1%}[/]")
    if not state.semarg_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _ofi_table() -> Table:
    t = Table(
        "Market", "OFI", "Regime", "Edge",
        box=box.SIMPLE_HEAD, title="[bold magenta]OFI Momentum[/]",
        title_style="bold", min_width=52,
    )
    for o in state.ofi_opps[:5]:
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        regime = "follow" if "follow" in o.source else "fade"
        clr = "green" if o.edge > 0 else "red"
        t.add_row(q, o.recommended_side, regime, f"[{clr}]{o.edge:+.1%}[/]")
    if not state.ofi_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _mm_table() -> Table:
    t = Table(
        "Market", "YES@", "NO@", "Mid",
        box=box.SIMPLE_HEAD, title=f"[bold cyan]Market Maker ({state.mm_active} active)[/]",
        title_style="bold", min_width=52,
    )
    for q in state.mm_quotes[:5]:
        mkt = q.get("question", "")[:22]
        yes_b = f"{q['yes_bid']:.2f}" if q.get("yes_bid") else "—"
        no_b  = f"{q['no_bid']:.2f}"  if q.get("no_bid")  else "—"
        mid   = f"{q.get('fair_value', 0):.2f}"
        t.add_row(mkt, yes_b, no_b, mid)
    if not state.mm_quotes:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _stats_table(storage: Storage, pos_manager: PositionManager) -> Table:
    trade_stats = storage.get_pnl_summary()
    pos = pos_manager.summary()
    t = Table(box=box.SIMPLE, show_header=False, min_width=26)
    t.add_column(style="dim", no_wrap=True)
    t.add_column(justify="right")
    t.add_row("Mode", _mode())
    t.add_row("Balance (USDC)", f"${pos.get('balance_usd', 0):.2f}")
    t.add_row("Positions", str(pos.get("open_positions", 0)))
    t.add_row("Unrealized", f"${pos.get('unrealized_pnl', 0):.2f}")
    t.add_row("Realized", f"${pos.get('realized_pnl', 0):.2f}")
    t.add_row("─" * 10, "─" * 7)
    t.add_row("Markets", str(len(state.markets)))
    t.add_row("Arb scans", str(state.arb_scans))
    t.add_row("Corr scans", str(state.corr_scans))
    t.add_row("Crypto scans", str(state.crypto_scans))
    t.add_row("Cross scans", str(state.cross_scans))
    t.add_row("Range scans", str(state.range_scans))
    t.add_row("Fade scans", str(state.fade_scans))
    t.add_row("Forecast scans", str(state.forecast_scans))
    t.add_row("LLM scans", str(state.llm_scans))
    t.add_row("Weather scans", str(state.weather_scans))
    t.add_row("Fav scans", str(state.fav_scans))
    t.add_row("Squeeze scans", str(state.squeeze_scans))
    t.add_row("Sem arb scans", str(state.semarg_scans))
    t.add_row("OFI scans", str(state.ofi_scans))
    t.add_row("BTC 5min scans", str(state.btc5min_scans))
    t.add_row("MM active", str(state.mm_active))
    t.add_row("─" * 10, "─" * 7)
    t.add_row("Trades", str(trade_stats.get("total_trades", 0)))
    t.add_row("Est. P&L", f"${trade_stats.get('estimated_pnl_usd') or 0:.2f}")
    return t


def build_ui(storage: Storage, pos_manager: PositionManager) -> Panel:
    row1 = Columns([_arb_table(), _crypto_table()], equal=True)
    row2 = Columns([_cross_table(), _range_table()], equal=True)
    row3 = Columns([_fade_table(), _forecast_table()], equal=True)
    row4 = Columns([_llm_table(), _weather_table()], equal=True)
    row5 = Columns([_fav_table(), _squeeze_table()], equal=True)
    row6 = Columns([_semarg_table(), _ofi_table()], equal=True)
    row7 = Columns([_mm_table(), _stats_table(storage, pos_manager)], equal=True)
    return Panel(
        Group(row1, row2, row3, row4, row5, row6, row7),
        title=(
            "[bold]Leo — Polymarket Trading Bot[/bold]  "
            + datetime.now().strftime("%H:%M:%S")
        ),
        border_style="cyan",
    )


# ---------------------------------------------------------------------------
# Strategy loops
# ---------------------------------------------------------------------------

_BANKROLL_REFRESH_EVERY = 10


async def _await_resume() -> None:
    """Block while the bot is paused. Zero overhead when running (event is set)."""
    ev = _bot_state._resume_event
    if ev is not None:
        await ev.wait()


def _alert_task_standalone(coro) -> None:
    """Fire-and-forget an alerter coroutine from outside the Trader class."""
    if coro is not None:
        asyncio.create_task(coro)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _refresh_bankroll(trader: Trader, sizer) -> None:
    try:
        bal = await trader.client.get_balance()
        sizer.update_bankroll(bal)
    except Exception:
        pass


async def market_refresh_loop(client: PolymarketClient):
    """Dedicated loop: fetch all Polymarket markets and update shared state."""
    while True:
        try:
            markets = await client.get_all_markets()
            state.markets = markets
            state.market_map = {m.market_id: m for m in markets}
            _bot_state._last_gamma_at = datetime.now(timezone.utc).isoformat()
            logger.debug(f"Market refresh: {len(markets)} markets loaded")
        except Exception as e:
            logger.error(f"Market refresh: {e}")
        finally:
            _bot_state._force_market_refresh = False
        # Sleep 30s but wake early if force-refresh is requested
        for _ in range(30):
            await asyncio.sleep(1)
            if _bot_state._force_market_refresh:
                break


async def arb_loop(detector: ArbitrageDetector, trader: Trader, client: PolymarketClient):
    """Strategy: Overround arb — buy YES + NO < $1.00 using live CLOB orderbooks."""
    while True:
        await _await_resume()
        if not config.arbitrage.enabled:
            await asyncio.sleep(5)
            continue
        try:
            if state.markets:
                if config.arbitrage.live_candidates > 0:
                    state.arb_opps = await detector.scan_live(
                        state.markets, client, config.arbitrage.live_candidates
                    )
                else:
                    state.arb_opps = detector.scan(state.markets)
                state.arb_scans += 1
                if state.arb_opps:
                    state.last_signal_at["arb"] = _now_iso()
                for opp in state.arb_opps:
                    await trader.execute(opp)
        except Exception as e:
            logger.error(f"Arb loop: {e}")
        await asyncio.sleep(config.arbitrage.poll_interval_sec)


async def crypto_signal_loop(binance: BinanceClient, trader: Trader):
    """Strategy: Real-time Binance price signal → Polymarket crypto markets."""
    sig_cfg = SignalArbConfig(
        min_edge=config.crypto_signal.min_edge,
        fee_pct=config.arbitrage.fee_pct,
        max_position_usd=config.crypto_signal.max_position_usd,
        kelly_fraction=config.crypto_signal.kelly_fraction,
    )
    try:
        bankroll = await trader.client.get_balance()
        if bankroll <= 0:
            bankroll = config.risk.paper_bankroll
    except Exception:
        bankroll = config.risk.paper_bankroll

    detector = CryptoSignalDetector(sig_cfg, binance, bankroll)
    while True:
        await _await_resume()
        if not config.crypto_signal.enabled:
            await asyncio.sleep(5)
            continue
        try:
            await binance.refresh_all()
            if state.markets:
                state.crypto_opps = detector.scan(state.markets)
                state.signal_opps = state.crypto_opps
                state.crypto_scans += 1
                state.signal_scans = state.crypto_scans
                if state.crypto_opps:
                    state.last_signal_at["crypto"] = _now_iso()
                if state.crypto_scans % _BANKROLL_REFRESH_EVERY == 0:
                    await _refresh_bankroll(trader, detector.sizer)
                for sig in state.crypto_opps:
                    await trader.execute_signal(sig, state.market_map)
                if state.crypto_opps:
                    logger.info(
                        f"Crypto signal: {len(state.crypto_opps)} opps, "
                        f"top edge={state.crypto_opps[0].edge:.1%}"
                    )
        except Exception as e:
            logger.error(f"Crypto signal loop: {e}")
        await asyncio.sleep(config.crypto_signal.refresh_interval_sec)


async def cross_platform_loop(trader: Trader):
    """Strategy: Kalshi public prices as signal → trade on Polymarket."""
    cross_cfg = CrossPlatformConfig(
        min_profit_pct=config.cross_arb.min_profit_pct,
        max_position_usd=config.cross_arb.max_position_usd,
    )
    detector = CrossPlatformDetector(cross_cfg)

    async with KalshiSignalClient() as kalshi_signal:
        while True:
            await _await_resume()
            if not config.cross_arb.enabled:
                await asyncio.sleep(5)
                continue
            try:
                if state.markets:
                    ext_markets = await kalshi_signal.get_external_markets(max_pages=5)
                    detector.load_external_markets(ext_markets)
                    state.cross_opps = detector.scan(state.markets)
                    state.cross_scans += 1
                    if state.cross_opps:
                        state.last_signal_at["cross_arb"] = _now_iso()
                    for opp in state.cross_opps:
                        # Determine which side to buy on Polymarket
                        # buy_price is the YES or NO ask — detect from context
                        market = state.market_map.get(opp.polymarket_market_id)
                        if not market:
                            continue
                        side = (
                            "yes"
                            if abs(opp.buy_price - market.yes_ask) < abs(opp.buy_price - market.no_ask)
                            else "no"
                        )
                        sig = AggregatedSignal(
                            market_id=opp.polymarket_market_id,
                            question=opp.question,
                            market_prob=opp.buy_price,
                            model_prob=opp.sell_price,
                            edge=opp.net_profit_pct,
                            recommended_side=side,
                            source=f"cross:{opp.sell_platform}",
                            recommended_size_usd=opp.max_size_usd,
                        )
                        await trader.execute_signal(sig, state.market_map)
                    if state.cross_opps:
                        logger.info(f"Cross-platform: {len(state.cross_opps)} opps")
            except Exception as e:
                logger.error(f"Cross-platform loop: {e}")
            await asyncio.sleep(config.cross_arb.refresh_interval_sec)


async def range_straddle_loop(binance: BinanceClient, trader: Trader):
    """Strategy: Range straddle + logical arb on Polymarket crypto range markets."""
    sig_cfg = SignalArbConfig(
        min_edge=config.range_straddle.min_edge,
        fee_pct=config.arbitrage.fee_pct,
        max_position_usd=config.range_straddle.max_position_usd,
        kelly_fraction=config.range_straddle.kelly_fraction,
    )
    try:
        bankroll = await trader.client.get_balance()
        if bankroll <= 0:
            bankroll = config.risk.paper_bankroll
    except Exception:
        bankroll = config.risk.paper_bankroll

    detector = RangeStraddleDetector(sig_cfg, binance, bankroll)
    while True:
        await _await_resume()
        if not config.range_straddle.enabled:
            await asyncio.sleep(5)
            continue
        try:
            await binance.fetch_snapshots()
            if state.markets:
                state.range_opps = detector.scan(state.markets)
                state.range_scans += 1
                if state.range_opps:
                    state.last_signal_at["range_straddle"] = _now_iso()
                if state.range_scans % _BANKROLL_REFRESH_EVERY == 0:
                    await _refresh_bankroll(trader, detector.sizer)
                for sig in state.range_opps:
                    await trader.execute_signal(sig, state.market_map)
        except Exception as e:
            logger.error(f"Range straddle loop: {e}")
        await asyncio.sleep(60)


async def corr_loop(trader: Trader):
    """Strategy: Logical/correlation arb."""
    from strategies.correlated import CorrelatedConfig
    from auto_correlator import AutoCorrelatorConfig, build_correlated_detector

    corr_cfg = CorrelatedConfig(
        min_edge=config.correlated.min_edge,
        fee_pct=config.correlated.fee_pct,
        max_position_usd=config.correlated.max_position_usd,
        min_confidence=config.correlated.min_confidence,
    )
    auto_cfg = AutoCorrelatorConfig(
        relation_confidence=config.correlated.min_confidence,
        min_liquidity_usd=config.correlated.min_liquidity_usd,
    )

    detector = None
    last_market_count = 0

    while True:
        await _await_resume()
        if not config.correlated.enabled:
            await asyncio.sleep(5)
            continue
        try:
            if state.markets:
                if detector is None or len(state.markets) != last_market_count:
                    detector = build_correlated_detector(
                        state.markets, corr_cfg, auto_cfg
                    )
                    last_market_count = len(state.markets)

                state.corr_opps = detector.scan(state.markets)
                state.corr_scans += 1
                if state.corr_opps:
                    state.last_signal_at["correlated"] = _now_iso()
                for opp in state.corr_opps:
                    await trader.execute_correlated(opp, state.market_map)
        except Exception as e:
            logger.error(f"Corr loop: {e}")
        await asyncio.sleep(config.correlated.poll_interval_sec)


async def news_fade_loop(trader: Trader, news_client: NewsClient):
    """Strategy: News overreaction fade with optional NewsAPI confirmation."""
    sig_cfg = SignalArbConfig(
        min_edge=config.news_fade.min_edge,
        fee_pct=config.arbitrage.fee_pct,
        max_position_usd=config.news_fade.max_position_usd,
        kelly_fraction=config.news_fade.kelly_fraction,
    )
    fade_cfg = NewsFadeConfig(
        min_spike_pct=config.news_fade.min_spike_pct,
        min_hours_old=config.news_fade.min_hours_old,
        max_hours_old=config.news_fade.max_hours_old,
        fade_fraction=config.news_fade.fade_fraction,
        min_liquidity_usd=config.news_fade.min_liquidity_usd,
        min_edge=config.news_fade.min_edge,
    )
    try:
        bankroll = await trader.client.get_balance()
        if bankroll <= 0:
            bankroll = config.risk.paper_bankroll
    except Exception:
        bankroll = config.risk.paper_bankroll

    detector = NewsFadeDetector(sig_cfg, fade_cfg, bankroll)
    while True:
        await _await_resume()
        if not config.news_fade.enabled:
            await asyncio.sleep(5)
            continue
        try:
            if state.markets:
                state.fade_opps = detector.scan(state.markets)
                state.fade_scans += 1
                if state.fade_opps:
                    state.last_signal_at["news_fade"] = _now_iso()
                if state.fade_scans % _BANKROLL_REFRESH_EVERY == 0:
                    await _refresh_bankroll(trader, detector.sizer)
                for sig in state.fade_opps:
                    if config.news_fade.require_news_confirmation:
                        confirmed = await news_client.has_recent_news(
                            sig.question,
                            hours=config.news_fade.max_hours_old,
                        )
                        if not confirmed:
                            logger.debug(
                                f"NewsFade: no recent news for {sig.market_id[:12]}, skipping"
                            )
                            continue
                    await trader.execute_signal(sig, state.market_map)
        except Exception as e:
            logger.error(f"News fade loop: {e}")
        await asyncio.sleep(300)


async def forecast_loop(trader: Trader):
    """Strategy: Community forecast aggregation (Metaculus + Manifold)."""
    sig_cfg = SignalArbConfig(
        min_edge=config.forecast.min_edge,
        fee_pct=config.arbitrage.fee_pct,
        max_position_usd=config.forecast.max_position_usd,
        kelly_fraction=config.forecast.kelly_fraction,
    )
    try:
        bankroll = await trader.client.get_balance()
        if bankroll <= 0:
            bankroll = config.risk.paper_bankroll
    except Exception:
        bankroll = config.risk.paper_bankroll

    fc = ForecastClient()
    detector = ForecastSignalDetector(sig_cfg, fc, bankroll)
    async with fc:
        while True:
            await _await_resume()
            if not config.forecast.enabled:
                await asyncio.sleep(5)
                continue
            try:
                await fc.refresh()
                if state.markets:
                    state.forecast_opps = detector.scan(state.markets)
                    state.forecast_scans += 1
                    if state.forecast_opps:
                        state.last_signal_at["forecast"] = _now_iso()
                    if state.forecast_scans % _BANKROLL_REFRESH_EVERY == 0:
                        await _refresh_bankroll(trader, detector.sizer)
                    for sig in state.forecast_opps:
                        await trader.execute_signal(sig, state.market_map)
                if state.forecast_opps:
                    logger.info(
                        f"Forecast: {len(state.forecast_opps)} opps, "
                        f"top edge={state.forecast_opps[0].edge:.1%}"
                    )
            except Exception as e:
                logger.error(f"Forecast loop: {e}")
            await asyncio.sleep(config.forecast.refresh_interval_sec)


async def llm_loop(trader: Trader, binance: BinanceClient):
    """Strategy: LLM fundamental analysis (Claude Haiku)."""
    sig_cfg = SignalArbConfig(
        min_edge=config.llm.min_edge,
        fee_pct=config.arbitrage.fee_pct,
        max_position_usd=config.llm.max_position_usd,
        kelly_fraction=config.llm.kelly_fraction,
    )
    try:
        bankroll = await trader.client.get_balance()
        if bankroll <= 0:
            bankroll = config.risk.paper_bankroll
    except Exception:
        bankroll = config.risk.paper_bankroll

    fc = ForecastClient() if config.forecast.enabled else None
    llm = LLMClient(
        api_key=config.llm.api_key,
        model=config.llm.model,
        max_concurrent=config.llm.max_concurrent,
        cache_ttl_min=config.llm.cache_ttl_min,
    )
    detector = LLMSignalDetector(
        sig_cfg,
        llm,
        bankroll,
        forecast_client=fc,
        max_markets_per_scan=config.llm.max_markets_per_scan,
        min_liquidity_usd=config.llm.min_liquidity_usd,
    )
    while True:
        await _await_resume()
        if not config.llm.enabled:
            await asyncio.sleep(5)
            continue
        try:
            llm.spot_prices = {
                s: p for s in binance.SYMBOLS
                if (p := binance.get_price(s))
            }
            if state.markets:
                state.llm_opps = await detector.scan(state.markets)
                state.llm_scans += 1
                if state.llm_opps:
                    state.last_signal_at["llm"] = _now_iso()
                if state.llm_scans % _BANKROLL_REFRESH_EVERY == 0:
                    await _refresh_bankroll(trader, detector.sizer)
                for sig in state.llm_opps:
                    await trader.execute_signal(sig, state.market_map)
                if state.llm_opps:
                    logger.info(
                        f"LLM: {len(state.llm_opps)} opps, "
                        f"top edge={state.llm_opps[0].edge:.1%}"
                    )
        except Exception as e:
            logger.error(f"LLM loop: {e}")
        await asyncio.sleep(config.llm.refresh_interval_sec)


async def weather_loop(trader: Trader):
    """Strategy: Weather market signal trading (Open-Meteo)."""
    sig_cfg = SignalArbConfig(
        min_edge=config.weather.min_edge,
        fee_pct=config.arbitrage.fee_pct,
        max_position_usd=config.weather.max_position_usd,
        kelly_fraction=config.weather.kelly_fraction,
    )
    try:
        bankroll = await trader.client.get_balance()
        if bankroll <= 0:
            bankroll = config.risk.paper_bankroll
    except Exception:
        bankroll = config.risk.paper_bankroll

    weather = WeatherClient()
    detector = WeatherSignalDetector(sig_cfg, weather, bankroll)

    async with weather:
        while True:
            await _await_resume()
            if not config.weather.enabled:
                await asyncio.sleep(5)
                continue
            try:
                await weather.refresh_all()
                if state.markets:
                    state.weather_opps = detector.scan(state.markets)
                    state.weather_scans += 1
                    if state.weather_opps:
                        state.last_signal_at["weather"] = _now_iso()
                    if state.weather_scans % _BANKROLL_REFRESH_EVERY == 0:
                        await _refresh_bankroll(trader, detector.sizer)
                    for sig in state.weather_opps:
                        await trader.execute_signal(sig, state.market_map)
                    if state.weather_opps:
                        logger.info(
                            f"Weather: {len(state.weather_opps)} opps, "
                            f"top edge={state.weather_opps[0].edge:.1%}"
                        )
            except Exception as e:
                logger.error(f"Weather loop: {e}")
            await asyncio.sleep(config.weather.refresh_interval_sec)


async def favorite_short_loop(trader: Trader):
    """Strategy: Short overpriced favorites (favorite-longshot bias)."""
    cfg = config.favorite_short
    try:
        bankroll = await trader.client.get_balance()
        if bankroll <= 0:
            bankroll = config.risk.paper_bankroll
    except Exception:
        bankroll = config.risk.paper_bankroll

    detector = FavoriteShortDetector(cfg, bankroll)
    while True:
        await _await_resume()
        if not cfg.enabled:
            await asyncio.sleep(5)
            continue
        try:
            if state.markets:
                state.fav_opps = detector.scan(state.markets)
                state.fav_scans += 1
                if state.fav_opps:
                    state.last_signal_at["favorite_short"] = _now_iso()
                if state.fav_scans % _BANKROLL_REFRESH_EVERY == 0:
                    await _refresh_bankroll(trader, detector.sizer)
                for sig in state.fav_opps:
                    await trader.execute_signal(sig, state.market_map)
                if state.fav_opps:
                    logger.info(
                        f"FavoriteShort: {len(state.fav_opps)} opps, "
                        f"top edge={state.fav_opps[0].edge:.1%}"
                    )
        except Exception as e:
            logger.error(f"Favorite short loop: {e}")
        await asyncio.sleep(cfg.refresh_interval_sec)


async def oracle_squeeze_loop(trader: Trader):
    """Strategy: Buy near-certain outcomes in oracle resolution window."""
    cfg = config.oracle_squeeze
    try:
        bankroll = await trader.client.get_balance()
        if bankroll <= 0:
            bankroll = config.risk.paper_bankroll
    except Exception:
        bankroll = config.risk.paper_bankroll

    detector = OracleSqueezeDetector(cfg, bankroll)
    while True:
        await _await_resume()
        if not cfg.enabled:
            await asyncio.sleep(5)
            continue
        try:
            if state.markets:
                state.squeeze_opps = detector.scan(state.markets)
                state.squeeze_scans += 1
                if state.squeeze_opps:
                    state.last_signal_at["oracle_squeeze"] = _now_iso()
                if state.squeeze_scans % _BANKROLL_REFRESH_EVERY == 0:
                    await _refresh_bankroll(trader, detector.sizer)
                for sig in state.squeeze_opps:
                    await trader.execute_signal(sig, state.market_map)
                if state.squeeze_opps:
                    logger.info(
                        f"OracleSqueeze: {len(state.squeeze_opps)} opps, "
                        f"top edge={state.squeeze_opps[0].edge:.1%}"
                    )
        except Exception as e:
            logger.error(f"Oracle squeeze loop: {e}")
        await asyncio.sleep(cfg.refresh_interval_sec)


async def semantic_arb_loop(trader: Trader):
    """Strategy: Exact cross-platform price gap on semantically identical markets.

    Sources: Kalshi (public API) + Manifold (from ForecastClient).
    """
    cfg = config.semantic_arb
    try:
        bankroll = await trader.client.get_balance()
        if bankroll <= 0:
            bankroll = config.risk.paper_bankroll
    except Exception:
        bankroll = config.risk.paper_bankroll

    detector = SemanticArbDetector(cfg, bankroll)
    fc = ForecastClient()

    async with KalshiSignalClient() as kalshi_signal, fc:
        while True:
            await _await_resume()
            if not cfg.enabled:
                await asyncio.sleep(5)
                continue
            try:
                if state.markets:
                    # Kalshi markets as signal source
                    ext_raw = await kalshi_signal.get_external_markets(max_pages=5)
                    ext_markets = [
                        {
                            "question": m.get("question", m.get("title", "")),
                            "yes_price": float(m.get("yes_price", 0.5)),
                            "platform": "kalshi",
                        }
                        for m in ext_raw
                    ]

                    # Manifold markets as additional signal source
                    try:
                        await fc.refresh()
                        manifold_ext = [
                            {
                                "question": q.title,
                                "yes_price": q.yes_prob,
                                "platform": "manifold",
                            }
                            for q in fc._questions
                            if q.source == "manifold" and q.yes_prob > 0
                        ]
                        ext_markets.extend(manifold_ext)
                        logger.debug(
                            f"SemanticArb: {len(ext_raw)} kalshi + "
                            f"{len(manifold_ext)} manifold ext markets"
                        )
                    except Exception as e:
                        logger.debug(f"SemanticArb manifold fetch: {e}")

                    detector.load_external(ext_markets)
                    state.semarg_opps = detector.scan(state.markets)
                    state.semarg_scans += 1
                    if state.semarg_opps:
                        state.last_signal_at["semantic_arb"] = _now_iso()
                    if state.semarg_scans % _BANKROLL_REFRESH_EVERY == 0:
                        await _refresh_bankroll(trader, detector.sizer)
                    for sig in state.semarg_opps:
                        await trader.execute_signal(sig, state.market_map)
                    if state.semarg_opps:
                        logger.info(
                            f"SemanticArb: {len(state.semarg_opps)} opps, "
                            f"top edge={state.semarg_opps[0].edge:.1%}"
                        )
            except Exception as e:
                logger.error(f"Semantic arb loop: {e}")
            await asyncio.sleep(cfg.refresh_interval_sec)


async def orderbook_momentum_loop(client: PolymarketClient, trader: Trader):
    """Strategy: OFI momentum from live CLOB order book depth."""
    cfg = config.orderbook_momentum
    try:
        bankroll = await trader.client.get_balance()
        if bankroll <= 0:
            bankroll = config.risk.paper_bankroll
    except Exception:
        bankroll = config.risk.paper_bankroll

    detector = OrderbookMomentumDetector(cfg, client, bankroll)
    while True:
        await _await_resume()
        if not cfg.enabled:
            await asyncio.sleep(5)
            continue
        try:
            if state.markets:
                state.ofi_opps = await detector.scan(state.markets)
                state.ofi_scans += 1
                if state.ofi_opps:
                    state.last_signal_at["orderbook_momentum"] = _now_iso()
                if state.ofi_scans % _BANKROLL_REFRESH_EVERY == 0:
                    await _refresh_bankroll(trader, detector.sizer)
                for sig in state.ofi_opps:
                    await trader.execute_signal(sig, state.market_map)
                if state.ofi_opps:
                    logger.info(
                        f"OFI: {len(state.ofi_opps)} opps, "
                        f"top edge={state.ofi_opps[0].edge:.1%}"
                    )
        except Exception as e:
            logger.error(f"OFI loop: {e}")
        await asyncio.sleep(cfg.refresh_interval_sec)


async def btc_5min_loop(binance: BinanceClient, trader: Trader):
    """Strategy: 5-minute BTC up/down markets using Coinbase microstructure signals."""
    cfg = config.btc_5min
    try:
        bankroll = await trader.client.get_balance()
        if bankroll <= 0:
            bankroll = config.risk.paper_bankroll
    except Exception:
        bankroll = config.risk.paper_bankroll

    detector = BTC5MinDetector(cfg, binance, bankroll)
    while True:
        await _await_resume()
        if not cfg.enabled:
            await asyncio.sleep(10)
            continue
        try:
            # 5-min markets have very low volume and sit at offset ~3400+ in the
            # volume-sorted list. Fetch recently-opened markets (sorted by startDate
            # desc) to reliably surface the current 5-min window market.
            recent = await trader.client.get_recent_markets(limit=200)
            # Merge: recent markets first, then de-dup against state.markets
            seen_ids = {m.market_id for m in recent}
            combined = recent + [m for m in state.markets if m.market_id not in seen_ids]

            state.btc5min_opps = await detector.scan(combined)
            _bot_state._btc5min_signals = {**detector._last_signals}
            state.btc5min_scans += 1
            if state.btc5min_opps:
                state.last_signal_at["btc_5min"] = _now_iso()
            if state.btc5min_scans % _BANKROLL_REFRESH_EVERY == 0:
                await _refresh_bankroll(trader, detector.sizer)
            # Build a market_map that includes the recent markets for execution
            # and for position P&L tracking (state.market_map only has volume-sorted markets)
            combined_map = {m.market_id: m for m in combined}
            state.extended_market_map = combined_map
            for sig in state.btc5min_opps:
                await trader.execute_signal(sig, combined_map)
            if state.btc5min_opps:
                logger.info(
                    f"BTC5Min: {len(state.btc5min_opps)} opps, "
                    f"top edge={state.btc5min_opps[0].edge:.1%}"
                )
        except Exception as e:
            logger.error(f"BTC 5-min loop: {e}")
        await asyncio.sleep(cfg.refresh_interval_sec)


async def market_maker_loop(client: PolymarketClient):
    """Strategy: Resting limit orders on both sides + CLOB v2 maker rebates."""
    mm = MarketMakerStrategy(config.market_maker, client)
    try:
        while True:
            await _await_resume()
            if not config.market_maker.enabled:
                await asyncio.sleep(5)
                continue
            try:
                if state.markets:
                    active = await mm.run_once(state.markets)
                    state.mm_active = active
                    state.mm_scans += 1
                    state.mm_quotes = mm.summary()
            except Exception as e:
                logger.error(f"MM loop: {e}")
            await asyncio.sleep(config.market_maker.refresh_interval_sec)
    finally:
        try:
            await mm.cancel_all()
        except Exception:
            pass


async def position_loop(pos_manager: PositionManager):
    while True:
        try:
            m_map = state.extended_market_map if state.extended_market_map else state.market_map
            await pos_manager.refresh(m_map)
        except Exception as e:
            logger.error(f"Position loop: {e}")
        await asyncio.sleep(config.positions.refresh_interval_sec)


async def pnl_snapshot_loop(storage: Storage, pos_manager: PositionManager):
    """Write a P&L snapshot every minute for the history chart."""
    while True:
        await asyncio.sleep(60)
        try:
            snap = pos_manager.snapshot
            if snap:
                storage.log_pnl_snapshot(
                    balance=snap.balance_usd,
                    unrealized=snap.unrealized_pnl,
                    realized=snap.realized_pnl,
                )
        except Exception as e:
            logger.error(f"PnL snapshot: {e}")


async def exit_loop(
    client: PolymarketClient,
    pos_manager: PositionManager,
    storage: Storage,
):
    """Strategy: sell positions where current mid > exit_threshold to lock profit."""
    from config import ExitConfig as _ExitCfg
    ex_cfg = config.position_exit
    strategy = ExitStrategy(ex_cfg, client)
    while True:
        await _await_resume()
        if not ex_cfg.enabled:
            await asyncio.sleep(10)
            continue
        try:
            snap = pos_manager.snapshot
            if snap and snap.open_positions:
                exits = await strategy.scan_and_exit(
                    snap.open_positions, state.market_map, dry_run=config.dry_run
                )
                if exits:
                    logger.info(f"Exit strategy: closed {len(exits)} position(s)")
        except Exception as e:
            logger.error(f"Exit loop: {e}")
        await asyncio.sleep(ex_cfg.refresh_interval_sec)


async def kyle_lambda_loop(client: PolymarketClient):
    """Background: compute Kyle's Lambda for top markets to assess price impact."""
    kl_cfg = config.kyle_lambda
    if not kl_cfg.enabled:
        return
    tracker = KyleLambdaTracker(client, max_markets=kl_cfg.max_markets)
    _bot_state._kyle_lambda_ref = tracker
    while True:
        try:
            if state.markets:
                await tracker.refresh(state.markets)
                logger.debug(f"Kyle Lambda: updated {len(tracker._lambdas)} markets")
        except Exception as e:
            logger.error(f"Kyle Lambda loop: {e}")
        await asyncio.sleep(kl_cfg.refresh_interval_sec)


async def evolution_loop(storage: Storage, llm_client):
    """Weekly LLM performance review with parameter suggestions."""
    ev_cfg = config.evolution
    if not ev_cfg.enabled or not llm_client:
        return
    agent = EvolutionAgent(ev_cfg, llm_client, storage)
    _bot_state._evolution_ref = agent
    while True:
        try:
            rec = await agent.maybe_run()
            if rec:
                logger.info("Evolution agent completed review — check /api/evolution")
        except Exception as e:
            logger.error(f"Evolution loop: {e}")
        # Check every hour whether a review is due
        await asyncio.sleep(3600)


async def daily_summary_loop(
    alerter: Alerter, storage: Storage, pos_manager: PositionManager
):
    """Send a Discord daily summary at midnight UTC."""
    while True:
        now = datetime.now(timezone.utc)
        tomorrow_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        wait = (tomorrow_midnight - now).total_seconds()
        await asyncio.sleep(wait)
        try:
            stats = storage.get_strategy_pnl()
            snap = pos_manager.snapshot
            if snap and alerter:
                await alerter.daily_summary(
                    balance=snap.balance_usd,
                    total_pnl=snap.total_pnl,
                    unrealized=snap.unrealized_pnl,
                    top_strategies=stats[:5],
                )
        except Exception as e:
            logger.error(f"Daily summary loop: {e}")


async def outcome_tracker_loop(
    storage: Storage, pos_manager: PositionManager, alerter: "Alerter"
):
    """Periodically resolve trade outcomes by checking if the market settled."""
    while True:
        await asyncio.sleep(300)
        try:
            unresolved = storage.get_unresolved_trades()
            combined_map = state.extended_market_map if state.extended_market_map else state.market_map
            for row in unresolved:
                market = combined_map.get(row["market_id"])
                if not market or market.status != "settled" or not market.result:
                    continue
                our_side = row["side"]
                if our_side == "both":
                    # Overround arb: we hold YES + NO; guaranteed $1 payout at settlement
                    storage.update_trade_outcome(row["id"], "won")
                    logger.info(
                        f"Outcome: arb trade {row['id']} → won "
                        f"(market settled, both legs pay)"
                    )
                    if alerter:
                        _alert_task_standalone(alerter.outcome_resolved(
                            row["id"], row["market_id"], "won", row["arb_type"] or "arb"
                        ))
                    continue
                if not our_side:
                    continue
                outcome = "won" if market.result == our_side else "lost"
                storage.update_trade_outcome(row["id"], outcome)
                logger.info(
                    f"Outcome: trade {row['id']} → {outcome} "
                    f"(market={market.result}, held={our_side}, type={row['arb_type']})"
                )
                if alerter:
                    _alert_task_standalone(alerter.outcome_resolved(
                        row["id"], row["market_id"], outcome, row["arb_type"] or "signal"
                    ))
        except Exception as e:
            logger.error(f"Outcome tracker: {e}")


async def ui_loop(live: Live, storage: Storage, pos_manager: PositionManager):
    while True:
        live.update(build_ui(storage, pos_manager))
        await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------

def _free_port(port: int):
    import subprocess
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if not pid:
                continue
            subprocess.run(["kill", pid], capture_output=True)
        if pids:
            import time
            time.sleep(0.5)
            result2 = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True,
            )
            for pid in result2.stdout.strip().split():
                if pid:
                    subprocess.run(["kill", "-9", pid], capture_output=True)
                    logger.info(f"Freed port {port} (killed PID {pid})")
    except Exception:
        pass


async def web_server_loop():
    try:
        import uvicorn
        from web_gui import app as web_app, broadcast_loop
        web_port = int(os.getenv("LEO_WEB_PORT", "5002"))
        _free_port(web_port)
        await asyncio.sleep(0.5)
        server_cfg = uvicorn.Config(
            web_app,
            host="0.0.0.0",
            port=web_port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(server_cfg)
        console.print(
            f"[bold cyan]Web dashboard:[/] http://localhost:{web_port}"
        )
        await asyncio.gather(server.serve(), broadcast_loop())
    except ImportError:
        logger.warning("uvicorn not installed — web GUI disabled.")
    except Exception as e:
        logger.error(f"Web server error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run():
    storage = Storage(config.storage)
    _bot_state._storage_ref = storage
    _bot_state.config = config

    # Pause / stop events — shared with web_gui control endpoints
    _resume_event = asyncio.Event()
    _resume_event.set()   # start running
    _bot_state._resume_event = _resume_event
    stop_event_main = asyncio.Event()
    _bot_state._stop_event = stop_event_main

    if not config.polymarket.private_key:
        console.print(
            "[bold yellow]POLY_PRIVATE_KEY not set — "
            "running in read-only / dry-run mode[/]"
        )

    alerter = Alerter(config.alerting.discord_webhook_url)
    news_client = NewsClient(config.news_fade.news_api_key)

    async with PolymarketClient(config.polymarket) as client:
        _bot_state._client_ref = client
        arb_detector = ArbitrageDetector(config.arbitrage)
        pos_manager = PositionManager(
            client, config.positions.refresh_interval_sec
        )

        # Shared cross-strategy trackers
        confluence = ConfluenceTracker(config.confluence)
        _bot_state._confluence_ref = confluence
        hurst_tracker = HurstTracker()
        _bot_state._hurst_ref = hurst_tracker

        trader = Trader(config, client, storage, pos_manager, alerter, confluence)
        _bot_state._pos_manager_ref = pos_manager
        _bot_state._trader_ref = trader

        async with BinanceClient() as binance:
            tasks = [
                asyncio.create_task(
                    market_refresh_loop(client), name="market-refresh"
                ),
                asyncio.create_task(
                    arb_loop(arb_detector, trader, client), name="arb"
                ),
                asyncio.create_task(
                    crypto_signal_loop(binance, trader), name="crypto"
                ),
                asyncio.create_task(
                    cross_platform_loop(trader), name="cross"
                ),
                asyncio.create_task(
                    range_straddle_loop(binance, trader), name="range"
                ),
                asyncio.create_task(corr_loop(trader), name="corr"),
                asyncio.create_task(
                    news_fade_loop(trader, news_client), name="fade"
                ),
                asyncio.create_task(forecast_loop(trader), name="forecast"),
                asyncio.create_task(llm_loop(trader, binance), name="llm"),
                asyncio.create_task(weather_loop(trader), name="weather"),
                asyncio.create_task(
                    favorite_short_loop(trader), name="fav_short"
                ),
                asyncio.create_task(
                    oracle_squeeze_loop(trader), name="oracle_squeeze"
                ),
                asyncio.create_task(
                    semantic_arb_loop(trader), name="semantic_arb"
                ),
                asyncio.create_task(
                    orderbook_momentum_loop(client, trader), name="ofi"
                ),
                asyncio.create_task(
                    btc_5min_loop(binance, trader), name="btc_5min"
                ),
                asyncio.create_task(
                    market_maker_loop(client), name="market_maker"
                ),
                asyncio.create_task(
                    position_loop(pos_manager), name="positions"
                ),
                asyncio.create_task(
                    pnl_snapshot_loop(storage, pos_manager), name="pnl-snapshot"
                ),
                asyncio.create_task(
                    exit_loop(client, pos_manager, storage), name="exit"
                ),
                asyncio.create_task(
                    kyle_lambda_loop(client), name="kyle-lambda"
                ),
                asyncio.create_task(
                    outcome_tracker_loop(storage, pos_manager, alerter), name="outcome-tracker"
                ),
                asyncio.create_task(web_server_loop(), name="web"),
            ]

            # LLM-dependent tasks (only if LLM is configured)
            if config.llm.enabled and config.llm.api_key:
                llm_for_evolution = LLMClient(
                    api_key=config.llm.api_key,
                    model=config.llm.model,
                    max_concurrent=1,
                    cache_ttl_min=0,
                )
                tasks.append(asyncio.create_task(
                    evolution_loop(storage, llm_for_evolution), name="evolution"
                ))

            tasks.append(asyncio.create_task(
                daily_summary_loop(alerter, storage, pos_manager), name="daily-summary"
            ))

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
                    # Wait for SIGINT/SIGTERM or web-triggered stop
                    done, _ = await asyncio.wait(
                        [asyncio.ensure_future(stop_event.wait()),
                         asyncio.ensure_future(stop_event_main.wait())],
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
