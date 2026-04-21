"""
Leo — Kalshi Prediction Market Trading Bot

Strategies (all execute on Kalshi):
  1. Real-time crypto price signal  (Coinbase → RSI/momentum/VWAP)
  2. Range straddle + logical arb   (weekly BTC/ETH range contracts)
  3. Correlation/logical arb        (mathematically impossible pricing)
  4. News overreaction fade         (fade spikes 1.5-4h after move)
  +  Overround arb                  (YES_bid + NO_bid > 1)
  +  Position tracking              (live P&L)
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from api_clients.binance_client import BinanceClient
from api_clients.forecast_client import ForecastClient
from api_clients.kalshi_client import KalshiClient, Market
from api_clients.llm_client import LLMClient
from api_clients.polymarket_client import PolymarketClient
from api_clients.weather_client import WeatherClient
from arbitrage import ArbitrageDetector, ArbOpportunity
from strategies.cross_platform import (
    CrossPlatformConfig,
    CrossPlatformDetector,
    CrossPlatformOpportunity,
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
from trader import Trader

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("data/leo.log"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("leo.main")
console = Console()

os.makedirs("data", exist_ok=True)


# ---------------------------------------------------------------------------
# Shared bot state
# ---------------------------------------------------------------------------

class BotState:
    def __init__(self):
        self.markets: list[Market] = []
        self.market_map: dict[str, Market] = {}

        # Opportunities per strategy
        self.arb_opps: list[ArbOpportunity] = []
        self.crypto_opps: list[AggregatedSignal] = []
        self.cross_opps: list[CrossPlatformOpportunity] = []
        self.range_opps: list[AggregatedSignal] = []
        self.corr_opps: list[CorrelatedOpportunity] = []
        self.fade_opps: list[AggregatedSignal] = []

        self.forecast_opps: list[AggregatedSignal] = []
        self.llm_opps: list[AggregatedSignal] = []
        self.weather_opps: list[AggregatedSignal] = []

        # Scan counters
        self.arb_scans = 0
        self.crypto_scans = 0
        self.cross_scans = 0
        self.range_scans = 0
        self.corr_scans = 0
        self.fade_scans = 0
        self.forecast_scans = 0
        self.llm_scans = 0
        self.weather_scans = 0

        # Web-gui compat aliases
        self.signal_opps: list[AggregatedSignal] = []
        self.signal_scans = 0
        self.mm_scans = 0
        self.whale_signals = 0


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
    h = (
        close_time - datetime.now(timezone.utc)
    ).total_seconds() / 3600
    return f"{h:.1f}h"


def _arb_table() -> Table:
    t = Table(
        "Market", "Sum", "Net%", "Closes",
        box=box.SIMPLE_HEAD, title="[cyan]Overround Arb[/]",
        title_style="bold", min_width=50,
    )
    for o in state.arb_opps[:5]:
        t.add_row(
            o.market_id[:26] + ("…" if len(o.market_id) > 26 else ""),
            f"[green]{o.yes_bid + o.no_bid:.4f}[/]",
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
        t.add_row(
            o.market_id[:24] + ("…" if len(o.market_id) > 24 else ""),
            f"{o.market_prob:.0%}→{o.model_prob:.0%}",
            f"[{clr}]{o.edge:+.1%}[/]",
            o.recommended_side,
        )
    if not state.crypto_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _cross_table() -> Table:
    t = Table(
        "Market", "Buy@", "Sell@", "Net%",
        box=box.SIMPLE_HEAD, title="[blue]Cross-Platform[/]",
        title_style="bold", min_width=50,
    )
    for o in state.cross_opps[:5]:
        t.add_row(
            o.kalshi_market_id[:24]
            + ("…" if len(o.kalshi_market_id) > 24 else ""),
            f"{o.buy_platform[:3]} {o.buy_price:.3f}",
            f"{o.sell_platform[:3]} {o.sell_price:.3f}",
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
            t.add_row(
                o.market_id[:24] + ("…" if len(o.market_id) > 24 else ""),
                f"{o.model_prob:.0%}",
                f"{o.market_prob:.0%}",
                f"[{clr}]{o.edge:+.1%}[/]",
            )
        else:
            t.add_row(
                o.market_id_mispriced[:24]
                + ("…" if len(o.market_id_mispriced) > 24 else ""),
                o.relation.value,
                f"{o.edge:.1%}",
                o.recommended_side,
            )
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
        t.add_row(
            o.market_id[:30] + ("…" if len(o.market_id) > 30 else ""),
            o.recommended_side,
            f"[{clr}]{o.edge:+.1%}[/]",
        )
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
        t.add_row(
            o.market_id[:22] + ("…" if len(o.market_id) > 22 else ""),
            (o.source or "")[:10],
            f"{o.model_prob:.0%}",
            f"{o.market_prob:.0%}",
            f"[{clr}]{o.edge:+.1%}[/]",
        )
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
        t.add_row(
            o.market_id[:22] + ("…" if len(o.market_id) > 22 else ""),
            f"{o.model_prob:.0%}",
            f"{o.market_prob:.0%}",
            f"[{clr}]{o.edge:+.1%}[/]",
            f"{o.confidence:.0%}" if o.confidence else "—",
        )
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
        t.add_row(
            o.market_id[:24] + ("…" if len(o.market_id) > 24 else ""),
            f"{o.model_prob:.0%}",
            f"{o.market_prob:.0%}",
            f"[{clr}]{o.edge:+.1%}[/]",
        )
    if not state.weather_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _stats_table(storage: Storage, pos_manager: PositionManager) -> Table:
    trade_stats = storage.get_pnl_summary()
    pos = pos_manager.summary()
    t = Table(box=box.SIMPLE, show_header=False, min_width=26)
    t.add_column(style="dim", no_wrap=True)
    t.add_column(justify="right")
    t.add_row("Mode", _mode())
    t.add_row("Balance", f"${pos.get('balance_usd', 0):.2f}")
    t.add_row("Positions", str(pos.get("open_positions", 0)))
    t.add_row("Unrealized", f"${pos.get('unrealized_pnl', 0):.2f}")
    t.add_row("Realized", f"${pos.get('realized_pnl', 0):.2f}")
    t.add_row("─" * 10, "─" * 7)
    t.add_row("Markets", str(len(state.markets)))
    t.add_row("Arb scans", str(state.arb_scans))
    t.add_row("Crypto scans", str(state.crypto_scans))
    t.add_row("Cross scans", str(state.cross_scans))
    t.add_row("Range scans", str(state.range_scans))
    t.add_row("Fade scans", str(state.fade_scans))
    t.add_row("Forecast scans", str(state.forecast_scans))
    t.add_row("LLM scans", str(state.llm_scans))
    t.add_row("Weather scans", str(state.weather_scans))
    t.add_row("─" * 10, "─" * 7)
    t.add_row("Trades", str(trade_stats.get("total_trades", 0)))
    t.add_row(
        "Est. P&L",
        f"${trade_stats.get('estimated_pnl_usd') or 0:.2f}",
    )
    return t


def build_ui(storage: Storage, pos_manager: PositionManager) -> Panel:
    row1 = Columns([_arb_table(), _crypto_table()], equal=True)
    row2 = Columns([_cross_table(), _range_table()], equal=True)
    row3 = Columns([_fade_table(), _forecast_table()], equal=True)
    row4 = Columns([_llm_table(), _weather_table()], equal=True)
    row5 = Columns([_stats_table(storage, pos_manager)])
    return Panel(
        Group(row1, row2, row3, row4, row5),
        title=(
            "[bold]Leo — Kalshi Trading Bot[/bold]  "
            + datetime.now().strftime("%H:%M:%S")
        ),
        border_style="cyan",
    )


# ---------------------------------------------------------------------------
# Strategy loops
# ---------------------------------------------------------------------------

_BANKROLL_REFRESH_EVERY = 10   # scans between balance re-fetches


async def _refresh_bankroll(trader: Trader, sizer) -> None:
    """Fetch current Kalshi balance and update a KellySizer in-place."""
    try:
        bal = await trader.client.get_balance()
        sizer.update_bankroll(bal)
    except Exception:
        pass


async def market_refresh_loop(client: KalshiClient):
    """Dedicated loop: fetch all Kalshi markets and update shared state."""
    while True:
        try:
            markets = await client.get_all_markets()
            state.markets = markets
            state.market_map = {m.market_id: m for m in markets}
            logger.debug(f"Market refresh: {len(markets)} markets loaded")
        except Exception as e:
            logger.error(f"Market refresh: {e}")
        await asyncio.sleep(30)


async def arb_loop(detector: ArbitrageDetector, trader: Trader):
    """Strategy: Overround arb — YES_bid + NO_bid > 1."""
    while True:
        try:
            if state.markets:
                state.arb_opps = detector.scan(state.markets)
                state.arb_scans += 1
                for opp in state.arb_opps:
                    await trader.execute(opp)
        except Exception as e:
            logger.error(f"Arb loop: {e}")
        await asyncio.sleep(config.arbitrage.poll_interval_sec)


async def crypto_signal_loop(
    binance: BinanceClient, trader: Trader
):
    """Strategy 1: Real-time Coinbase price signal."""
    if not config.crypto_signal.enabled:
        return

    sig_cfg = SignalArbConfig(
        min_edge=config.crypto_signal.min_edge,
        fee_pct=config.arbitrage.fee_pct,
        max_position_usd=config.crypto_signal.max_position_usd,
        kelly_fraction=config.crypto_signal.kelly_fraction,
    )
    try:
        bankroll = await trader.client.get_balance()
    except Exception:
        bankroll = 500.0

    detector = CryptoSignalDetector(sig_cfg, binance, bankroll)
    while True:
        try:
            await binance.refresh_all()
            if state.markets:
                state.crypto_opps = detector.scan(state.markets)
                state.signal_opps = state.crypto_opps
                state.crypto_scans += 1
                state.signal_scans = state.crypto_scans
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
    """Strategy 2: Polymarket prices as a Kalshi signal source."""
    if not config.cross_arb.enabled:
        return

    cross_cfg = CrossPlatformConfig(
        min_profit_pct=config.cross_arb.min_profit_pct,
        max_position_usd=config.cross_arb.max_position_usd,
        signal_only=config.cross_arb.signal_only,
    )
    detector = CrossPlatformDetector(cross_cfg)

    async with PolymarketClient() as poly:
        while True:
            try:
                if state.markets:
                    poly_markets = await poly.get_all_active_markets(
                        max_pages=3
                    )
                    ext_markets = poly.to_external_markets(poly_markets)
                    detector.load_external_markets(ext_markets)
                    state.cross_opps = detector.scan(state.markets)
                    state.cross_scans += 1
                    # Always execute on Kalshi, even when signal_only=False
                    for opp in state.cross_opps:
                        sig = AggregatedSignal(
                            market_id=opp.kalshi_market_id,
                            question=opp.question,
                            market_prob=opp.buy_price,
                            model_prob=opp.sell_price,
                            edge=opp.net_profit_pct,
                            recommended_side=(
                                "yes"
                                if opp.buy_platform == "kalshi"
                                else "no"
                            ),
                            source=f"cross:{opp.sell_platform}",
                            recommended_size_usd=opp.max_size_usd,
                        )
                        await trader.execute_signal(sig, state.market_map)
                    if state.cross_opps:
                        logger.info(
                            f"Cross-platform: {len(state.cross_opps)} opps"
                        )
            except Exception as e:
                logger.error(f"Cross-platform loop: {e}")
            await asyncio.sleep(config.cross_arb.refresh_interval_sec)


async def range_straddle_loop(
    binance: BinanceClient, trader: Trader
):
    """Strategy 2: Range straddle + logical arb."""
    if not config.range_straddle.enabled:
        return

    sig_cfg = SignalArbConfig(
        min_edge=config.range_straddle.min_edge,
        fee_pct=config.arbitrage.fee_pct,
        max_position_usd=config.range_straddle.max_position_usd,
        kelly_fraction=config.range_straddle.kelly_fraction,
    )
    try:
        bankroll = await trader.client.get_balance()
    except Exception:
        bankroll = 500.0

    detector = RangeStraddleDetector(sig_cfg, binance, bankroll)
    while True:
        try:
            # Snapshots already refreshed by crypto_signal_loop;
            # just need up-to-date prices, no full candle fetch needed.
            await binance.fetch_snapshots()
            if state.markets:
                state.range_opps = detector.scan(state.markets)
                state.range_scans += 1
                if state.range_scans % _BANKROLL_REFRESH_EVERY == 0:
                    await _refresh_bankroll(trader, detector.sizer)
                for sig in state.range_opps:
                    await trader.execute_signal(sig, state.market_map)
        except Exception as e:
            logger.error(f"Range straddle loop: {e}")
        await asyncio.sleep(60)


async def corr_loop(trader: Trader):
    """Strategy 3: Logical/correlation arb."""
    if not config.correlated.enabled:
        return

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
        try:
            if state.markets:
                # Rebuild detector only when market list changes size
                if detector is None or len(state.markets) != last_market_count:
                    detector = build_correlated_detector(
                        state.markets, corr_cfg, auto_cfg
                    )
                    last_market_count = len(state.markets)

                state.corr_opps = detector.scan(state.markets)
                state.corr_scans += 1
                for opp in state.corr_opps:
                    await trader.execute_correlated(opp, state.market_map)
        except Exception as e:
            logger.error(f"Corr loop: {e}")
        await asyncio.sleep(config.correlated.poll_interval_sec)


async def news_fade_loop(trader: Trader):
    """Strategy 4: News overreaction fade."""
    if not config.news_fade.enabled:
        return

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
    except Exception:
        bankroll = 500.0

    detector = NewsFadeDetector(sig_cfg, fade_cfg, bankroll)
    while True:
        try:
            if state.markets:
                state.fade_opps = detector.scan(state.markets)
                state.fade_scans += 1
                if state.fade_scans % _BANKROLL_REFRESH_EVERY == 0:
                    await _refresh_bankroll(trader, detector.sizer)
                for sig in state.fade_opps:
                    await trader.execute_signal(sig, state.market_map)
        except Exception as e:
            logger.error(f"News fade loop: {e}")
        await asyncio.sleep(300)   # check every 5 min


async def forecast_loop(trader: Trader):
    """Strategy 5: Community forecast aggregation (Metaculus + Manifold)."""
    if not config.forecast.enabled:
        return

    sig_cfg = SignalArbConfig(
        min_edge=config.forecast.min_edge,
        fee_pct=config.arbitrage.fee_pct,
        max_position_usd=config.forecast.max_position_usd,
        kelly_fraction=config.forecast.kelly_fraction,
    )
    try:
        bankroll = await trader.client.get_balance()
    except Exception:
        bankroll = 500.0

    fc = ForecastClient()
    detector = ForecastSignalDetector(sig_cfg, fc, bankroll)
    async with fc:
        while True:
            try:
                await fc.refresh()
                if state.markets:
                    state.forecast_opps = detector.scan(state.markets)
                    state.forecast_scans += 1
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
    """Strategy 6: LLM fundamental analysis (Claude Haiku)."""
    if not config.llm.enabled:
        return

    sig_cfg = SignalArbConfig(
        min_edge=config.llm.min_edge,
        fee_pct=config.arbitrage.fee_pct,
        max_position_usd=config.llm.max_position_usd,
        kelly_fraction=config.llm.kelly_fraction,
    )
    try:
        bankroll = await trader.client.get_balance()
    except Exception:
        bankroll = 500.0

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
        try:
            # Keep LLM grounded with live prices before each scan
            llm.spot_prices = {
                s: p for s in binance.SYMBOLS
                if (p := binance.get_price(s))
            }
            if state.markets:
                state.llm_opps = await detector.scan(state.markets)
                state.llm_scans += 1
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
    """Strategy 7: Weather market signal trading (Open-Meteo)."""
    if not config.weather.enabled:
        return

    sig_cfg = SignalArbConfig(
        min_edge=config.weather.min_edge,
        fee_pct=config.arbitrage.fee_pct,
        max_position_usd=config.weather.max_position_usd,
        kelly_fraction=config.weather.kelly_fraction,
    )
    try:
        bankroll = await trader.client.get_balance()
    except Exception:
        bankroll = 500.0

    weather = WeatherClient()
    detector = WeatherSignalDetector(sig_cfg, weather, bankroll)

    async with weather:
        while True:
            try:
                await weather.refresh_all()
                if state.markets:
                    state.weather_opps = detector.scan(state.markets)
                    state.weather_scans += 1
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


async def position_loop(pos_manager: PositionManager):
    while True:
        try:
            await pos_manager.refresh(state.market_map)
        except Exception as e:
            logger.error(f"Position loop: {e}")
        await asyncio.sleep(config.positions.refresh_interval_sec)


async def ui_loop(
    live: Live,
    storage: Storage,
    pos_manager: PositionManager,
):
    while True:
        live.update(build_ui(storage, pos_manager))
        await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------

def _free_port(port: int):
    """SIGTERM then SIGKILL any process holding the given port."""
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
            # Give processes a moment to exit cleanly
            import time
            time.sleep(0.5)
            # Force-kill any survivors
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

    if not config.kalshi.key_id:
        console.print(
            "[bold yellow]KALSHI_KEY_ID not set — "
            "running in dry-run mode[/]"
        )

    async with KalshiClient(config.kalshi) as client:
        arb_detector = ArbitrageDetector(config.arbitrage)
        pos_manager = PositionManager(
            client, config.positions.refresh_interval_sec
        )
        trader = Trader(config, client, storage, pos_manager)
        _bot_state._pos_manager_ref = pos_manager

        # Single shared Coinbase client for all price-dependent strategies
        async with BinanceClient() as binance:
            tasks = [
                asyncio.create_task(
                    market_refresh_loop(client), name="market-refresh"
                ),
                asyncio.create_task(
                    arb_loop(arb_detector, trader), name="arb"
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
                asyncio.create_task(
                    corr_loop(trader), name="corr"
                ),
                asyncio.create_task(
                    news_fade_loop(trader), name="fade"
                ),
                asyncio.create_task(
                    forecast_loop(trader), name="forecast"
                ),
                asyncio.create_task(
                    llm_loop(trader, binance), name="llm"
                ),
                asyncio.create_task(
                    weather_loop(trader), name="weather"
                ),
                asyncio.create_task(
                    position_loop(pos_manager), name="positions"
                ),
                asyncio.create_task(
                    web_server_loop(), name="web"
                ),
            ]

            stop_event = asyncio.Event()

            def _handle_signal():
                console.print("\n[yellow]Shutting down Leo...[/yellow]")
                stop_event.set()

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _handle_signal)

            try:
                with Live(console=console, refresh_per_second=1) as live:
                    ui_task = asyncio.create_task(
                        ui_loop(live, storage, pos_manager), name="ui"
                    )
                    await stop_event.wait()
            finally:
                for t in tasks:
                    t.cancel()
                ui_task.cancel()
                await asyncio.gather(
                    *tasks, ui_task, return_exceptions=True
                )
                console.print("[green]Leo stopped cleanly.[/]")


def main():
    console.print("[bold cyan]Starting Leo — Kalshi Trading Bot[/]")
    asyncio.run(run())


if __name__ == "__main__":
    main()
