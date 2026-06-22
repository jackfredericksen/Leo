"""
Async strategy and infrastructure loops for Leo.

Extracted from main.py — each loop runs independently via asyncio.create_task.
"""

import asyncio
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.live import Live

import bot_state as _bot_state
from bot_state import state
from alerting import Alerter
from api_clients.binance_client import BinanceClient
from api_clients.forecast_client import ForecastClient
from api_clients.llm_client import LLMClient
from api_clients.news_client import NewsClient
from api_clients.polymarket_client import PolymarketClient
from api_clients.pyth_client import PythClient
from api_clients.weather_client import WeatherClient
from arbitrage import ArbitrageDetector
from auto_correlator import AutoCorrelatorConfig, build_correlated_detector
from config import config
from position_manager import PositionManager
from storage import Storage
from strategies.btc_5min import BTC5MinDetector
from strategies.correlated import CorrelatedConfig
from strategies.cross_platform import (
    CrossPlatformConfig,
    CrossPlatformDetector,
    KalshiSignalClient,
)
from strategies.crypto_signal import CryptoSignalDetector
from strategies.evolution import EvolutionAgent
from strategies.exit_strategy import ExitStrategy
from strategies.favorite_short import FavoriteShortDetector
from strategies.forecast_signal import ForecastSignalDetector
from strategies.hurst import HurstTracker
from strategies.kyle_lambda import KyleLambdaTracker
from strategies.llm_signal import LLMSignalDetector
from strategies.market_maker import MarketMakerStrategy
from strategies.news_fade import NewsFadeConfig, NewsFadeDetector
from strategies.oracle_squeeze import OracleSqueezeDetector
from strategies.orderbook_momentum import OrderbookMomentumDetector
from strategies.range_straddle import RangeStraddleDetector
from strategies.semantic_arb import SemanticArbDetector
from strategies.signal_arb import AggregatedSignal, SignalArbConfig
from strategies.weather_signal import WeatherSignalDetector
from terminal_ui import build_ui
from trader import Trader
from order_gate import update_trading_health
from web_auth import web_host

logger = logging.getLogger("leo.loops")
console = Console()

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
            # Feed current mid-prices into the Hurst regime tracker
            hurst = _bot_state._hurst_ref
            if hurst:
                for m in markets:
                    if m.yes_price > 0:
                        hurst.update(m.market_id, m.yes_price)
        except Exception as e:
            logger.error(f"Market refresh: {e}")
        finally:
            _bot_state._force_market_refresh = False
            try:
                update_trading_health()
            except Exception:
                pass
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



async def range_straddle_loop(binance: BinanceClient, trader: Trader):
    """Strategy: Crypto range bracket pricing vs log-normal model."""
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
            await binance.refresh_all()
            if state.markets:
                state.range_opps = detector.scan(state.markets)
                state.range_scans += 1
                if state.range_opps:
                    state.last_signal_at["range_straddle"] = _now_iso()
                if state.range_scans % _BANKROLL_REFRESH_EVERY == 0:
                    await _refresh_bankroll(trader, detector.sizer)
                for sig in state.range_opps:
                    await trader.execute_signal(sig, state.market_map)
                if state.range_opps:
                    logger.info(
                        f"RangeStraddle: {len(state.range_opps)} opps, "
                        f"top edge={state.range_opps[0].edge:.1%}"
                    )
        except Exception as e:
            logger.error(f"Range straddle loop: {e}")
        await asyncio.sleep(config.range_straddle.refresh_interval_sec)


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


async def pyth_stream_loop(pyth: PythClient):
    """Maintain Pyth SSE price stream. Reconnects automatically — runs forever."""
    await pyth.stream()


async def btc_5min_scan_loop(detector: BTC5MinDetector, binance: BinanceClient, trader: Trader):
    """Slow scan (every cfg.refresh_interval_sec): candle fetch + market discovery + full B-S signals."""
    cfg = config.btc_5min
    while True:
        await _await_resume()
        if not cfg.enabled:
            await asyncio.sleep(10)
            continue
        try:
            # 5-min markets sit at offset ~3400+ in the volume-sorted list.
            # Fetch recent markets (sorted by startDate desc) so new 5-min windows
            # are detected within the 0.2–1.5 min entry window.
            await binance.fetch_candles("BTCUSDT", limit=60)
            await binance.fetch_snapshots()
            recent = await trader.client.get_recent_markets(limit=200, ttl_sec=0)
            seen_ids = {m.market_id for m in recent}
            combined = recent + [m for m in state.markets if m.market_id not in seen_ids]
            state.btc5min_opps = await detector.scan(combined)
            _bot_state._btc5min_signals = {**detector._last_signals}
            state.btc5min_scans += 1
            if state.btc5min_opps:
                state.last_signal_at["btc_5min"] = _now_iso()
            if state.btc5min_scans % _BANKROLL_REFRESH_EVERY == 0:
                await _refresh_bankroll(trader, detector.sizer)
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
            logger.error(f"BTC 5-min scan: {e}")
        await asyncio.sleep(cfg.refresh_interval_sec)


async def btc_5min_fast_loop(detector: BTC5MinDetector, trader: Trader):
    """Fast re-evaluation (2s): fresh Pyth price + cached slow signals, no I/O."""
    cfg = config.btc_5min
    while True:
        await _await_resume()
        if not cfg.enabled:
            await asyncio.sleep(2)
            continue
        try:
            fast_opps = detector.fast_scan()
            if fast_opps:
                m_map = state.extended_market_map or state.market_map
                for sig in fast_opps:
                    await trader.execute_signal(sig, m_map)
                logger.debug(
                    f"BTC5Min fast: {len(fast_opps)} opps, "
                    f"top edge={fast_opps[0].edge:.1%}"
                )
        except Exception as e:
            logger.error(f"BTC 5-min fast: {e}")
        await asyncio.sleep(2)


async def market_maker_loop(client: PolymarketClient):
    """Strategy: Resting limit orders on both sides + CLOB v2 maker rebates."""
    mm = MarketMakerStrategy(config.market_maker, client)
    try:
        await mm.cancel_all()
        logger.info("MM: cleared any orphan resting orders from prior session")
    except Exception as e:
        logger.debug(f"MM startup cancel: {e}")
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
                m_map = state.extended_market_map if state.extended_market_map else state.market_map
                exits = await strategy.scan_and_exit(
                    snap.open_positions, m_map, dry_run=config.dry_run
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
                question_label = row["question"] or row["market_id"]
                if our_side == "both":
                    # Overround arb: we hold YES + NO; guaranteed $1 payout at settlement
                    storage.update_trade_outcome(row["id"], "won")
                    logger.info(
                        f"Outcome: arb trade {row['id']} → won "
                        f"(market settled, both legs pay)"
                    )
                    if alerter:
                        _alert_task_standalone(alerter.outcome_resolved(
                            row["id"], question_label, "won", row["arb_type"] or "arb"
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
                        row["id"], question_label, outcome, row["arb_type"] or "signal"
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
        host = web_host()
        server_cfg = uvicorn.Config(
            web_app,
            host=host,
            port=web_port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(server_cfg)
        bind = "localhost" if host in ("127.0.0.1", "localhost") else host
        console.print(
            f"[bold cyan]Web dashboard:[/] http://{bind}:{web_port}"
        )
        await asyncio.gather(server.serve(), broadcast_loop())
    except ImportError:
        logger.warning("uvicorn not installed — web GUI disabled.")
    except Exception as e:
        logger.error(f"Web server error: {e}")



def create_bot_tasks(
    *,
    client: PolymarketClient,
    trader: Trader,
    storage: Storage,
    pos_manager: PositionManager,
    arb_detector: ArbitrageDetector,
    binance: BinanceClient,
    pyth: PythClient,
    btc_detector: BTC5MinDetector,
    news_client: NewsClient,
    alerter: Alerter,
    llm_for_evolution=None,
) -> list[asyncio.Task]:
    """Build the full asyncio task list for the running bot."""
    tasks = [
        asyncio.create_task(market_refresh_loop(client), name="market-refresh"),
        asyncio.create_task(arb_loop(arb_detector, trader, client), name="arb"),
        asyncio.create_task(crypto_signal_loop(binance, trader), name="crypto"),
        asyncio.create_task(range_straddle_loop(binance, trader), name="range"),
        asyncio.create_task(cross_platform_loop(trader), name="cross"),
        asyncio.create_task(corr_loop(trader), name="corr"),
        asyncio.create_task(news_fade_loop(trader, news_client), name="fade"),
        asyncio.create_task(forecast_loop(trader), name="forecast"),
        asyncio.create_task(llm_loop(trader, binance), name="llm"),
        asyncio.create_task(weather_loop(trader), name="weather"),
        asyncio.create_task(favorite_short_loop(trader), name="fav_short"),
        asyncio.create_task(oracle_squeeze_loop(trader), name="oracle_squeeze"),
        asyncio.create_task(semantic_arb_loop(trader), name="semantic_arb"),
        asyncio.create_task(orderbook_momentum_loop(client, trader), name="ofi"),
        asyncio.create_task(pyth_stream_loop(pyth), name="pyth-stream"),
        asyncio.create_task(btc_5min_scan_loop(btc_detector, binance, trader), name="btc_5min_scan"),
        asyncio.create_task(btc_5min_fast_loop(btc_detector, trader), name="btc_5min_fast"),
        asyncio.create_task(market_maker_loop(client), name="market_maker"),
        asyncio.create_task(position_loop(pos_manager), name="positions"),
        asyncio.create_task(pnl_snapshot_loop(storage, pos_manager), name="pnl-snapshot"),
        asyncio.create_task(exit_loop(client, pos_manager, storage), name="exit"),
        asyncio.create_task(kyle_lambda_loop(client), name="kyle-lambda"),
        asyncio.create_task(outcome_tracker_loop(storage, pos_manager, alerter), name="outcome-tracker"),
        asyncio.create_task(web_server_loop(), name="web"),
    ]
    if llm_for_evolution is not None:
        tasks.append(asyncio.create_task(
            evolution_loop(storage, llm_for_evolution), name="evolution"
        ))
    tasks.append(asyncio.create_task(
        daily_summary_loop(alerter, storage, pos_manager), name="daily-summary"
    ))
    return tasks
