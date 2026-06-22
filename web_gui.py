"""
Leo Web Dashboard — FastAPI + WebSocket

Serves a real-time dashboard at http://localhost:5002

Architecture:
  - FastAPI handles HTTP + WebSocket on a single port
  - A background broadcaster task pushes state to all connected clients
    every second via WebSocket JSON messages
  - REST endpoints handle control actions (start/pause/stop dry-run toggle)
  - The BotState object from main.py is imported directly (shared memory)

To run standalone (without the trading bot):
  uvicorn web_gui:app --host 0.0.0.0 --port 5002 --reload

To run embedded (normal use — main.py starts this automatically):
  python main.py   # web GUI starts on port 5002 alongside all bot loops
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from order_gate import update_trading_health
from runtime_config import save_runtime, snapshot_runtime
from web_auth import WebAuthMiddleware, verify_ws_token

logger = logging.getLogger(__name__)

app = FastAPI(title="Leo Dashboard", docs_url=None, redoc_url=None)
app.add_middleware(WebAuthMiddleware)

_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)
        logger.debug(
            f"WS client connected ({len(self._clients)} total)"
        )

    def disconnect(self, ws: WebSocket):
        try:
            self._clients.remove(ws)
        except ValueError:
            pass
        logger.debug(
            f"WS client disconnected ({len(self._clients)} total)"
        )

    async def broadcast(self, data: dict):
        dead = []
        msg = json.dumps(data, default=_json_default)
        for ws in self._clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                self._clients.remove(ws)
            except ValueError:
                pass


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# State serialisation
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _serialise_state() -> dict:
    """Build the JSON payload sent to every connected browser client."""
    import bot_state
    state = bot_state.state
    config = bot_state.config
    if state is None or config is None:
        return {"error": "bot not running"}

    slug_map: dict[str, str] = {
        m.market_id: m.slug
        for m in (state.markets or [])
        if m.slug
    }

    arb_opps = [
        {
            "market_id": o.market_id,
            "slug": slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "yes_bid": round(o.yes_bid, 3),
            "no_bid": round(o.no_bid, 3),
            "sum": round(o.yes_bid + o.no_bid, 4),
            "net_pct": round(o.net_profit_pct * 100, 2),
            "close_time": o.close_time,
            "hours_left": round(
                (o.close_time - datetime.now(timezone.utc)).total_seconds()
                / 3600, 1
            ),
        }
        for o in state.arb_opps[:20]
    ]

    corr_opps = [
        {
            "market_id": o.market_id_mispriced,
            "slug": slug_map.get(o.market_id_mispriced, ""),
            "anchor_id": o.market_id_anchor,
            "question": o.question_mispriced[:60],
            "relation": o.relation.value,
            "edge_pct": round(o.edge * 100, 2),
            "side": o.recommended_side,
            "max_size": round(o.max_size_usd, 2),
        }
        for o in state.corr_opps[:20]
    ]

    signal_opps = [
        {
            "market_id": o.market_id,
            "slug": slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "source": o.source,
            "market_prob": round(o.market_prob * 100, 1),
            "model_prob": round(o.model_prob * 100, 1),
            "edge_pct": round(o.edge * 100, 2),
            "side": o.recommended_side,
            "size": round(o.recommended_size_usd, 2),
            "reasoning": getattr(o, "reasoning", ""),
            "detected_at": getattr(o, "detected_at", None),
        }
        for o in state.signal_opps[:20]
    ]

    cross_opps = [
        {
            "market_id": o.polymarket_market_id,   # normalised key for detail modal
            "polymarket_id": o.polymarket_market_id,
            "ext_id": o.external_market_id,
            "slug": slug_map.get(o.polymarket_market_id, ""),
            "question": o.question[:60],
            "buy_platform": o.buy_platform,
            "sell_platform": o.sell_platform,
            "buy_price": round(o.buy_price * 100, 1),
            "sell_price": round(o.sell_price * 100, 1),
            "net_pct": round(o.net_profit_pct * 100, 2),
        }
        for o in state.cross_opps[:20]
    ]

    fav_opps = [
        {
            "market_id": o.market_id,
            "slug": slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "market_prob": round(o.market_prob * 100, 1),
            "model_prob": round(o.model_prob * 100, 1),
            "edge_pct": round(o.edge * 100, 2),
            "side": getattr(o, "recommended_side", "no"),
            "reasoning": getattr(o, "reasoning", ""),
            "detected_at": getattr(o, "detected_at", None),
        }
        for o in state.fav_opps[:20]
    ]

    squeeze_opps = [
        {
            "market_id": o.market_id,
            "slug": slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "market_prob": round(o.market_prob * 100, 1),
            "side": o.recommended_side,
            "edge_pct": round(o.edge * 100, 2),
            "reasoning": getattr(o, "reasoning", ""),
            "detected_at": getattr(o, "detected_at", None),
        }
        for o in state.squeeze_opps[:20]
    ]

    semarg_opps = [
        {
            "market_id": o.market_id,
            "slug": slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "poly_prob": round(o.market_prob * 100, 1),
            "ext_prob": round(o.model_prob * 100, 1),
            "side": o.recommended_side,
            "edge_pct": round(o.edge * 100, 2),
            "source": o.source,
            "detected_at": getattr(o, "detected_at", None),
        }
        for o in state.semarg_opps[:20]
    ]

    ofi_opps = [
        {
            "market_id": o.market_id,
            "slug": slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "side": o.recommended_side,
            "edge_pct": round(o.edge * 100, 2),
            "source": o.source,
            "reasoning": getattr(o, "reasoning", ""),
            "detected_at": getattr(o, "detected_at", None),
        }
        for o in state.ofi_opps[:20]
    ]

    btc5min_opps = [
        {
            "market_id": o.market_id,
            # slug comes from the signal itself (market not in volume-sorted state.markets)
            "slug": getattr(o, "slug", "") or slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "market_prob": round(o.market_prob * 100, 1),
            "model_prob": round(o.model_prob * 100, 1),
            "side": o.recommended_side,
            "edge_pct": round(o.edge * 100, 2),
            "reasoning": getattr(o, "reasoning", ""),
            "detected_at": getattr(o, "detected_at", None),
        }
        for o in state.btc5min_opps[:20]
    ]

    range_opps = [
        {
            "market_id": o.market_id,
            "slug": slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "source": getattr(o, "source", "range"),
            "market_prob": round(o.market_prob * 100, 1),
            "model_prob": round(o.model_prob * 100, 1),
            "edge_pct": round(o.edge * 100, 2),
            "side": o.recommended_side,
            "size": round(o.recommended_size_usd, 2),
            "reasoning": getattr(o, "reasoning", ""),
            "detected_at": getattr(o, "detected_at", None),
        }
        for o in state.range_opps[:20]
    ]

    fade_opps = [
        {
            "market_id": o.market_id,
            "slug": slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "market_prob": round(o.market_prob * 100, 1),
            "side": o.recommended_side,
            "edge_pct": round(o.edge * 100, 2),
            "reasoning": getattr(o, "reasoning", ""),
            "detected_at": getattr(o, "detected_at", None),
        }
        for o in state.fade_opps[:20]
    ]

    forecast_opps = [
        {
            "market_id": o.market_id,
            "slug": slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "source": getattr(o, "source", "forecast"),
            "market_prob": round(o.market_prob * 100, 1),
            "model_prob": round(o.model_prob * 100, 1),
            "edge_pct": round(o.edge * 100, 2),
            "side": o.recommended_side,
            "size": round(o.recommended_size_usd, 2),
            "reasoning": getattr(o, "reasoning", ""),
            "detected_at": getattr(o, "detected_at", None),
        }
        for o in state.forecast_opps[:20]
    ]

    llm_opps = [
        {
            "market_id": o.market_id,
            "slug": slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "market_prob": round(o.market_prob * 100, 1),
            "model_prob": round(o.model_prob * 100, 1),
            "edge_pct": round(o.edge * 100, 2),
            "side": o.recommended_side,
            "size": round(o.recommended_size_usd, 2),
            "confidence": round(getattr(o, "confidence", 0) * 100, 1),
            "reasoning": getattr(o, "reasoning", ""),
            "detected_at": getattr(o, "detected_at", None),
        }
        for o in state.llm_opps[:20]
    ]

    weather_opps = [
        {
            "market_id": o.market_id,
            "slug": slug_map.get(o.market_id, ""),
            "question": o.question[:60],
            "market_prob": round(o.market_prob * 100, 1),
            "model_prob": round(o.model_prob * 100, 1),
            "edge_pct": round(o.edge * 100, 2),
            "side": o.recommended_side,
            "size": round(o.recommended_size_usd, 2),
            "reasoning": getattr(o, "reasoning", ""),
            "detected_at": getattr(o, "detected_at", None),
        }
        for o in state.weather_opps[:20]
    ]

    # Best single opportunity across all strategy types
    all_edge_opps = (
        [(o["net_pct"],  "arb",      o) for o in arb_opps]
        + [(o["edge_pct"], "corr",    o) for o in corr_opps]
        + [(o["edge_pct"], "signal",  o) for o in signal_opps]
        + [(o["edge_pct"], "range",   o) for o in range_opps]
        + [(o["edge_pct"], "fade",    o) for o in fade_opps]
        + [(o["edge_pct"], "forecast",o) for o in forecast_opps]
        + [(o["edge_pct"], "llm",     o) for o in llm_opps]
        + [(o["edge_pct"], "weather", o) for o in weather_opps]
        + [(o["edge_pct"], "fav",     o) for o in fav_opps]
        + [(o["edge_pct"], "squeeze", o) for o in squeeze_opps]
        + [(o["edge_pct"], "semarg",  o) for o in semarg_opps]
        + [(o["edge_pct"], "ofi",     o) for o in ofi_opps]
        + [(o["edge_pct"], "btc5min", o) for o in btc5min_opps]
    )
    best_opp = None
    if all_edge_opps:
        _edge, _tag, _opp = max(all_edge_opps, key=lambda x: x[0])
        best_opp = {"strategy": _tag, **_opp}

    # Strategy enabled states (for controls page)
    strategy_states = {}
    for name in (
        "arbitrage", "crypto_signal", "range_straddle", "cross_arb",
        "correlated", "news_fade", "forecast", "llm", "weather",
        "favorite_short", "oracle_squeeze", "semantic_arb",
        "orderbook_momentum", "market_maker", "btc_5min",
    ):
        cfg = getattr(config, name, None)
        if cfg and hasattr(cfg, "enabled"):
            strategy_states[name] = cfg.enabled

    # Health indicators (recomputed on market refresh; refresh here if stale)
    health = getattr(bot_state, "last_health", None) or {}
    if not health:
        try:
            health = update_trading_health()
        except Exception:
            health = {}
    _client = getattr(bot_state, "_client_ref", None)
    _clob_errors = getattr(_client, "_clob_errors", 0) if _client else 0
    _clob_ok = health.get("clob_ok", True)
    _gamma_age = health.get("gamma_age_sec")

    trader = getattr(bot_state, "_trader_ref", None)
    exposure = {
        "today_usd":   round(trader._today_usd_deployed, 2) if trader else 0.0,
        "total_usd":   round(trader._total_exposure,     2) if trader else 0.0,
        "daily_limit": config.risk.max_daily_usd_deployed,
        "total_limit": config.arbitrage.max_total_exposure_usd,
    }
    risk_config = {
        "max_position_usd": config.arbitrage.max_position_usd,
        "max_daily_usd":    config.risk.max_daily_usd_deployed,
        "min_edge_pct":     round(config.arbitrage.min_profit_pct * 100, 2),
        "total_limit_usd":  config.arbitrage.max_total_exposure_usd,
    }

    # Signal staleness — seconds since last non-empty scan per strategy
    _now = datetime.now(timezone.utc)
    signal_ages: dict[str, Optional[int]] = {}
    for _k, _ts in (getattr(state, "last_signal_at", {}) or {}).items():
        try:
            _dt = datetime.fromisoformat(_ts)
            signal_ages[_k] = round((_now - _dt).total_seconds())
        except Exception:
            signal_ages[_k] = None

    # Confluence best opportunities
    _conf = getattr(bot_state, "_confluence_ref", None)
    confluence_opps = _conf.best_opps() if _conf else []

    mm_enabled = bool(
        getattr(config.market_maker, "enabled", False)
        if hasattr(config, "market_maker") else False
    )

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "dry_run": config.dry_run,
        "paused": getattr(bot_state, "paused", False),
        "pnl_source": (
            "resolved_trades_db" if config.dry_run else "position_manager_live"
        ),
        "strategy_audit": getattr(bot_state, "strategy_audit", []),
        "mm_sim_mode": config.dry_run and mm_enabled,
        "health": {
            "clob_ok": _clob_ok,
            "clob_errors": _clob_errors,
            "gamma_age_sec": _gamma_age,
            "trading_blocked": health.get("trading_blocked", False),
            "block_reason": health.get("block_reason", ""),
            "markets_count": health.get("markets_count", len(state.markets)),
        },
        "btc5min_signals": getattr(bot_state, "_btc5min_signals", {}),
        "signal_ages": signal_ages,
        "confluence_opps": confluence_opps,
        "strategy_states": strategy_states,
        "exposure": exposure,
        "risk_config": risk_config,
        "scans": {
            "arb":      state.arb_scans,
            "mm":       state.mm_scans,
            "mm_active": state.mm_active,
            "corr":     state.corr_scans,
            "signal":   state.signal_scans,
            "crypto":   state.crypto_scans,
            "cross":    state.cross_scans,
            "range":    state.range_scans,
            "fade":     state.fade_scans,
            "forecast": state.forecast_scans,
            "llm":      state.llm_scans,
            "weather":  state.weather_scans,
            "whale":    state.whale_signals,
            "fav":      state.fav_scans,
            "squeeze":  state.squeeze_scans,
            "semarg":   state.semarg_scans,
            "ofi":      state.ofi_scans,
            "btc5min":  state.btc5min_scans,
        },
        "markets_count": len(state.markets),
        "arb_opps": arb_opps,
        "corr_opps": corr_opps,
        "signal_opps": signal_opps,
        "range_opps": range_opps,
        "fade_opps": fade_opps,
        "forecast_opps": forecast_opps,
        "llm_opps": llm_opps,
        "weather_opps": weather_opps,
        "cross_opps": cross_opps,
        "fav_opps": fav_opps,
        "squeeze_opps": squeeze_opps,
        "semarg_opps": semarg_opps,
        "ofi_opps": ofi_opps,
        "btc5min_opps": btc5min_opps,
        "mm_quotes": getattr(state, "mm_quotes", [])[:20],
        "best_opp": best_opp,
    }


def _serialise_slugs() -> dict:
    """Map condition_id → event slug for all loaded markets."""
    import bot_state
    state = bot_state.state
    if not state or not state.markets:
        return {}
    return {m.market_id: m.slug for m in state.markets if m.slug}


def _serialise_portfolio() -> dict:
    import bot_state
    pm      = bot_state._pos_manager_ref
    cfg     = bot_state.config
    storage = bot_state._storage_ref

    is_dry_run    = cfg and cfg.dry_run
    paper_bankroll = cfg.risk.paper_bankroll if cfg else 1000.0

    # Live path — use real position snapshot
    if pm and pm.snapshot and not is_dry_run:
        s = pm.snapshot
        positions = [
            {
                "market_id": p.market_id,
                "question": p.question[:60],
                "side": p.side,
                "contracts": p.contracts,
                "avg_price": round(p.avg_price * 100, 1),
                "current_mid": round(p.current_mid * 100, 1),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "status": p.status,
                "hours_left": round(
                    (p.close_time - datetime.now(timezone.utc)).total_seconds()
                    / 3600, 1
                ),
            }
            for p in s.open_positions[:30]
        ]
        balance = s.balance_usd if s.balance_usd > 0 else paper_bankroll
        return {
            "balance": round(balance, 2),
            "unrealized_pnl": round(s.unrealized_pnl, 2),
            "realized_pnl": round(s.realized_pnl, 2),
            "total_pnl": round(s.total_pnl, 2),
            "position_count": s.position_count,
            "positions": positions,
            "paper": False,
            "pnl_source": "position_manager_live",
        }

    # Paper / dry-run path — derive P&L from simulated trades in DB
    est_pnl = 0.0
    trade_count = 0
    pending = 0
    resolved = 0
    if storage:
        summary = storage.get_pnl_summary()
        est_pnl     = round(summary.get("estimated_pnl_usd") or 0.0, 2)
        trade_count = summary.get("total_trades") or 0
        with storage._connect() as conn:
            pending = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE outcome = 'pending'"
            ).fetchone()[0]
            resolved = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE outcome IN ('won','lost')"
            ).fetchone()[0]

    return {
        "balance": round(paper_bankroll + est_pnl, 2),
        "unrealized_pnl": 0.0,
        "realized_pnl": est_pnl,
        "total_pnl": est_pnl,
        "position_count": trade_count,
        "pending_trades": pending,
        "resolved_trades": resolved,
        "positions": [],
        "paper": True,
        "pnl_source": "resolved_trades_db",
    }


def _serialise_trades(limit: int = 50) -> list:
    import bot_state
    storage = bot_state._storage_ref
    if not storage:
        return []
    rows = storage.get_recent_trades(limit)
    result = []
    for row in rows:
        result.append({
            "id": row["id"],
            "market_id": row["market_id"],
            "question": (row["question"] or "")[:60],
            "arb_type": row["arb_type"],
            "side": row["side"],
            "yes_price": row["yes_price"],
            "no_price": row["no_price"],
            "contracts": row["contracts"],
            "size_usd": row["size_usd"],
            "net_profit_pct": round(
                (row["net_profit_pct"] or 0) * 100, 2
            ),
            "dry_run": bool(row["dry_run"]),
            "status": row["status"],
            "outcome": row["outcome"],
            "created_at": row["created_at"],
        })
    return result


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(), headers=_NO_CACHE)


@app.get("/api/state")
async def api_state():
    return _serialise_state()


@app.get("/api/portfolio")
async def api_portfolio():
    return _serialise_portfolio()


@app.get("/api/trades")
async def api_trades(limit: int = 50):
    return _serialise_trades(limit)


@app.get("/api/pnl-history")
async def api_pnl_history(hours: int = 24):
    import bot_state
    storage = bot_state._storage_ref
    if not storage:
        return []
    real = storage.get_pnl_history(hours)
    # Fall back to cumulative estimated P&L from trades when position manager
    # returns all-zero snapshots (dry-run mode / no live Polymarket positions).
    if not real or all(r.get("total_pnl", 0) == 0 for r in real):
        return storage.get_estimated_pnl_history(hours)
    return real


@app.get("/api/strategy-pnl")
async def api_strategy_pnl():
    import bot_state
    storage = bot_state._storage_ref
    if not storage:
        return []
    return storage.get_strategy_pnl()


@app.get("/api/win-rates")
async def api_win_rates():
    import bot_state
    storage = bot_state._storage_ref
    if not storage:
        return {}
    return storage.get_strategy_win_rates()


@app.get("/api/calibration")
async def api_calibration():
    import bot_state
    storage = bot_state._storage_ref
    if not storage:
        return {"strategies": [], "summary": {}}
    return storage.get_calibration_report()


@app.get("/api/runtime")
async def api_runtime():
    import bot_state
    if bot_state.config is None:
        raise HTTPException(503, "Bot not running")
    return snapshot_runtime(bot_state.config)


@app.get("/api/evolution")
async def api_evolution():
    import bot_state
    agent = getattr(bot_state, "_evolution_ref", None)
    if not agent:
        return {"recommendation": None, "last_review": None}
    return {
        "recommendation": agent.last_recommendation,
        "last_review": agent.last_review_ts,
    }


@app.get("/api/kyle-lambda")
async def api_kyle_lambda():
    import bot_state
    tracker = getattr(bot_state, "_kyle_lambda_ref", None)
    if not tracker:
        return []
    return tracker.summary(getattr(bot_state.state, "markets", []))


@app.get("/api/hurst")
async def api_hurst():
    import bot_state
    tracker = getattr(bot_state, "_hurst_ref", None)
    if not tracker:
        return []
    return tracker.summary()


@app.get("/api/confluence")
async def api_confluence():
    import bot_state
    tracker = getattr(bot_state, "_confluence_ref", None)
    if not tracker:
        return []
    return tracker.best_opps()


class _BoolBody(BaseModel):
    enabled: bool


@app.post("/api/control/dry-run")
async def toggle_dry_run(body: _BoolBody):
    try:
        import bot_state
        bot_state.config.dry_run = body.enabled
        save_runtime(dry_run=body.enabled)
        return {"dry_run": body.enabled}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/control/pause")
async def pause_bot():
    import bot_state
    bot_state.paused = True
    ev = getattr(bot_state, "_resume_event", None)
    if ev is not None:
        ev.clear()
    return {"paused": True}


@app.post("/api/control/resume")
async def resume_bot():
    import bot_state
    bot_state.paused = False
    ev = getattr(bot_state, "_resume_event", None)
    if ev is not None:
        ev.set()
    return {"paused": False}


@app.post("/api/control/stop")
async def stop_bot():
    import bot_state
    ev = getattr(bot_state, "_stop_event", None)
    if ev is None:
        raise HTTPException(503, "Stop event not initialised — bot may not be running")
    ev.set()
    return {"stopping": True}


@app.post("/api/control/strategy/{name}")
async def toggle_strategy(name: str, body: _BoolBody):
    import bot_state
    if bot_state.config is None:
        raise HTTPException(503, "Bot not running")
    cfg = getattr(bot_state.config, name, None)
    if cfg is None:
        raise HTTPException(404, f"Unknown strategy: {name}")
    if not hasattr(cfg, "enabled"):
        raise HTTPException(400, f"Strategy '{name}' has no enabled flag")
    cfg.enabled = body.enabled
    save_runtime(strategy=(name, body.enabled))
    return {"strategy": name, "enabled": body.enabled}


@app.post("/api/control/refresh-markets")
async def refresh_markets():
    import bot_state
    bot_state._force_market_refresh = True
    return {"triggered": True}


@app.post("/api/control/clear-cooldowns")
async def clear_cooldowns():
    import bot_state
    trader = getattr(bot_state, "_trader_ref", None)
    if not trader:
        raise HTTPException(503, "Trader not initialised")
    count = len(trader._cooldowns)
    trader._cooldowns.clear()
    return {"cleared": count}


class _RiskBody(BaseModel):
    max_position_usd: Optional[float] = None
    max_daily_usd: Optional[float] = None
    min_edge_pct: Optional[float] = None
    total_limit_usd: Optional[float] = None
    per_strategy_max_usd: Optional[float] = None


@app.post("/api/control/risk")
async def update_risk(body: _RiskBody):
    import bot_state
    if bot_state.config is None:
        raise HTTPException(503, "Bot not running")
    cfg = bot_state.config
    if body.max_position_usd is not None:
        for attr in (
            "arbitrage", "crypto_signal", "range_straddle", "news_fade",
            "forecast", "llm", "weather", "favorite_short", "oracle_squeeze",
            "semantic_arb", "orderbook_momentum", "btc_5min",
        ):
            sub = getattr(cfg, attr, None)
            if sub and hasattr(sub, "max_position_usd"):
                sub.max_position_usd = body.max_position_usd
    if body.max_daily_usd is not None:
        cfg.risk.max_daily_usd_deployed = body.max_daily_usd
    if body.min_edge_pct is not None:
        edge = body.min_edge_pct / 100.0
        cfg.arbitrage.min_profit_pct = edge
        for attr in (
            "correlated", "crypto_signal", "range_straddle", "news_fade",
            "forecast", "llm", "weather", "favorite_short", "oracle_squeeze",
            "orderbook_momentum", "btc_5min",
        ):
            sub = getattr(cfg, attr, None)
            if sub and hasattr(sub, "min_edge"):
                sub.min_edge = edge
        if hasattr(cfg, "cross_arb") and hasattr(cfg.cross_arb, "min_profit_pct"):
            cfg.cross_arb.min_profit_pct = edge
        # SemanticArbConfig uses min_price_gap instead of min_edge
        if hasattr(cfg, "semantic_arb") and hasattr(cfg.semantic_arb, "min_price_gap"):
            cfg.semantic_arb.min_price_gap = edge
    if body.total_limit_usd is not None:
        cfg.arbitrage.max_total_exposure_usd = body.total_limit_usd
    if body.per_strategy_max_usd is not None:
        cfg.risk.per_strategy_max_usd = body.per_strategy_max_usd
    risk_patch = {}
    if body.max_daily_usd is not None:
        risk_patch["max_daily_usd"] = body.max_daily_usd
    if body.max_position_usd is not None:
        risk_patch["max_position_usd"] = body.max_position_usd
    if body.total_limit_usd is not None:
        risk_patch["max_total_exposure_usd"] = body.total_limit_usd
    if body.min_edge_pct is not None:
        risk_patch["min_edge_pct"] = body.min_edge_pct
    if body.per_strategy_max_usd is not None:
        risk_patch["per_strategy_max_usd"] = body.per_strategy_max_usd
    if risk_patch:
        save_runtime(risk=risk_patch)
    return {
        "max_position_usd": cfg.arbitrage.max_position_usd,
        "max_daily_usd":    cfg.risk.max_daily_usd_deployed,
        "min_edge_pct":     round(cfg.arbitrage.min_profit_pct * 100, 2),
        "total_limit_usd":  cfg.arbitrage.max_total_exposure_usd,
        "per_strategy_max_usd": cfg.risk.per_strategy_max_usd,
    }


@app.get("/api/health")
async def api_health():
    import bot_state
    health = update_trading_health()
    client = getattr(bot_state, "_client_ref", None)
    clob_errors = getattr(client, "_clob_errors", 0) if client else 0
    gamma_at = getattr(bot_state, "_last_gamma_at", "") or ""
    return {
        "clob_ok": health.get("clob_ok", True),
        "clob_errors": clob_errors,
        "gamma_age_sec": health.get("gamma_age_sec"),
        "gamma_at": gamma_at or None,
        "trading_blocked": health.get("trading_blocked", False),
        "block_reason": health.get("block_reason", ""),
        "btc5min_signals": getattr(bot_state, "_btc5min_signals", {}),
    }


@app.get("/api/cooldowns")
async def api_cooldowns():
    import bot_state
    from datetime import timedelta
    trader = getattr(bot_state, "_trader_ref", None)
    if not trader:
        return []
    now = datetime.now(timezone.utc)
    cooldown_mins = 30  # mirrors _COOLDOWN_MINUTES in trader.py
    result = []
    m_map = {}
    if bot_state.state:
        m_map = getattr(bot_state.state, "extended_market_map", {}) or \
                getattr(bot_state.state, "market_map", {})
    for market_id, started_at in list(getattr(trader, "_cooldowns", {}).items()):
        ends_at = started_at + timedelta(minutes=cooldown_mins)
        secs_left = max(0, round((ends_at - now).total_seconds()))
        if secs_left == 0:
            continue
        m = m_map.get(market_id)
        result.append({
            "market_id": market_id,
            "question": (m.question[:60] if m else ""),
            "secs_left": secs_left,
            "ends_at": ends_at.isoformat(),
        })
    result.sort(key=lambda x: x["secs_left"])
    return result


@app.get("/api/log-feed")
async def api_log_feed(n: int = 50):
    import bot_state
    buf = getattr(bot_state, "_log_buffer", [])
    return list(buf)[-n:]


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if not await verify_ws_token(ws):
        return
    await manager.connect(ws)
    try:
        # Send initial full state immediately on connect
        await ws.send_text(
            json.dumps(
                {
                    **_serialise_state(),
                    "portfolio": _serialise_portfolio(),
                    "trades": _serialise_trades(50),
                    "market_slugs": _serialise_slugs(),
                },
                default=_json_default,
            )
        )
        # Keep connection alive — broadcast loop handles updates
        while True:
            # We don't expect messages from the client, but we must
            # await something to detect disconnection
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Broadcast task — called from main.py
# ---------------------------------------------------------------------------

async def broadcast_loop():
    """
    Push state updates to all connected WebSocket clients every second.
    Must run as an asyncio task alongside the bot loops.
    """
    tick = 0
    while True:
        try:
            if manager._clients:
                payload = _serialise_state()
                # Portfolio + trades every 2s so trade log / positions feel live
                if tick % 2 == 0:
                    payload["portfolio"] = _serialise_portfolio()
                    payload["trades"] = _serialise_trades(40)
                if tick % 15 == 0:
                    payload["market_slugs"] = _serialise_slugs()
                if tick % 30 == 0:
                    import bot_state as _bs30
                    _kl = getattr(_bs30, "_kyle_lambda_ref", None)
                    payload["kyle_lambda"] = (
                        _kl.summary(getattr(_bs30.state, "markets", []))
                        if _kl else []
                    )
                    _hu = getattr(_bs30, "_hurst_ref", None)
                    payload["hurst"] = _hu.summary() if _hu else []
                await manager.broadcast(payload)
            tick += 1
        except Exception as e:
            logger.debug(f"Broadcast error: {e}")
        await asyncio.sleep(1)
