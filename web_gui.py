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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Leo Dashboard", docs_url=None, redoc_url=None)

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

    arb_opps = [
        {
            "market_id": o.market_id,
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
            "question": o.question[:60],
            "source": o.source,
            "market_prob": round(o.market_prob * 100, 1),
            "model_prob": round(o.model_prob * 100, 1),
            "edge_pct": round(o.edge * 100, 2),
            "side": o.recommended_side,
            "size": round(o.recommended_size_usd, 2),
            "reasoning": getattr(o, "reasoning", ""),
        }
        for o in state.signal_opps[:20]
    ]

    cross_opps = [
        {
            "kalshi_id": o.kalshi_market_id,
            "ext_id": o.external_market_id,
            "question": o.question[:60],
            "buy_platform": o.buy_platform,
            "sell_platform": o.sell_platform,
            "buy_price": round(o.buy_price * 100, 1),
            "sell_price": round(o.sell_price * 100, 1),
            "net_pct": round(o.net_profit_pct * 100, 2),
            "signal_only": o.signal_only,
        }
        for o in state.cross_opps[:20]
    ]

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "dry_run": config.dry_run,
        "scans": {
            "arb": state.arb_scans,
            "mm": state.mm_scans,
            "corr": state.corr_scans,
            "signal": state.signal_scans,
            "cross": state.cross_scans,
            "whale": state.whale_signals,
        },
        "markets_count": len(state.markets),
        "arb_opps": arb_opps,
        "corr_opps": corr_opps,
        "signal_opps": signal_opps,
        "cross_opps": cross_opps,
    }


def _serialise_portfolio() -> dict:
    import bot_state
    pm = bot_state._pos_manager_ref
    if not pm or not pm.snapshot:
        return {}

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

    return {
        "balance": round(s.balance_usd, 2),
        "unrealized_pnl": round(s.unrealized_pnl, 2),
        "realized_pnl": round(s.realized_pnl, 2),
        "total_pnl": round(s.total_pnl, 2),
        "position_count": s.position_count,
        "positions": positions,
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
            "yes_price": row["yes_price"],
            "no_price": row["no_price"],
            "contracts": row["contracts"],
            "net_profit_pct": round(
                (row["net_profit_pct"] or 0) * 100, 2
            ),
            "dry_run": bool(row["dry_run"]),
            "status": row["status"],
            "created_at": row["created_at"],
        })
    return result


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/state")
async def api_state():
    return _serialise_state()


@app.get("/api/portfolio")
async def api_portfolio():
    return _serialise_portfolio()


@app.get("/api/trades")
async def api_trades(limit: int = 50):
    return _serialise_trades(limit)


@app.post("/api/control/dry-run")
async def toggle_dry_run(enabled: bool):
    try:
        import bot_state
        bot_state.config.dry_run = enabled
        return {"dry_run": enabled}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send initial full state immediately on connect
        await ws.send_text(
            json.dumps(
                {
                    **_serialise_state(),
                    "portfolio": _serialise_portfolio(),
                    "trades": _serialise_trades(50),
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
                # Send portfolio + trades every 5 ticks (5 seconds)
                if tick % 5 == 0:
                    payload["portfolio"] = _serialise_portfolio()
                if tick % 30 == 0:
                    payload["trades"] = _serialise_trades(50)
                await manager.broadcast(payload)
            tick += 1
        except Exception as e:
            logger.debug(f"Broadcast error: {e}")
        await asyncio.sleep(1)
