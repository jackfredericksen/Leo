"""
Central trading safety gate — single place for pause, health, and dry-run checks.

All live CLOB placement (Trader, MM, exit) must pass live_orders_allowed().
Strategy sim/live execution must pass trading_allowed() first.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import bot_state

logger = logging.getLogger(__name__)

_GAMMA_STALE_SEC = 120
_MIN_MARKETS = 1


def _paused() -> bool:
    if getattr(bot_state, "paused", False):
        return True
    ev = getattr(bot_state, "_resume_event", None)
    return ev is not None and not ev.is_set()


def set_health_block(reason: str) -> None:
    bot_state.health_block_trading = True
    bot_state.health_block_reason = reason


def clear_health_block() -> None:
    bot_state.health_block_trading = False
    bot_state.health_block_reason = ""


def update_trading_health() -> dict:
    """
    Recompute health flags from client + market state.
    Returns a dict suitable for dashboard serialisation.
    """
    client = getattr(bot_state, "_client_ref", None)
    state = getattr(bot_state, "state", None)
    reasons: list[str] = []

    clob_ok = True
    circuit_open = False
    if client:
        circuit_open = client.circuit_open
        clob_ok = not circuit_open
        if circuit_open:
            reasons.append("CLOB circuit breaker open")

    gamma_age: Optional[int] = None
    gamma_at = getattr(bot_state, "_last_gamma_at", "") or ""
    if gamma_at:
        try:
            gdt = datetime.fromisoformat(gamma_at)
            gamma_age = int((datetime.now(timezone.utc) - gdt).total_seconds())
            if gamma_age > _GAMMA_STALE_SEC:
                reasons.append(f"Gamma stale ({gamma_age}s)")
        except Exception:
            reasons.append("Gamma timestamp invalid")
    else:
        reasons.append("Gamma never refreshed")

    markets_count = len(state.markets) if state and state.markets else 0
    if markets_count < _MIN_MARKETS:
        reasons.append("No markets loaded")

    if reasons:
        set_health_block(reasons[0])
    else:
        clear_health_block()

    health = {
        "clob_ok": clob_ok,
        "circuit_open": circuit_open,
        "gamma_age_sec": gamma_age,
        "markets_count": markets_count,
        "trading_blocked": bot_state.health_block_trading,
        "block_reason": bot_state.health_block_reason,
    }
    bot_state.last_health = health
    return health


def trading_allowed() -> tuple[bool, str]:
    """Block strategy execution (sim + live) when paused or unhealthy."""
    if _paused():
        return False, "bot paused"
    if getattr(bot_state, "health_block_trading", False):
        return False, bot_state.health_block_reason or "health block"
    return True, ""


def live_orders_allowed() -> tuple[bool, str]:
    """Block real CLOB order placement."""
    cfg = getattr(bot_state, "config", None)
    if cfg and cfg.dry_run:
        return False, "dry_run"
    ok, reason = trading_allowed()
    if not ok:
        return False, reason
    client = getattr(bot_state, "_client_ref", None)
    if client and client.circuit_open:
        return False, "CLOB circuit breaker open"
    if not cfg or not cfg.polymarket.private_key:
        return False, "no private key"
    return True, ""


def can_place_live_orders() -> tuple[bool, str]:
    """Alias used by premortem docs / external callers."""
    return live_orders_allowed()


def on_circuit_breaker_open(error_count: int, pause_sec: int) -> None:
    """Called from PolymarketClient when the CLOB circuit opens."""
    set_health_block(f"CLOB circuit breaker open ({error_count} errors)")
    alerter = getattr(bot_state, "_alerter_ref", None)
    if alerter:
        try:
            asyncio.get_running_loop().create_task(
                alerter.circuit_breaker_open(error_count, pause_sec)
            )
        except RuntimeError:
            pass
    logger.warning(
        "Trading halted: CLOB circuit breaker open for %ss", pause_sec
    )