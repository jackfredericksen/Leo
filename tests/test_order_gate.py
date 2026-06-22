"""Tests for order_gate trading safety checks."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import bot_state
import order_gate


def _reset_bot_state():
    bot_state.paused = False
    bot_state.health_block_trading = False
    bot_state.health_block_reason = ""
    bot_state._resume_event = None
    bot_state._client_ref = None
    bot_state.config = None
    bot_state.state.markets = []
    bot_state._last_gamma_at = ""


def test_trading_blocked_when_paused():
    _reset_bot_state()
    bot_state.paused = True
    ok, reason = order_gate.trading_allowed()
    assert not ok
    assert "paused" in reason


def test_live_orders_blocked_in_dry_run():
    _reset_bot_state()
    cfg = MagicMock()
    cfg.dry_run = True
    cfg.polymarket.private_key = "0xabc"
    bot_state.config = cfg
    ok, reason = order_gate.live_orders_allowed()
    assert not ok
    assert reason == "dry_run"


def test_health_block_on_stale_gamma():
    _reset_bot_state()
    bot_state.config = MagicMock(dry_run=True)
    bot_state.state.markets = [MagicMock()]
    old = datetime.now(timezone.utc).replace(year=2020).isoformat()
    bot_state._last_gamma_at = old
    health = order_gate.update_trading_health()
    assert health["trading_blocked"]
    assert "Gamma stale" in health["block_reason"]


def test_circuit_breaker_sets_health_block():
    _reset_bot_state()
    client = MagicMock()
    client.circuit_open = True
    bot_state._client_ref = client
    bot_state.config = MagicMock(dry_run=False, polymarket=MagicMock(private_key="k"))
    bot_state.state.markets = [MagicMock()]
    bot_state._last_gamma_at = datetime.now(timezone.utc).isoformat()

    ok, reason = order_gate.live_orders_allowed()
    assert not ok
    assert "circuit" in reason.lower()


def test_on_circuit_breaker_open_fires_alert():
    _reset_bot_state()
    alerter = MagicMock()
    alerter.circuit_breaker_open = MagicMock(
        return_value=asyncio.sleep(0, result=None)
    )
    bot_state._alerter_ref = alerter

    async def _run():
        order_gate.on_circuit_breaker_open(5, 120)
        await asyncio.sleep(0.05)

    asyncio.run(_run())
    alerter.circuit_breaker_open.assert_called_once_with(5, 120)