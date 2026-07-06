"""Tests for network connectivity tracking."""

import asyncio

import aiohttp
import bot_state
import network_status
from network_status import NetworkTracker, is_network_error, record_failure, record_success


def _reset():
    bot_state.network_online = True
    bot_state.network_status = {}
    t = network_status._tracker
    t.online = True
    t.degraded = False
    t.consecutive_failures = 0
    t.last_error = ""
    t.services.clear()
    t._alerted_offline = False


def test_is_network_error_detects_timeout():
    assert is_network_error(asyncio.TimeoutError())
    assert is_network_error(ConnectionError("connection refused"))


def test_is_network_error_ignores_http_status():
    assert not is_network_error(aiohttp.ClientResponseError(
        request_info=None, history=(), status=404, message="not found"
    ))


def test_tracker_goes_offline_after_repeated_failures():
    _reset()
    t = NetworkTracker()
    for _ in range(3):
        t.record_failure("gamma", ConnectionError("unreachable"), force=True)
    assert not t.online
    assert t.degraded


def test_tracker_recovers_on_success():
    _reset()
    t = NetworkTracker()
    t.record_failure("gamma", ConnectionError("down"), force=True)
    t.record_failure("gamma", ConnectionError("down"), force=True)
    t.record_failure("gamma", ConnectionError("down"), force=True)
    assert not t.online
    t.record_success("probe")
    assert t.online
    assert t.consecutive_failures == 0


def test_record_success_clears_bot_state_flag():
    _reset()
    network_status._tracker.online = False
    bot_state.network_online = False
    record_success("gamma")
    assert bot_state.network_online is True