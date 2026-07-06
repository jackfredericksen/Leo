"""
Network connectivity tracking — graceful offline / reconnect handling.

Detects transport-level failures (DNS, timeouts, connection refused), pauses
new trades while offline, keeps serving cached market data, and auto-resumes
when probes succeed again.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import aiohttp

import bot_state

logger = logging.getLogger(__name__)

_OFFLINE_AFTER_FAILURES = int(os.getenv("NETWORK_OFFLINE_THRESHOLD", "3"))
_PROBE_INTERVAL_OFFLINE_SEC = int(os.getenv("NETWORK_PROBE_INTERVAL_SEC", "20"))
_PROBE_INTERVAL_ONLINE_SEC = int(os.getenv("NETWORK_PROBE_OK_INTERVAL_SEC", "90"))

_PROBE_URLS = (
    "https://gamma-api.polymarket.com/markets?limit=1",
    "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD",
)

_NETWORK_EXCEPTIONS = (
    aiohttp.ClientConnectorError,
    aiohttp.ClientOSError,
    aiohttp.ServerDisconnectedError,
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    BrokenPipeError,
    OSError,
)


def is_network_error(exc: BaseException) -> bool:
    """True for transport/DNS/timeout failures (not HTTP 4xx/5xx from a reachable server)."""
    if isinstance(exc, _NETWORK_EXCEPTIONS):
        return True
    if isinstance(exc, aiohttp.ClientConnectorError):
        return True
    if isinstance(exc, aiohttp.ServerDisconnectedError):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return False
    msg = str(exc).lower()
    needles = (
        "cannot connect",
        "connection reset",
        "connection refused",
        "nodename nor servname",
        "name or service not known",
        "network is unreachable",
        "no route to host",
        "timed out",
        "timeout",
        "temporary failure in name resolution",
        "ssl:",
        "broken pipe",
        "disconnected",
    )
    return any(n in msg for n in needles)


class NetworkTracker:
    def __init__(self) -> None:
        self.online: bool = True
        self.degraded: bool = False
        self.consecutive_failures: int = 0
        self.last_success_at: Optional[str] = None
        self.last_failure_at: Optional[str] = None
        self.last_error: str = ""
        self.services: dict[str, dict] = {}
        self._alerted_offline: bool = False

    def record_success(self, service: str = "api") -> None:
        now = datetime.now(timezone.utc).isoformat()
        was_offline = not self.online
        self.consecutive_failures = 0
        self.last_success_at = now
        self.last_error = ""
        self.online = True
        self.degraded = False
        self.services[service] = {"ok": True, "at": now, "error": ""}
        bot_state.network_online = True
        bot_state.network_status = self.snapshot()
        if was_offline:
            logger.info("Network connectivity restored — resuming normal operation")
            if self._alerted_offline:
                self._alerted_offline = False
                _fire_alert_restored()

    def record_failure(
        self, service: str, exc: Optional[BaseException] = None, *, force: bool = False
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        err = str(exc)[:200] if exc else "request failed"
        self.last_failure_at = now
        self.services[service] = {"ok": False, "at": now, "error": err}

        if force or (exc and is_network_error(exc)):
            self.consecutive_failures += 1
            self.last_error = err
            self.degraded = True
        else:
            return

        bot_state.network_status = self.snapshot()
        if self.consecutive_failures >= _OFFLINE_AFTER_FAILURES and self.online:
            self.online = False
            bot_state.network_online = False
            self._alerted_offline = True
            logger.warning(
                "Network offline after %s failures — pausing new trades, "
                "using cached data (%s)",
                self.consecutive_failures,
                err,
            )
            _fire_alert_offline(err)

    def snapshot(self) -> dict:
        return {
            "online": self.online,
            "degraded": self.degraded,
            "consecutive_failures": self.consecutive_failures,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "last_error": self.last_error,
            "services": dict(self.services),
        }


_tracker = NetworkTracker()


def get_tracker() -> NetworkTracker:
    return _tracker


def record_success(service: str = "api") -> None:
    _tracker.record_success(service)


def record_failure(
    service: str, exc: Optional[BaseException] = None, *, force: bool = False
) -> None:
    _tracker.record_failure(service, exc, force=force)


def is_online() -> bool:
    return _tracker.online


def snapshot() -> dict:
    return _tracker.snapshot()


def _fire_alert_offline(reason: str) -> None:
    alerter = getattr(bot_state, "_alerter_ref", None)
    if not alerter:
        return
    try:
        asyncio.get_running_loop().create_task(
            alerter.network_offline(reason)
        )
    except RuntimeError:
        pass


def _fire_alert_restored() -> None:
    alerter = getattr(bot_state, "_alerter_ref", None)
    if not alerter:
        return
    try:
        asyncio.get_running_loop().create_task(
            alerter.network_restored()
        )
    except RuntimeError:
        pass


async def probe_connectivity(session: aiohttp.ClientSession) -> bool:
    """Active probe — any reachable endpoint counts as online."""
    last_exc: Optional[Exception] = None
    for url in _PROBE_URLS:
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status < 500:
                    record_success("probe")
                    return True
                last_exc = RuntimeError(f"HTTP {resp.status} from {url}")
        except Exception as e:
            last_exc = e
            logger.debug("Connectivity probe failed %s: %s", url, e)
    record_failure("probe", last_exc, force=True)
    return False


async def network_watchdog_loop() -> None:
    """Background reconnect probe — runs faster while offline."""
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                interval = (
                    _PROBE_INTERVAL_OFFLINE_SEC
                    if not _tracker.online
                    else _PROBE_INTERVAL_ONLINE_SEC
                )
                await probe_connectivity(session)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("Network watchdog: %s", e)
                await asyncio.sleep(_PROBE_INTERVAL_OFFLINE_SEC)