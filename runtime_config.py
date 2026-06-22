"""
Persist dashboard control changes across restarts.

Writes data/runtime.json (gitignored via data/). Applied on startup in main.py
before loops start. Does not overwrite .env — runtime overrides win until restart
clears them or you edit .env explicitly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUNTIME_PATH = Path("data/runtime.json")

_STRATEGY_KEYS = (
    "arbitrage", "crypto_signal", "range_straddle", "cross_arb",
    "correlated", "news_fade", "forecast", "llm", "weather",
    "favorite_short", "oracle_squeeze", "semantic_arb",
    "orderbook_momentum", "market_maker", "btc_5min",
)


def load_runtime(cfg) -> dict:
    """Apply saved runtime overrides to the live Config object."""
    if not RUNTIME_PATH.is_file():
        return {}
    try:
        data = json.loads(RUNTIME_PATH.read_text())
    except Exception as e:
        logger.warning("Could not read runtime.json: %s", e)
        return {}

    if "dry_run" in data:
        cfg.dry_run = bool(data["dry_run"])

    for key in _STRATEGY_KEYS:
        if key in data.get("strategies", {}):
            block = getattr(cfg, key, None)
            if block is not None and hasattr(block, "enabled"):
                block.enabled = bool(data["strategies"][key])

    risk = data.get("risk", {})
    if risk:
        if "max_daily_usd" in risk:
            cfg.risk.max_daily_usd_deployed = float(risk["max_daily_usd"])
        if "max_position_usd" in risk:
            cfg.arbitrage.max_position_usd = float(risk["max_position_usd"])
        if "max_total_exposure_usd" in risk:
            cfg.arbitrage.max_total_exposure_usd = float(risk["max_total_exposure_usd"])
        if "min_edge_pct" in risk:
            cfg.arbitrage.min_profit_pct = float(risk["min_edge_pct"]) / 100.0
        if "per_strategy_max_usd" in risk:
            cfg.risk.per_strategy_max_usd = float(risk["per_strategy_max_usd"])

    logger.info("Loaded runtime overrides from %s", RUNTIME_PATH)
    return data


def save_runtime(
    *,
    dry_run: bool | None = None,
    strategy: tuple[str, bool] | None = None,
    risk: dict[str, Any] | None = None,
) -> dict:
    """Merge one control change into runtime.json."""
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if RUNTIME_PATH.is_file():
        try:
            data = json.loads(RUNTIME_PATH.read_text())
        except Exception:
            data = {}

    if dry_run is not None:
        data["dry_run"] = dry_run
    if strategy:
        name, enabled = strategy
        data.setdefault("strategies", {})[name] = enabled
    if risk:
        data.setdefault("risk", {}).update(risk)

    RUNTIME_PATH.write_text(json.dumps(data, indent=2))
    return data


def snapshot_runtime(cfg) -> dict:
    """Current effective config for API / debugging."""
    strategies = {}
    for key in _STRATEGY_KEYS:
        block = getattr(cfg, key, None)
        if block is not None and hasattr(block, "enabled"):
            strategies[key] = block.enabled
    return {
        "dry_run": cfg.dry_run,
        "strategies": strategies,
        "risk": {
            "max_daily_usd": cfg.risk.max_daily_usd_deployed,
            "max_position_usd": cfg.arbitrage.max_position_usd,
            "max_total_exposure_usd": cfg.arbitrage.max_total_exposure_usd,
            "min_edge_pct": round(cfg.arbitrage.min_profit_pct * 100, 2),
            "per_strategy_max_usd": cfg.risk.per_strategy_max_usd,
        },
        "pnl_source": "resolved_trades_db" if cfg.dry_run else "position_manager_live",
    }