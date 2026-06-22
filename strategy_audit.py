"""
Verify enabled strategies have a scheduled asyncio loop.
Logs warnings at startup for config/scheduler mismatches.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# config attribute -> loop task name in create_bot_tasks()
_SCHEDULED = {
    "arbitrage": "arb",
    "crypto_signal": "crypto",
    "range_straddle": "range",
    "cross_arb": "cross",
    "correlated": "corr",
    "news_fade": "fade",
    "forecast": "forecast",
    "llm": "llm",
    "weather": "weather",
    "favorite_short": "fav_short",
    "oracle_squeeze": "oracle_squeeze",
    "semantic_arb": "semantic_arb",
    "orderbook_momentum": "ofi",
    "market_maker": "market_maker",
    "btc_5min": "btc_5min_scan",  # + btc_5min_fast
}


def audit_scheduled_strategies(cfg, scheduled_task_names: set[str]) -> list[dict]:
    """
    Return audit rows and log warnings for enabled-but-unscheduled strategies.
    """
    rows = []
    for attr, loop_name in _SCHEDULED.items():
        block = getattr(cfg, attr, None)
        if block is None or not hasattr(block, "enabled"):
            continue
        enabled = bool(block.enabled)
        # btc_5min has two loops
        if attr == "btc_5min":
            scheduled = "btc_5min_scan" in scheduled_task_names
        else:
            scheduled = loop_name in scheduled_task_names
        status = "ok"
        if enabled and not scheduled:
            status = "ENABLED_NOT_SCHEDULED"
            logger.warning(
                "Strategy %s is enabled in config but has no scheduled loop (%s)",
                attr, loop_name,
            )
        elif not enabled and scheduled:
            status = "scheduled_off"
        rows.append({
            "strategy": attr,
            "enabled": enabled,
            "loop": loop_name,
            "scheduled": scheduled,
            "status": status,
        })
    return rows