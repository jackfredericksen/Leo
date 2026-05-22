"""
Kyle's Lambda — order-flow price impact from CLOB trade tape.

High lambda: large price move per dollar of net flow → information-driven market
Low lambda: small price move per dollar of net flow → liquidity/noise dominated

Uses OLS regression: ΔP = λ × Q  (Q = signed notional, + = buy, − = sell)
Lambda is scaled to price change per $1M notional for interpretability.

A high-impact market is better for directional signal strategies (follow the flow).
A low-impact market is better for market making (stable spread, noise trading).
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_SCALE = 1_000_000.0


def compute_lambda(trades: list[dict]) -> Optional[float]:
    """
    Estimate Kyle's Lambda from a list of CLOB trade dicts.
    Each dict should have: price (float 0-1), size (float shares), side (str).
    Returns None if insufficient or degenerate data.
    """
    if len(trades) < 10:
        return None

    prices, signed_vols = [], []
    for t in trades:
        try:
            p = float(t.get("price", 0))
            sz = float(t.get("size", t.get("amount", 0)))
            side = str(t.get("side", t.get("type", ""))).upper()
            if p <= 0 or sz <= 0:
                continue
            sign = 1.0 if "BUY" in side else -1.0 if "SELL" in side else 0.0
            if sign == 0.0:
                continue
            prices.append(p)
            signed_vols.append(sign * sz)
        except (ValueError, TypeError):
            continue

    if len(prices) < 5:
        return None

    dp = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    q = signed_vols[1:]

    if len(dp) < 4:
        return None

    n = len(dp)
    mean_q = sum(q) / n
    mean_dp = sum(dp) / n
    num = sum((q[i] - mean_q) * (dp[i] - mean_dp) for i in range(n))
    den = sum((q[i] - mean_q) ** 2 for i in range(n))

    if den < 1e-12:
        return None

    raw = num / den
    return abs(raw) * _SCALE


class KyleLambdaTracker:
    """
    Fetches trade tape from the CLOB and maintains rolling lambda estimates.
    Call refresh(markets) from a background asyncio task every ~10 minutes.
    """

    def __init__(self, client, max_markets: int = 30):
        self.client = client
        self.max_markets = max_markets
        self._lambdas: dict[str, float] = {}
        self._updated: dict[str, datetime] = {}

    async def refresh(self, markets: list) -> None:
        """Update lambda estimates for the top markets by 24h volume."""
        top = sorted(markets, key=lambda m: m.volume_24h, reverse=True)[: self.max_markets]
        for market in top:
            try:
                trades = await self.client.get_trades(market.yes_token_id, limit=100)
                lam = compute_lambda(trades)
                if lam is not None:
                    self._lambdas[market.market_id] = lam
                    self._updated[market.market_id] = datetime.now(timezone.utc)
            except Exception as e:
                logger.debug(f"KyleLambda {market.market_id[:12]}: {e}")

    def get(self, market_id: str) -> Optional[float]:
        return self._lambdas.get(market_id)

    def is_high_impact(self, market_id: str, threshold: float = 0.5) -> bool:
        lam = self._lambdas.get(market_id)
        return lam is not None and lam > threshold

    def summary(self, markets: list, limit: int = 20) -> list[dict]:
        """Return lambda table with market context for the UI."""
        mid_map = {m.market_id: m for m in markets}
        items = []
        for mid, lam in self._lambdas.items():
            m = mid_map.get(mid)
            items.append({
                "market_id": mid,
                "question": (m.question[:60] if m else mid[:20]),
                "lambda": round(lam, 4),
                "regime": "high-impact" if lam > 0.5 else "liquidity",
            })
        items.sort(key=lambda x: x["lambda"], reverse=True)
        return items[:limit]
