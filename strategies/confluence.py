"""
Confluence tracker — require cross-strategy agreement before executing.

A trade passes only when >= min_strategies detect the same market + direction.
Reduces false positives and concentrates capital on high-conviction setups.

Usage:
  1. After each scan loop, call confluence.update(market_id, strategy, side)
  2. Before each execute call, call confluence.passes(market_id, side)
  3. Call confluence.clear() at the start of each cycle (not required but keeps
     signals fresh — the tracker accumulates across cycles by default so older
     agreements persist until overwritten).
"""

from dataclasses import dataclass


@dataclass
class ConfluenceConfig:
    enabled: bool = True
    min_strategies: int = 1         # 1 = effectively off; raise to 2+ to gate trades
    max_edge_bonus: float = 0.02    # +2% virtual edge at maximum agreement (5 strategies)


class ConfluenceTracker:
    """
    Thread-safe (asyncio-safe) in-memory confluence store.
    Keys are "market_id:side"; values are sets of strategy names.
    """

    def __init__(self, cfg: ConfluenceConfig):
        self.cfg = cfg
        self._signals: dict[str, set[str]] = {}

    def update(self, market_id: str, strategy: str, side: str) -> None:
        key = f"{market_id}:{side.lower()}"
        if key not in self._signals:
            self._signals[key] = set()
        self._signals[key].add(strategy)

    def clear(self) -> None:
        self._signals.clear()

    def count(self, market_id: str, side: str) -> int:
        return len(self._signals.get(f"{market_id}:{side.lower()}", set()))

    def passes(self, market_id: str, side: str) -> bool:
        if not self.cfg.enabled or self.cfg.min_strategies <= 1:
            return True
        return self.count(market_id, side) >= self.cfg.min_strategies

    def edge_bonus(self, market_id: str, side: str) -> float:
        """Extra virtual edge proportional to agreement count (saturates at 5 strategies)."""
        n = min(self.count(market_id, side), 5)
        return self.cfg.max_edge_bonus * n / 5

    def best_opps(self, limit: int = 10) -> list[dict]:
        """Top multi-strategy opportunities sorted by agreement count."""
        items = []
        for key, strategies in self._signals.items():
            if len(strategies) < 2:
                continue
            market_id, side = key.rsplit(":", 1)
            items.append({
                "market_id": market_id,
                "side": side,
                "count": len(strategies),
                "strategies": sorted(strategies),
            })
        items.sort(key=lambda x: x["count"], reverse=True)
        return items[:limit]
