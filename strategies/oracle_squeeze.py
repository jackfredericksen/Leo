"""
Strategy: Oracle Resolution Squeeze.

Polymarket markets continue trading on the CLOB after their real-world event
ends (past close_time) while the UMA Optimistic Oracle processes resolution.
During this window — which can last hours to days — the market price should
converge toward 0 or 1, but often lags.

Edge: markets past close_time with strong directional pricing (>0.90 or <0.10)
are near-certain outcomes. Buying the winning side captures the remaining gap
between current price and 1.00 at resolution.

This is latency-insensitive: the window is measured in hours, not seconds.
A home network polling every 60s is perfectly adequate.

Selection criteria:
  - close_time is in the past (event has ended)
  - result is not yet known (oracle still deliberating)
  - YES price > (1 - min_gap) → strongly priced YES  → buy YES
  - YES price < min_gap        → strongly priced NO   → buy NO

Risk: the oracle occasionally resolves against the apparent outcome (N/A, dispute).
The min_gap threshold and minimum hours filter reduce this risk substantially.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from api_clients.polymarket_client import Market
from strategies.kelly import KellySizer
from strategies.signal_arb import AggregatedSignal

logger = logging.getLogger(__name__)


@dataclass
class OracleSqueezeConfig:
    enabled: bool = True
    min_gap: float = 0.08           # price must be within 8¢ of 0 or 1 to squeeze
    max_gap: float = 0.25           # don't trade if price hasn't moved yet (>25¢ gap)
    min_hours_past_close: float = 0.5   # avoid markets that just closed (still volatile)
    max_hours_past_close: float = 168.0  # skip stale — 1 week max
    fee_pct: float = 0.02
    min_edge: float = 0.03
    max_position_usd: float = 100.0
    kelly_fraction: float = 0.12
    refresh_interval_sec: int = 60


class OracleSqueezeDetector:
    """
    Scans for markets past their close_time where price has converged toward
    a clear outcome but hasn't reached 1.00 / 0.00 yet.
    """

    def __init__(self, cfg: OracleSqueezeConfig, bankroll: float):
        self.cfg = cfg
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)

    def scan(self, markets: list[Market]) -> list[AggregatedSignal]:
        results = []
        now = datetime.now(timezone.utc)

        for market in markets:
            try:
                if market.status != "open":
                    continue
                if market.result is not None:
                    continue

                hours_past = (now - market.close_time).total_seconds() / 3600
                if not (self.cfg.min_hours_past_close <= hours_past <= self.cfg.max_hours_past_close):
                    continue

                yes_price = market.yes_price
                squeeze = self._detect_squeeze(yes_price)
                if squeeze is None:
                    continue

                side, model_prob, current_price = squeeze
                edge = model_prob - current_price - self.cfg.fee_pct
                if edge < self.cfg.min_edge:
                    continue

                size = self.sizer.size(model_prob, current_price, self.cfg.max_position_usd)

                results.append(AggregatedSignal(
                    market_id=market.market_id,
                    question=market.question,
                    market_prob=yes_price,
                    model_prob=model_prob,
                    edge=edge,
                    recommended_side=side,
                    source="oracle_squeeze",
                    confidence=0.85,
                    reasoning=(
                        f"closed {hours_past:.1f}h ago "
                        f"yes={yes_price:.2f} "
                        f"target={'1.00' if side == 'yes' else '0.00'} "
                        f"gap={abs(model_prob - current_price):.2f}"
                    ),
                    recommended_size_usd=size,
                ))

            except Exception as e:
                logger.debug(f"OracleSqueeze error on {market.market_id}: {e}")

        return sorted(results, key=lambda x: x.edge, reverse=True)

    def _detect_squeeze(
        self, yes_price: float
    ) -> Optional[tuple[str, float, float]]:
        """
        Returns (side, model_prob, entry_price) if there's a squeeze opportunity,
        else None.

        YES-side squeeze: price in [1 - max_gap, 1 - min_gap]
        NO-side squeeze:  price in [min_gap, max_gap]
        """
        if (1.0 - self.cfg.max_gap) <= yes_price <= (1.0 - self.cfg.min_gap):
            # Market priced near 1 but not there yet → buy YES → converges to 1.00
            return ("yes", 1.0, yes_price)

        if self.cfg.min_gap <= yes_price <= self.cfg.max_gap:
            # Market priced near 0 → buy NO (no_price near 1.00) → converges
            no_price = 1.0 - yes_price
            return ("no", 1.0, no_price)

        return None
