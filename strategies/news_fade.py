"""
Strategy 5: News overreaction fade.

When a Kalshi crypto market spikes sharply (e.g. a BTC ETF approval
pushes a contract from 50% to 80%), fade the move by taking the other side.

Research shows ~60% of initial news-driven overreactions revert within
90-120 minutes. The key is NOT entering in the first 1.5 hours — wait
for the initial spike to stabilize, then fade.

Spike detection uses cross-cycle price history: on each scan the detector
compares the current yes_price against the price it saw on the previous
scan. When the move exceeds min_spike_pct the spike is logged with a
timestamp; fades are only opened after min_hours_old have elapsed.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from api_clients.kalshi_client import Market
from strategies.signal_arb import AggregatedSignal, SignalArbConfig
from strategies.kelly import KellySizer

logger = logging.getLogger(__name__)

_CRYPTO_KEYWORDS = [
    "btc", "bitcoin", "eth", "ethereum", "sol", "solana", "crypto",
]


@dataclass
class NewsFadeConfig:
    min_spike_pct: float = 0.12    # 12% price move to qualify as a spike
    min_hours_old: float = 1.5     # don't fade until 1.5h after the move
    max_hours_old: float = 4.0     # stop fading after 4h
    fade_fraction: float = 0.50    # expect 50% of spike to revert
    min_liquidity_usd: float = 5000.0
    min_edge: float = 0.05


class NewsFadeDetector:
    """
    Detects over-reacted crypto markets and generates fade signals.

    Maintains two pieces of state across scan() calls:
      _prev_prices  — yes_price seen on the previous scan cycle
      _spike_log    — (spike_time, direction, price_before_spike)
    """

    def __init__(
        self,
        cfg: SignalArbConfig,
        fade_cfg: NewsFadeConfig,
        bankroll: float,
    ):
        self.cfg = cfg
        self.fade = fade_cfg
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)
        # market_id → yes_price from previous scan
        self._prev_prices: dict[str, float] = {}
        # market_id → (spike_detected_at, direction, price_before_spike)
        self._spike_log: dict[str, tuple[datetime, str, float]] = {}

    def scan(self, markets: list[Market]) -> list[AggregatedSignal]:
        results = []
        now = datetime.now(timezone.utc)
        new_prices: dict[str, float] = {}

        for market in markets:
            try:
                if market.liquidity_usd < self.fade.min_liquidity_usd:
                    continue

                q = market.question.lower()
                if not any(w in q for w in _CRYPTO_KEYWORDS):
                    continue

                current = market.yes_price
                new_prices[market.market_id] = current

                # --- Spike detection (requires a previous observation) ---
                prev = self._prev_prices.get(market.market_id)
                if (
                    prev
                    and prev > 0
                    and market.market_id not in self._spike_log
                ):
                    move_pct = abs(current - prev) / prev
                    if move_pct >= self.fade.min_spike_pct:
                        direction = "up" if current > prev else "down"
                        self._spike_log[market.market_id] = (
                            now, direction, prev
                        )
                        logger.info(
                            f"NewsFade: spike {direction} detected on "
                            f"{market.market_id} "
                            f"{prev:.2%}->{current:.2%}"
                        )

                spike_entry = self._spike_log.get(market.market_id)
                if not spike_entry:
                    continue

                spike_time, spike_dir, prior_price = spike_entry
                hours_since = (now - spike_time).total_seconds() / 3600

                if hours_since < self.fade.min_hours_old:
                    continue   # too soon — spike still in progress
                if hours_since > self.fade.max_hours_old:
                    del self._spike_log[market.market_id]
                    continue

                # --- Compute expected reversion ---
                spike_move = current - prior_price
                expected_reversion = spike_move * self.fade.fade_fraction
                target_price = current - expected_reversion

                if spike_dir == "up":
                    # Spiked up → fade by buying NO
                    model_prob = target_price
                    market_prob = current
                    side = "no"
                    edge = market_prob - model_prob - self.cfg.fee_pct
                else:
                    # Spiked down → fade by buying YES
                    model_prob = target_price
                    market_prob = current
                    side = "yes"
                    edge = model_prob - market_prob - self.cfg.fee_pct

                if edge < self.fade.min_edge:
                    continue

                trade_prob = (
                    model_prob if side == "yes" else (1 - model_prob)
                )
                size = self.sizer.size(
                    trade_prob, market_prob, self.cfg.max_position_usd
                )

                results.append(AggregatedSignal(
                    market_id=market.market_id,
                    question=market.question,
                    market_prob=market_prob,
                    model_prob=model_prob,
                    edge=edge,
                    recommended_side=side,
                    source="news-fade",
                    reasoning=(
                        f"spike {spike_dir} {hours_since:.1f}h ago: "
                        f"{prior_price:.2%}->{current:.2%} "
                        f"fade target={target_price:.2%}"
                    ),
                    recommended_size_usd=size,
                ))

            except Exception as e:
                logger.debug(
                    f"NewsFade error on {market.market_id}: {e}"
                )

        # Update price history for next scan cycle
        self._prev_prices = new_prices

        return sorted(results, key=lambda x: abs(x.edge), reverse=True)
