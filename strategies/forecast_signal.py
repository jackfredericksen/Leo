"""
Strategy: Forecast aggregation signal.

Compares Kalshi implied probability against community consensus from
Metaculus and Manifold Markets. When professional forecasters
disagree significantly with what Kalshi prices imply, trade the edge.

Confidence scales with:
  - number of forecasters (more = higher confidence)
  - number of matching sources (both meta + manifold = higher confidence)
  - similarity score of the question match

Edge is computed as: (weighted_forecast_prob - kalshi_prob) - fee
"""

import logging
import math
from datetime import datetime, timezone

from api_clients.forecast_client import ForecastClient, ForecastQuestion
from api_clients.polymarket_client import Market
from strategies.signal_arb import AggregatedSignal, SignalArbConfig, ask_edge
from strategies.kelly import KellySizer

logger = logging.getLogger(__name__)

# Forecaster count → confidence weight
_MIN_FORECASTERS = 10


def _confidence(fq: ForecastQuestion, similarity: float) -> float:
    """Scale confidence 0-1 based on forecaster count and match quality."""
    forecaster_score = min(1.0, fq.num_forecasters / 100)
    return round(similarity * forecaster_score, 3)


class ForecastSignalDetector:
    """
    Scans Kalshi markets for divergence from Metaculus/Manifold consensus.
    """

    def __init__(
        self,
        cfg: SignalArbConfig,
        forecast_client: ForecastClient,
        bankroll: float,
    ):
        self.cfg = cfg
        self.fc = forecast_client
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)

    def scan(self, markets: list[Market]) -> list[AggregatedSignal]:
        results = []
        now = datetime.now(timezone.utc)

        for market in markets:
            try:
                if market.status != "open":
                    continue
                if market.liquidity_usd < 200:
                    continue
                hours_left = (
                    market.close_time - now
                ).total_seconds() / 3600
                if hours_left <= 0:
                    continue

                # Get all matching forecast questions
                matches = self.fc.match_all(
                    market.question,
                    min_similarity=0.25,
                    min_forecasters=_MIN_FORECASTERS,
                    max_results=5,
                )
                if not matches:
                    continue

                # Weighted average of forecast probabilities
                # weight = similarity × log(forecasters)
                total_weight = 0.0
                weighted_prob = 0.0
                sources = set()
                for sim, fq in matches:
                    w = sim * math.log1p(fq.num_forecasters)
                    weighted_prob += fq.yes_prob * w
                    total_weight += w
                    sources.add(fq.source)

                if total_weight == 0:
                    continue

                forecast_prob = weighted_prob / total_weight

                result = ask_edge(
                    forecast_prob,
                    market.yes_ask, market.no_ask, market.yes_bid,
                    self.cfg.fee_pct, self.cfg.min_edge,
                )
                if not result:
                    continue
                edge, side, entry_price = result

                # Confidence: best match similarity × forecaster weight
                best_sim, best_fq = matches[0]
                conf = _confidence(best_fq, best_sim)
                # Boost if multiple independent sources agree
                if len(sources) > 1:
                    conf = min(1.0, conf * 1.2)

                trade_prob = forecast_prob if side == "yes" else (1 - forecast_prob)
                size = self.sizer.size(
                    trade_prob, entry_price, self.cfg.max_position_usd
                )

                source_label = "+".join(sorted(sources))
                results.append(AggregatedSignal(
                    market_id=market.market_id,
                    question=market.question,
                    market_prob=market.yes_price,
                    model_prob=forecast_prob,
                    edge=edge,
                    recommended_side=side,
                    source=f"forecast:{source_label}",
                    confidence=conf,
                    reasoning=(
                        f"forecast={forecast_prob:.2%} "
                        f"mkt={market.yes_price:.2%} "
                        f"sources={len(matches)} "
                        f"match={best_sim:.0%} "
                        f'"{best_fq.title[:60]}"'
                    ),
                    recommended_size_usd=size,
                ))

            except Exception as e:
                logger.debug(
                    f"ForecastSignal error on {market.market_id}: {e}"
                )

        return sorted(results, key=lambda x: abs(x.edge), reverse=True)
