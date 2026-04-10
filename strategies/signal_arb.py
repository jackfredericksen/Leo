"""
Common types for signal-based strategies.

AggregatedSignal is the shared output format for all signal strategies:
  crypto_signal, range_straddle, news_fade, correlated arb.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class SignalArbConfig:
    min_edge: float = 0.05
    uncertainty_buffer: float = 0.05
    fee_pct: float = 0.02
    max_position_usd: float = 50.0
    kelly_fraction: float = 0.10


@dataclass
class AggregatedSignal:
    market_id: str
    question: str
    market_prob: float
    model_prob: float
    edge: float
    recommended_side: str
    source: str
    confidence: float = 1.0
    reasoning: str = ""
    recommended_size_usd: float = 0.0
    detected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def _norm_cdf(x: float) -> float:
    a1, a2, a3, a4, a5 = (
        0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    )
    p = 0.2316419
    k = 1.0 / (1.0 + p * abs(x))
    poly = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
    cdf = 1.0 - (1.0 / (2 * math.pi) ** 0.5) * math.exp(-0.5 * x ** 2) * poly
    return float(cdf if x >= 0 else 1.0 - cdf)
