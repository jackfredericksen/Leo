"""
Common types for signal-based strategies.

AggregatedSignal is the shared output format for all signal strategies:
  crypto_signal, range_straddle, news_fade, correlated arb.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


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
    slug: str = ""
    detected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# Markets wider than this are too illiquid to trade cost-effectively.
# yes_ask - yes_bid > 8 cents means you lose ~4 cents on entry alone.
_MAX_SPREAD = 0.08


def ask_edge(
    model_prob: float,
    yes_ask: float,
    no_ask: float,
    yes_bid: float,
    fee_pct: float,
    min_edge: float,
) -> Optional[tuple[float, str, float]]:
    """
    Compute executable edge using ask prices (what you'd actually pay).

    Returns (edge, side, entry_price) or None when:
      - spread is too wide (> _MAX_SPREAD)
      - neither side has edge >= min_edge

    Args:
        model_prob: Your estimated P(YES resolves).
        yes_ask / no_ask: Current ask prices for each side.
        yes_bid: Current best YES bid (used to compute spread).
        fee_pct: Round-trip fee to subtract from edge.
        min_edge: Minimum edge threshold.
    """
    if yes_ask - yes_bid > _MAX_SPREAD:
        return None
    edge_yes = model_prob - yes_ask - fee_pct
    edge_no = (1.0 - model_prob) - no_ask - fee_pct
    if edge_yes >= edge_no and edge_yes >= min_edge:
        return edge_yes, "yes", yes_ask
    if edge_no > edge_yes and edge_no >= min_edge:
        return edge_no, "no", no_ask
    return None

