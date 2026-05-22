"""
Auto-correlation discovery for Polymarket markets.

Polymarket groups related markets via a shared groupItemTitle / category tag.
For example, a "Bitcoin Price" group might contain:
  "Will BTC close above $90,000 this week?"
  "Will BTC close above $100,000 this week?"
  "Will BTC close above $110,000 this week?"

These share a MONOTONE relationship:
  P(above $110k) ≤ P(above $100k) ≤ P(above $90k)
  If any market violates this ordering, it is mispriced.

We also detect MUTEX relationships from multi-outcome groups:
  "Who will win the 2024 election?" → candidate markets sum ≤ 1.00.
  If any two markets sum > 1.00 + fees, there is an overround to exploit.

Relations discovered here are fed into the existing CorrelatedDetector.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from api_clients.polymarket_client import Market
from strategies.correlated import (
    CorrelatedDetector,
    MarketRelation,
    RelationType,
    CorrelatedConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"[$]?([0-9][0-9,]*(?:\.[0-9]+)?)[kKmMbB%]?")
_THRESHOLD_WORDS = {
    "above", "over", "exceed", "exceeds", "at least", "more than",
    "below", "under", "less than", "at most",
}


def _extract_threshold(question: str) -> Optional[float]:
    """Pull the first numeric threshold from a question string."""
    m = _NUMBER_RE.search(question.replace(",", ""))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _normalise(question: str) -> str:
    q = question.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _direction(question: str) -> str:
    """Return 'above', 'below', or 'unknown'."""
    q = question.lower()
    for word in ("above", "over", "exceed", "at least", "more than"):
        if word in q:
            return "above"
    for word in ("below", "under", "less than", "at most"):
        if word in q:
            return "below"
    return "unknown"


def _strip_threshold(question: str) -> str:
    """Remove the numeric part so we can compare question shapes."""
    q = _normalise(question)
    q = _NUMBER_RE.sub("NUM", q)
    return q


# ---------------------------------------------------------------------------
# Core auto-discovery
# ---------------------------------------------------------------------------

@dataclass
class AutoCorrelatorConfig:
    # Min confidence to pass to CorrelatedDetector
    relation_confidence: float = 0.90
    # Only consider markets with at least this much liquidity
    min_liquidity_usd: float = 200.0
    # Max markets per event to consider (avoid O(n²) blowup on large events)
    max_markets_per_event: int = 20


class AutoCorrelator:
    """
    Discovers logical market relationships from Kalshi market structure.
    """

    def __init__(self, cfg: AutoCorrelatorConfig = None):
        self.cfg = cfg or AutoCorrelatorConfig()

    def discover(self, markets: list[Market]) -> list[MarketRelation]:
        """
        Given all open markets, return a list of MarketRelations.
        Feed these into CorrelatedDetector.load_relations().
        """
        relations: list[MarketRelation] = []

        # Group by group_id (Polymarket groupItemTitle / category)
        by_event: dict[str, list[Market]] = {}
        for m in markets:
            if m.status != "open":
                continue
            if m.liquidity_usd < self.cfg.min_liquidity_usd:
                continue
            key = m.group_id or m.market_id
            by_event.setdefault(key, []).append(m)

        for _group_key, group in by_event.items():
            group = group[: self.cfg.max_markets_per_event]
            if len(group) < 2:
                continue
            relations.extend(self._analyse_event_group(group))

        logger.info(
            f"AutoCorrelator: discovered {len(relations)} relations "
            f"from {len(by_event)} event groups"
        )
        return relations

    def _analyse_event_group(self, markets: list[Market]) -> list[MarketRelation]:
        """
        Analyse a group of markets from the same event.
        Detects MONOTONE chains and MUTEX pairs.
        """
        relations = []

        # Check if all have the same "shape" (same question minus the number)
        shapes = [_strip_threshold(m.question) for m in markets]
        thresholds = [_extract_threshold(m.question) for m in markets]
        directions = [_direction(m.question) for m in markets]

        # If shapes match and all have numeric thresholds → MONOTONE chain
        same_shape = len(set(shapes)) == 1
        all_have_threshold = all(t is not None for t in thresholds)
        consistent_direction = len(set(directions)) == 1 and directions[0] != "unknown"

        if same_shape and all_have_threshold and consistent_direction:
            direction = directions[0]
            # Sort by threshold
            sorted_markets = sorted(
                markets,
                key=lambda m: _extract_threshold(m.question) or 0,
            )
            # For "above X" markets: P(above lower) ≥ P(above higher)
            # So lower-threshold market IMPLIES higher-threshold market is ≤ it
            # Expressed as: the lower-threshold market has higher YES prob
            # → MONOTONE(A, B) means P(B) >= P(A)  (B is less demanding)
            for i in range(len(sorted_markets) - 1):
                low_thresh = sorted_markets[i]   # lower number → higher prob if "above"
                high_thresh = sorted_markets[i + 1]

                if direction == "above":
                    # P(above low) >= P(above high)
                    # So high_thresh.yes_price should ≤ low_thresh.yes_price
                    # MONOTONE(high, low) means "low should be >= high"
                    rel = MarketRelation(
                        market_id_a=high_thresh.market_id,  # anchor (lower prob)
                        market_id_b=low_thresh.market_id,   # should be >= anchor
                        relation=RelationType.MONOTONE,
                        confidence=self.cfg.relation_confidence,
                    )
                else:
                    # "below X" — P(below high) >= P(below low)
                    rel = MarketRelation(
                        market_id_a=low_thresh.market_id,
                        market_id_b=high_thresh.market_id,
                        relation=RelationType.MONOTONE,
                        confidence=self.cfg.relation_confidence,
                    )
                relations.append(rel)
                logger.debug(
                    f"MONOTONE: {rel.market_id_a} → {rel.market_id_b} "
                    f"(thresholds: {_extract_threshold(sorted_markets[i].question)} "
                    f"vs {_extract_threshold(sorted_markets[i+1].question)})"
                )

        else:
            # Different shapes — check for MUTEX (multi-outcome event)
            # e.g. "Will candidate A win?", "Will candidate B win?" in same election
            relations.extend(self._check_mutex(markets))

        return relations

    def _check_mutex(self, markets: list[Market]) -> list[MarketRelation]:
        """
        For a multi-outcome event (e.g. which candidate wins), any pair of
        markets that cannot both resolve YES is MUTEX.

        Heuristic: if all market questions in the group follow the pattern
        "Will [X] be/do [Y]?" where X varies, they are likely mutually exclusive
        outcomes. We add MUTEX for every pair.

        Also catches: "Democrat wins" + "Republican wins" style questions.
        """
        relations = []
        # Simple heuristic: if the event looks like a "which outcome" event,
        # mark all pairs as MUTEX with slightly lower confidence
        # We detect this by checking if the question shapes are all similar
        # but NOT matching (i.e., same event, different outcomes)

        if len(markets) >= 2:
            # All pairs
            for i in range(len(markets)):
                for j in range(i + 1, len(markets)):
                    a, b = markets[i], markets[j]
                    # Only add MUTEX if the questions are clearly different outcomes
                    # (avoid false positives from threshold chains)
                    if self._looks_mutex(a.question, b.question):
                        relations.append(MarketRelation(
                            market_id_a=a.market_id,
                            market_id_b=b.market_id,
                            relation=RelationType.MUTEX,
                            confidence=self.cfg.relation_confidence * 0.9,
                        ))
                        logger.debug(f"MUTEX: {a.market_id} ⊥ {b.market_id}")

        return relations

    @staticmethod
    def _looks_mutex(q1: str, q2: str) -> bool:
        """
        Heuristic: two questions look mutually exclusive if they have the
        same verb/template structure but differ on the subject/object.
        We avoid false positives by requiring that neither has a numeric
        threshold (those are handled as MONOTONE chains above).
        """
        if _extract_threshold(q1) or _extract_threshold(q2):
            return False

        # Both must be yes/no questions (not threshold markets)
        kw1 = set(re.findall(r"[a-z]{3,}", q1.lower()))
        kw2 = set(re.findall(r"[a-z]{3,}", q2.lower()))

        overlap = kw1 & kw2
        union = kw1 | kw2
        if not union:
            return False

        jaccard = len(overlap) / len(union)
        # Moderate overlap (same topic) but not identical (different outcomes)
        return 0.15 <= jaccard <= 0.65


# ---------------------------------------------------------------------------
# Convenience: build a CorrelatedDetector from live markets
# ---------------------------------------------------------------------------

def build_correlated_detector(
    markets: list[Market],
    corr_cfg: CorrelatedConfig = None,
    auto_cfg: AutoCorrelatorConfig = None,
) -> CorrelatedDetector:
    """
    Convenience function: auto-discover relations and return a ready-to-use
    CorrelatedDetector. Call this whenever the market list refreshes.
    """
    detector = CorrelatedDetector(corr_cfg or CorrelatedConfig())
    correlator = AutoCorrelator(auto_cfg or AutoCorrelatorConfig())
    relations = correlator.discover(markets)
    detector.load_relations(relations)
    return detector
