"""
Correlated market arbitrage — exploit logical dependencies between markets.

Examples:
  - "Candidate A wins state X" (p=0.90) and "Candidate A wins election" (p=0.40)
    If A needs X to win the election, the election market may be underpriced.

  - "Company files for IPO in Q1" (p=0.70) and "Company IPOs at >$10B valuation" (p=0.60)
    The second market can't resolve YES if the first resolves NO.

  - "BTC above $80k in Jan" (p=0.80) and "BTC above $80k in Feb" (p=0.75)
    Feb market should be >= Jan market (if still above in Jan, likely in Feb).
    If Feb < Jan, the Feb market is underpriced.

Logic:
  - Build a dependency graph from market metadata/tags
  - For each pair with a known logical relationship, check if prices are consistent
  - If inconsistent by more than fee threshold, trade the cheaper leg

This module provides the detection framework; the dependency relationships
must be configured manually or learned from historical resolution data.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from api_clients.kalshi_client import Market

logger = logging.getLogger(__name__)


class RelationType(Enum):
    IMPLIES = "implies"           # A resolves YES → B resolves YES (A is sufficient for B)
    MUTEX = "mutex"               # A and B cannot both resolve YES
    MONOTONE = "monotone"         # B's resolution date > A's, same threshold — P(B) >= P(A)
    SUBSET = "subset"             # A's criteria is a subset of B's — P(B) >= P(A)


@dataclass
class MarketRelation:
    market_id_a: str
    market_id_b: str
    relation: RelationType
    confidence: float = 1.0     # 0–1: how certain we are of this logical relationship


@dataclass
class CorrelatedOpportunity:
    market_id_mispriced: str
    market_id_anchor: str
    question_mispriced: str
    question_anchor: str
    relation: RelationType
    anchor_price: float
    mispriced_price: float
    fair_lower_bound: float     # Theoretical minimum price given the anchor
    edge: float                 # How far below the lower bound the price is
    recommended_side: str       # Which side to buy on the mispriced market
    max_size_usd: float
    detected_at: datetime


@dataclass
class CorrelatedConfig:
    min_edge: float = 0.04
    fee_pct: float = 0.02
    max_position_usd: float = 75.0
    min_confidence: float = 0.80


class CorrelatedDetector:
    def __init__(self, cfg: CorrelatedConfig):
        self.cfg = cfg
        self._relations: list[MarketRelation] = []

    def add_relation(self, relation: MarketRelation):
        """Register a known logical relationship between two markets."""
        self._relations.append(relation)

    def load_relations(self, relations: list[MarketRelation]):
        self._relations = relations

    def scan(self, markets: list[Market]) -> list[CorrelatedOpportunity]:
        market_map = {m.market_id: m for m in markets}
        opps = []

        for rel in self._relations:
            if rel.confidence < self.cfg.min_confidence:
                continue
            a = market_map.get(rel.market_id_a)
            b = market_map.get(rel.market_id_b)
            if not a or not b:
                continue
            opp = self._check_relation(a, b, rel)
            if opp:
                opps.append(opp)

        return sorted(opps, key=lambda o: o.edge, reverse=True)

    def _check_relation(
        self, a: Market, b: Market, rel: MarketRelation
    ) -> Optional[CorrelatedOpportunity]:
        net_edge_threshold = self.cfg.min_edge + self.cfg.fee_pct

        if rel.relation == RelationType.IMPLIES:
            # P(B) >= P(A) * confidence
            # If A implies B, then P(A) is a lower bound on P(B)
            lower_bound = a.yes_price * rel.confidence
            if b.yes_price < lower_bound - net_edge_threshold:
                edge = lower_bound - b.yes_price - self.cfg.fee_pct
                return self._make_opp(b, a, rel, lower_bound, edge, "YES")

        elif rel.relation == RelationType.MUTEX:
            # P(A) + P(B) <= 1
            # If they sum > 1 + fee, sell the expensive one (buy NO)
            total = a.yes_price + b.yes_price
            if total > 1.0 + net_edge_threshold:
                # Buy NO on whichever is more expensive
                if a.yes_price > b.yes_price:
                    edge = total - 1.0 - self.cfg.fee_pct
                    return self._make_opp(a, b, rel, 1.0 - b.yes_price, edge, "NO")
                else:
                    edge = total - 1.0 - self.cfg.fee_pct
                    return self._make_opp(b, a, rel, 1.0 - a.yes_price, edge, "NO")

        elif rel.relation in (RelationType.MONOTONE, RelationType.SUBSET):
            # P(B) >= P(A) — if B is priced below A, it's underpriced
            if b.yes_price < a.yes_price - net_edge_threshold:
                edge = a.yes_price - b.yes_price - self.cfg.fee_pct
                return self._make_opp(b, a, rel, a.yes_price, edge, "YES")

        return None

    def _make_opp(
        self,
        mispriced: Market,
        anchor: Market,
        rel: MarketRelation,
        lower_bound: float,
        edge: float,
        side: str,
    ) -> CorrelatedOpportunity:
        return CorrelatedOpportunity(
            market_id_mispriced=mispriced.market_id,
            market_id_anchor=anchor.market_id,
            question_mispriced=mispriced.question,
            question_anchor=anchor.question,
            relation=rel.relation,
            anchor_price=anchor.yes_price,
            mispriced_price=mispriced.yes_price,
            fair_lower_bound=lower_bound,
            edge=edge,
            recommended_side=side,
            max_size_usd=min(self.cfg.max_position_usd, mispriced.liquidity_usd * 0.05),
            detected_at=datetime.now(timezone.utc),
        )
