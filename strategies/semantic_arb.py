"""
Strategy: Semantic Cross-Platform Arbitrage.

Finds semantically IDENTICAL markets listed simultaneously on Polymarket and
Kalshi, then trades the persistent price gap between platforms.

Unlike the existing cross_platform strategy (which uses Kalshi as a directional
signal), this strategy requires near-identical market definitions before treating
the gap as a true arbitrage. The 2–4% persistent gap documented in academic
literature (arXiv:2601.01706) only holds for true semantic duplicates.

Matching algorithm (no ML required):
  1. Number extraction: both questions must contain the same numeric thresholds
     (within 1% tolerance). A market about "$100k" cannot match one about "$110k".
  2. Keyword Jaccard: content words (4+ chars, excluding common stop words) must
     overlap ≥ 65%.
  3. Direction agreement: both must use the same framing ("above" / "below").

Only pairs passing all three filters are considered. The minimum price gap
is 4% after fees — below that the execution risk is not worth it.

Note: Kalshi data is fetched via the existing KalshiSignalClient (no auth).
      We trade on Polymarket only (Kalshi side is read-only signal).
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from api_clients.polymarket_client import Market
from strategies.kelly import KellySizer
from strategies.signal_arb import AggregatedSignal

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset([
    "will", "the", "this", "that", "with", "from", "have", "been",
    "more", "than", "less", "what", "when", "where", "which", "there",
    "their", "would", "could", "should", "before", "after", "during",
    "into", "onto", "upon", "over", "under", "about", "above", "below",
])

_NUMBER_RE = re.compile(r"[$€£]?\s*([\d,]+(?:\.\d+)?)\s*([kKmMbB])?")


@dataclass
class SemanticArbConfig:
    enabled: bool = True
    min_price_gap: float = 0.04         # minimum gap after fees to trade
    min_jaccard: float = 0.65           # keyword overlap threshold
    number_tolerance: float = 0.01      # numbers must match within 1%
    max_position_usd: float = 100.0
    fee_pct: float = 0.02
    kelly_fraction: float = 0.10
    refresh_interval_sec: int = 120


def _extract_numbers(text: str) -> list[float]:
    nums = []
    for m in _NUMBER_RE.finditer(text):
        try:
            val = float(m.group(1).replace(",", ""))
            suffix = (m.group(2) or "").lower()
            if suffix == "k":
                val *= 1_000
            elif suffix == "m":
                val *= 1_000_000
            elif suffix == "b":
                val *= 1_000_000_000
            nums.append(val)
        except ValueError:
            pass
    return nums


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _direction(text: str) -> str:
    t = text.lower()
    for w in ("above", "exceed", "over", "more than", "higher", "at least"):
        if w in t:
            return "above"
    for w in ("below", "under", "less than", "lower", "at most"):
        if w in t:
            return "below"
    return "unknown"


def _numbers_match(nums_a: list[float], nums_b: list[float], tol: float) -> bool:
    if len(nums_a) != len(nums_b):
        return False
    if not nums_a:
        return True
    for a, b in zip(sorted(nums_a), sorted(nums_b)):
        if a == 0 and b == 0:
            continue
        denom = max(abs(a), abs(b))
        if abs(a - b) / denom > tol:
            return False
    return True


class SemanticArbDetector:
    """
    Compares Polymarket markets against Kalshi markets for identical semantic
    content, then trades price gaps on Polymarket only.
    """

    def __init__(self, cfg: SemanticArbConfig, bankroll: float):
        self.cfg = cfg
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)
        # Loaded externally: list of (question, price) tuples from Kalshi
        self._ext_markets: list[dict] = []

    def load_external(self, ext_markets: list[dict]):
        """
        Load external (Kalshi) market data.
        Each dict: {question, yes_price, platform}
        """
        self._ext_markets = ext_markets

    def scan(self, markets: list[Market]) -> list[AggregatedSignal]:
        if not self._ext_markets:
            return []

        results = []
        for market in markets:
            try:
                if market.status != "open":
                    continue
                if not market.question:
                    continue

                match = self._best_match(market.question)
                if match is None:
                    continue

                ext_price, platform = match
                poly_yes = market.yes_price

                gap = abs(poly_yes - ext_price)
                net_gap = gap - 2 * self.cfg.fee_pct
                if net_gap < self.cfg.min_price_gap:
                    continue

                # Trade the CHEAPER side on Polymarket
                # model_prob always stores YES probability (AggregatedSignal convention)
                model_prob = ext_price  # external platform's YES probability
                if poly_yes < ext_price:
                    # Polymarket YES is cheaper → buy YES on Polymarket
                    side = "yes"
                    entry = market.yes_ask
                else:
                    # Polymarket NO is cheaper → buy NO on Polymarket
                    side = "no"
                    entry = market.no_ask

                win_prob = model_prob if side == "yes" else (1.0 - model_prob)
                edge = win_prob - entry - self.cfg.fee_pct
                if edge < self.cfg.min_price_gap:
                    continue

                size = self.sizer.size(win_prob, entry, self.cfg.max_position_usd)

                results.append(AggregatedSignal(
                    market_id=market.market_id,
                    question=market.question,
                    market_prob=poly_yes,
                    model_prob=model_prob,
                    edge=edge,
                    recommended_side=side,
                    source=f"semantic_arb:{platform}",
                    confidence=0.80,
                    reasoning=(
                        f"poly={poly_yes:.0%} "
                        f"{platform}={ext_price:.0%} "
                        f"gap={gap:.0%} "
                        f"net={net_gap:.0%}"
                    ),
                    recommended_size_usd=size,
                ))

            except Exception as e:
                logger.debug(f"SemanticArb error on {market.market_id}: {e}")

        return sorted(results, key=lambda x: x.edge, reverse=True)

    def _best_match(
        self, poly_question: str
    ) -> Optional[tuple[float, str]]:
        """Find the best external match for a Polymarket question."""
        poly_nums = _extract_numbers(poly_question)
        poly_kw = _keywords(poly_question)
        poly_dir = _direction(poly_question)

        best_score = 0.0
        best_result = None

        for ext in self._ext_markets:
            ext_q = ext.get("question", "")
            if not ext_q:
                continue

            # Number match is a hard requirement when present
            ext_nums = _extract_numbers(ext_q)
            if poly_nums or ext_nums:
                if not _numbers_match(poly_nums, ext_nums, self.cfg.number_tolerance):
                    continue

            # Direction must agree (or at least one is unknown)
            ext_dir = _direction(ext_q)
            if poly_dir != "unknown" and ext_dir != "unknown" and poly_dir != ext_dir:
                continue

            jac = _jaccard(poly_kw, _keywords(ext_q))
            if jac < self.cfg.min_jaccard:
                continue

            if jac > best_score:
                best_score = jac
                best_result = (float(ext.get("yes_price", 0.5)), ext.get("platform", "ext"))

        return best_result
