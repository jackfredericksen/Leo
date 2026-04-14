"""
Strategy: LLM fundamental analysis signal.

Uses Claude Haiku to estimate probabilities for Kalshi markets that
other strategies can't price well — qualitative markets like elections,
policy decisions, geopolitical events, company actions.

Filters:
  - Skips crypto price markets (handled by crypto_signal.py)
  - Skips markets with < min_hours_to_resolve remaining
  - Skips markets already in the LLM cache (TTL = 30 min)
  - Prioritizes markets with high liquidity (more likely to be priceable)

Optionally enriches each question with matching Metaculus/Manifold
context from the forecast_client if available.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from api_clients.kalshi_client import Market
from api_clients.llm_client import LLMClient
from api_clients.forecast_client import ForecastClient
from strategies.signal_arb import AggregatedSignal, SignalArbConfig
from strategies.kelly import KellySizer

logger = logging.getLogger(__name__)

# Markets where LLM adds no value (handled by dedicated strategies)
_SKIP_PREFIXES = ("KXBTC", "KXETH", "KXSOL")
_SKIP_KEYWORDS = ("bitcoin", "ethereum", "solana", "btc price", "eth price")


def _is_crypto_price(market: Market) -> bool:
    uid = market.market_id.upper()
    if any(uid.startswith(p) for p in _SKIP_PREFIXES):
        return True
    q = market.question.lower()
    return any(kw in q for kw in _SKIP_KEYWORDS)


class LLMSignalDetector:
    """
    Generates trading signals by asking Claude to estimate probabilities
    for qualitative Kalshi markets.
    """

    def __init__(
        self,
        cfg: SignalArbConfig,
        llm: LLMClient,
        bankroll: float,
        forecast_client: Optional[ForecastClient] = None,
        max_markets_per_scan: int = 20,
        min_liquidity_usd: float = 500.0,
        min_hours_remaining: float = 2.0,
    ):
        self.cfg = cfg
        self.llm = llm
        self.fc = forecast_client
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)
        self.max_markets = max_markets_per_scan
        self.min_liquidity = min_liquidity_usd
        self.min_hours = min_hours_remaining

    def _build_context(self, market: Market) -> str:
        """Attach forecast community estimates as context if available."""
        if not self.fc:
            return ""
        matches = self.fc.match_all(
            market.question, min_similarity=0.25, min_forecasters=5,
            max_results=2
        )
        if not matches:
            return ""
        parts = []
        for sim, fq in matches:
            parts.append(
                f"{fq.source.title()} ({fq.num_forecasters} forecasters): "
                f"{fq.yes_prob:.1%} — \"{fq.title[:80]}\""
            )
        return "Community forecasts:\n" + "\n".join(parts)

    def _select_markets(
        self, markets: list[Market]
    ) -> list[dict]:
        """
        Pick the most promising markets for LLM analysis.
        Sorted by liquidity (high-liquidity markets are better priced
        and more likely to have useful external context).
        """
        now = datetime.now(timezone.utc)
        candidates = []
        for m in markets:
            if m.status != "open":
                continue
            if _is_crypto_price(m):
                continue
            if m.liquidity_usd < self.min_liquidity:
                continue
            hours = (m.close_time - now).total_seconds() / 3600
            if hours < self.min_hours:
                continue
            # Skip markets where we already have a fresh cached result
            # unless it resulted in a trade signal
            candidates.append({
                "market_id": m.market_id,
                "question": m.question or m.market_id,
                "prob": m.yes_price,
                "hours": hours,
                "liquidity": m.liquidity_usd,
                "context": self._build_context(m),
                "_market": m,
            })

        # Prioritise by liquidity so the limited LLM budget hits the
        # most-active markets first
        candidates.sort(key=lambda x: x["liquidity"], reverse=True)
        return candidates[:self.max_markets]

    async def scan(self, markets: list[Market]) -> list[AggregatedSignal]:
        """Async scan — calls LLM for each selected market."""
        selected = self._select_markets(markets)
        if not selected:
            return []

        estimates = await self.llm.estimate_batch(selected)
        est_by_id = {e.market_id: e for e in estimates}

        results = []
        for candidate in selected:
            try:
                est = est_by_id.get(candidate["market_id"])
                if not est:
                    continue

                market_prob = candidate["prob"]
                llm_prob = est.probability
                edge = llm_prob - market_prob - self.cfg.fee_pct

                if abs(edge) < self.cfg.min_edge:
                    continue

                # Scale by LLM confidence — low confidence = no trade
                if est.confidence < 0.3:
                    continue

                side = "yes" if edge > 0 else "no"
                trade_prob = (
                    llm_prob if side == "yes" else (1 - llm_prob)
                )
                trade_price = (
                    market_prob if side == "yes" else (1 - market_prob)
                )
                size = self.sizer.size(
                    trade_prob, trade_price, self.cfg.max_position_usd
                )
                # Scale size by LLM confidence
                size = round(size * est.confidence, 2)

                results.append(AggregatedSignal(
                    market_id=candidate["market_id"],
                    question=candidate["question"],
                    market_prob=market_prob,
                    model_prob=llm_prob,
                    edge=edge,
                    recommended_side=side,
                    source=f"llm:{self.llm._model[:20]}",
                    confidence=est.confidence,
                    reasoning=est.reasoning,
                    recommended_size_usd=size,
                ))
            except Exception as e:
                logger.debug(
                    f"LLMSignal error on {candidate['market_id']}: {e}"
                )

        return sorted(results, key=lambda x: abs(x.edge), reverse=True)
