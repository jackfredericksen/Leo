"""
Evolution agent — weekly LLM-driven performance review.

Reads per-strategy trade stats and win rates from storage, then asks Claude
to suggest parameter adjustments (min_edge changes, strategy disables).
The recommendation text is stored on bot_state for display in the web UI.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    enabled: bool = True
    review_interval_hours: int = 168   # weekly
    min_trades_for_review: int = 10
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 350


class EvolutionAgent:
    """
    Runs periodically (weekly by default). Generates a performance review
    and parameter recommendation using the LLM client.

    Requires:
      llm_client : api_clients.llm_client.LLMClient (already initialised)
      storage    : storage.Storage
    """

    def __init__(self, cfg: EvolutionConfig, llm_client, storage):
        self.cfg = cfg
        self.llm = llm_client
        self.storage = storage
        self._last_review: Optional[datetime] = None
        self._last_recommendation: Optional[str] = None

    @property
    def last_recommendation(self) -> Optional[str]:
        return self._last_recommendation

    @property
    def last_review_ts(self) -> Optional[str]:
        if self._last_review:
            return self._last_review.isoformat()
        return None

    def _due(self) -> bool:
        if not self.cfg.enabled or not self.llm:
            return False
        if self._last_review is None:
            return True
        return (
            datetime.now(timezone.utc) - self._last_review
            >= timedelta(hours=self.cfg.review_interval_hours)
        )

    async def maybe_run(self) -> Optional[str]:
        """Run if review is due. Returns recommendation text."""
        if not self._due():
            return self._last_recommendation
        try:
            result = await self._run_review()
            if result:
                self._last_recommendation = result
            return result
        except Exception as e:
            logger.error(f"Evolution agent: {e}")
            return None

    async def _run_review(self) -> Optional[str]:
        stats = self.storage.get_strategy_pnl()
        win_rates = self.storage.get_strategy_win_rates()

        if not stats:
            logger.info("Evolution: no trade data available")
            return None

        total = sum(s.get("trades", 0) for s in stats)
        if total < self.cfg.min_trades_for_review:
            logger.info(f"Evolution: {total} trades < {self.cfg.min_trades_for_review} minimum")
            return None

        lines = []
        for s in stats:
            wr_data = win_rates.get(s["arb_type"], {})
            wr = wr_data.get("win_rate", 0) or 0
            lines.append(
                f"{s['arb_type']}: {s['trades']} trades,"
                f" avg_edge={100*(s['avg_edge'] or 0):.1f}%,"
                f" win_rate={100*wr:.0f}%,"
                f" est_pnl=${s['est_pnl'] or 0:.2f}"
            )

        prompt = (
            "You are a performance analyst for Leo, a Polymarket prediction market trading bot.\n\n"
            "Strategy performance data:\n"
            + "\n".join(lines)
            + "\n\nProvide a concise review (under 200 words) covering:\n"
            "1. Which 1-2 strategies underperform (win rate < 45% or negative P&L)\n"
            "2. Specific min_edge adjustments (e.g. 'raise signal:crypto min_edge to 7%')\n"
            "3. One strategy to consider disabling if clearly unprofitable\n"
            "Be specific with numbers."
        )

        try:
            response = await self.llm._client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip() if response.content else ""
            if text:
                self._last_review = datetime.now(timezone.utc)
                logger.info(f"Evolution review complete")
                logger.debug(f"Evolution:\n{text}")
            return text or None
        except Exception as e:
            logger.debug(f"Evolution LLM call failed: {e}")
            return None
