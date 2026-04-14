"""
Forecast aggregation client — pulls community probability estimates from
Metaculus and Manifold Markets to use as signals against Kalshi prices.

Both APIs are public and require no authentication.

Metaculus:  https://www.metaculus.com/api2/questions/
Manifold:   https://api.manifold.markets/v0/markets

The client fetches recent active binary questions, caches them, and
exposes a match() method that finds the best Kalshi market overlap
using fuzzy word-set similarity.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

METACULUS_BASE = "https://www.metaculus.com/api2"
MANIFOLD_BASE = "https://api.manifold.markets/v0"

_CACHE_TTL_SEC = 900   # 15 minutes


@dataclass
class ForecastQuestion:
    source: str          # "metaculus" | "manifold"
    question_id: str
    title: str
    yes_prob: float      # community probability of YES (0-1)
    num_forecasters: int
    close_time: Optional[datetime]
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ForecastClient:
    """
    Aggregates binary probability estimates from Metaculus and Manifold.
    Call refresh() periodically; then use match() to find the best
    community estimate for a given Kalshi market question.
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._questions: list[ForecastQuestion] = []
        self._last_fetch: Optional[datetime] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------ #
    #  Refresh                                                             #
    # ------------------------------------------------------------------ #

    async def refresh(self):
        """Fetch latest questions from all sources."""
        now = datetime.now(timezone.utc)
        if self._last_fetch:
            age = (now - self._last_fetch).total_seconds()
            if age < _CACHE_TTL_SEC:
                return

        results: list[ForecastQuestion] = []
        meta, manifold = await asyncio.gather(
            self._fetch_metaculus(),
            self._fetch_manifold(),
            return_exceptions=True,
        )
        if isinstance(meta, list):
            results.extend(meta)
        else:
            logger.warning(f"Metaculus fetch failed: {meta}")

        if isinstance(manifold, list):
            results.extend(manifold)
        else:
            logger.warning(f"Manifold fetch failed: {manifold}")

        self._questions = results
        self._last_fetch = now
        logger.info(
            f"Forecast: loaded {len(results)} questions "
            f"({sum(1 for q in results if q.source=='metaculus')} meta, "
            f"{sum(1 for q in results if q.source=='manifold')} manifold)"
        )

    async def _fetch_metaculus(self) -> list[ForecastQuestion]:
        """Fetch recent active binary questions from Metaculus."""
        questions = []
        params = {
            "type": "forecast",
            "status": "open",
            "forecast_type": "binary",
            "limit": 200,
            "order_by": "-activity",
        }
        async with self._session.get(
            f"{METACULUS_BASE}/questions/", params=params
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()

        for q in data.get("results", []):
            try:
                cp = q.get("community_prediction", {})
                prob = cp.get("full", {}).get("q2")
                if prob is None:
                    continue
                close_raw = q.get("close_time")
                close_time = None
                if close_raw:
                    close_time = datetime.fromisoformat(
                        close_raw.replace("Z", "+00:00")
                    )
                questions.append(ForecastQuestion(
                    source="metaculus",
                    question_id=str(q.get("id", "")),
                    title=q.get("title", ""),
                    yes_prob=float(prob),
                    num_forecasters=q.get("number_of_forecasters", 0),
                    close_time=close_time,
                ))
            except Exception:
                continue
        return questions

    async def _fetch_manifold(self) -> list[ForecastQuestion]:
        """Fetch recent active binary markets from Manifold."""
        questions = []
        params = {
            "limit": 200,
            "sort": "score",
            "filter": "open",
            "contractType": "BINARY",
        }
        async with self._session.get(
            f"{MANIFOLD_BASE}/markets", params=params
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()

        for m in (data if isinstance(data, list) else []):
            try:
                prob = m.get("probability")
                if prob is None:
                    continue
                close_raw = m.get("closeTime")
                close_time = None
                if close_raw:
                    close_time = datetime.fromtimestamp(
                        close_raw / 1000, tz=timezone.utc
                    )
                questions.append(ForecastQuestion(
                    source="manifold",
                    question_id=str(m.get("id", "")),
                    title=m.get("question", ""),
                    yes_prob=float(prob),
                    num_forecasters=m.get("uniqueBettorCount", 0),
                    close_time=close_time,
                ))
            except Exception:
                continue
        return questions

    # ------------------------------------------------------------------ #
    #  Matching                                                            #
    # ------------------------------------------------------------------ #

    def match(
        self,
        kalshi_question: str,
        min_similarity: float = 0.25,
        min_forecasters: int = 5,
    ) -> Optional[ForecastQuestion]:
        """
        Find the best-matching forecast question for a Kalshi market.
        Uses Jaccard similarity on 3+ char words.
        Returns None if no match exceeds min_similarity.
        """
        q_words = set(_words(kalshi_question))
        best_score = min_similarity
        best: Optional[ForecastQuestion] = None

        for fq in self._questions:
            if fq.num_forecasters < min_forecasters:
                continue
            fq_words = set(_words(fq.title))
            if not q_words or not fq_words:
                continue
            score = len(q_words & fq_words) / len(q_words | fq_words)
            if score > best_score:
                best_score = score
                best = fq

        return best

    def match_all(
        self,
        kalshi_question: str,
        min_similarity: float = 0.25,
        min_forecasters: int = 5,
        max_results: int = 3,
    ) -> list[tuple[float, ForecastQuestion]]:
        """Return up to max_results matches with their similarity scores."""
        q_words = set(_words(kalshi_question))
        scored = []

        for fq in self._questions:
            if fq.num_forecasters < min_forecasters:
                continue
            fq_words = set(_words(fq.title))
            if not q_words or not fq_words:
                continue
            score = len(q_words & fq_words) / len(q_words | fq_words)
            if score >= min_similarity:
                scored.append((score, fq))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:max_results]

    @property
    def question_count(self) -> int:
        return len(self._questions)


def _words(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z]{3,}", text.lower())
