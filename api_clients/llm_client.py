"""
LLM client — uses Claude Haiku to estimate probabilities for Kalshi markets.

Model: claude-haiku-4-5-20251001 (fast, cheap, good at structured output)

The prompt gives Claude the market question, current price, time remaining,
and any available context (recent forecast matches), then asks for:
  1. A probability estimate (0.00-1.00)
  2. A confidence score (0.0-1.0)
  3. A brief reasoning string

Results are cached per market_id for `cache_ttl_min` minutes to avoid
re-querying the same question repeatedly.

Rate limiting: max `max_concurrent` simultaneous requests.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"
_SYSTEM_TEMPLATE = """You are a calibrated probability forecaster analyzing prediction market questions.
Today's date is {date}. Your training data has a knowledge cutoff and may be significantly out of date — treat any facts about current prices, recent news, or ongoing events as potentially stale unless explicitly provided in the context below.

Given a market question and context, respond with a JSON object ONLY — no explanation outside the JSON:
{{
  "probability": <float 0.00-1.00>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence using only information provided in this prompt>"
}}

Guidelines:
- probability: your best estimate of the YES outcome probability
- confidence: how confident you are (0 = total uncertainty, 1 = near-certain)
- Be calibrated — do not anchor too heavily on the current market price
- Use base rates, reference classes, and the provided context
- Do NOT cite facts from training data about current prices or recent events
- If you have very little information, confidence should be low (< 0.3)"""


@dataclass
class LLMEstimate:
    market_id: str
    probability: float
    confidence: float
    reasoning: str
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class LLMClient:
    """
    Async Claude-based probability estimator with caching and rate limiting.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _MODEL,
        max_concurrent: int = 3,
        cache_ttl_min: int = 30,
    ):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._sem = asyncio.Semaphore(max_concurrent)
        self._cache_ttl = timedelta(minutes=cache_ttl_min)
        self._cache: dict[str, LLMEstimate] = {}
        # Spot prices injected by the strategy loop before each batch
        self.spot_prices: dict[str, float] = {}

    def _system_prompt(self) -> str:
        today = date.today().isoformat()
        system = _SYSTEM_TEMPLATE.format(date=today)
        if self.spot_prices:
            lines = ", ".join(
                f"{sym.replace('USDT','')}=${price:,.0f}"
                for sym, price in self.spot_prices.items()
            )
            system += f"\n\nCurrent live spot prices: {lines}"
        return system

    def _cached(self, market_id: str) -> Optional[LLMEstimate]:
        est = self._cache.get(market_id)
        if not est:
            return None
        age = datetime.now(timezone.utc) - est.fetched_at
        if age > self._cache_ttl:
            del self._cache[market_id]
            return None
        return est

    async def estimate(
        self,
        market_id: str,
        question: str,
        current_prob: float,
        hours_remaining: float,
        context: str = "",
    ) -> Optional[LLMEstimate]:
        """
        Ask Claude to estimate the probability for a single market.
        Returns cached result if within TTL.
        """
        cached = self._cached(market_id)
        if cached:
            return cached

        prompt = (
            f"Market question: {question}\n"
            f"Current market price (implied probability): {current_prob:.1%}\n"
            f"Hours until resolution: {hours_remaining:.1f}\n"
        )
        if context:
            prompt += f"Additional context: {context}\n"
        prompt += "\nProvide your probability estimate as JSON."

        async with self._sem:
            try:
                msg = await self._client.messages.create(
                    model=self._model,
                    max_tokens=256,
                    system=self._system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                )
                text = msg.content[0].text.strip()
                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                data = json.loads(text)
                prob = float(data["probability"])
                conf = float(data["confidence"])
                reasoning = str(data.get("reasoning", ""))

                # Sanity bounds
                prob = max(0.01, min(0.99, prob))
                conf = max(0.0, min(1.0, conf))

                est = LLMEstimate(
                    market_id=market_id,
                    probability=prob,
                    confidence=conf,
                    reasoning=reasoning,
                )
                self._cache[market_id] = est
                return est

            except Exception as e:
                logger.warning(
                    f"LLM estimate failed for {market_id}: {e}"
                )
                return None

    async def estimate_batch(
        self,
        markets: list[dict],   # list of {market_id, question, prob, hours}
        max_markets: int = 20,
    ) -> list[LLMEstimate]:
        """Estimate probabilities for a batch of markets concurrently."""
        tasks = []
        for m in markets[:max_markets]:
            tasks.append(self.estimate(
                market_id=m["market_id"],
                question=m["question"],
                current_prob=m["prob"],
                hours_remaining=m["hours"],
                context=m.get("context", ""),
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, LLMEstimate)]

    @property
    def cache_size(self) -> int:
        return len(self._cache)
