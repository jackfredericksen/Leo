"""NewsAPI client used to confirm news-fade signals."""

import logging
import time

import aiohttp

logger = logging.getLogger(__name__)

_BASE = "https://newsapi.org/v2/everything"
_CACHE_TTL = 900.0  # 15 minutes


class NewsClient:
    """
    Wraps newsapi.org /everything to confirm that a news event backing
    a spike actually exists before entering a fade position.

    When `api_key` is empty every call returns True (don't block trades).
    Results are cached per query-string for 15 minutes.
    """

    def __init__(self, api_key: str):
        self._key = api_key
        self._cache: dict[str, tuple[float, bool]] = {}

    async def has_recent_news(self, question: str, hours: float = 4.0) -> bool:
        """
        Return True if at least one article mentions keywords from `question`
        within the last `hours`.  Returns True on any error so trades are not
        accidentally blocked by API failures.
        """
        if not self._key:
            return True

        # Keyword query: first 4 meaningful words
        words = [w for w in question.split() if len(w) >= 4][:4]
        query = " ".join(words)
        if not query:
            return True

        cached = self._cache.get(query)
        if cached:
            ts, result = cached
            if time.monotonic() - ts < _CACHE_TTL:
                return result

        try:
            from datetime import datetime, timedelta, timezone
            since = (
                datetime.now(timezone.utc) - timedelta(hours=hours)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            params = {
                "q": query,
                "from": since,
                "sortBy": "publishedAt",
                "pageSize": 1,
                "apiKey": self._key,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    _BASE,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        return True
                    data = await resp.json()

            found = (data.get("totalResults") or 0) > 0
            self._cache[query] = (time.monotonic(), found)
            return found

        except Exception as e:
            logger.debug(f"NewsClient: {e}")
            return True
