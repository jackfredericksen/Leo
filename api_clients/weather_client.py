"""
Weather client — fetches forecasts from Open-Meteo (free, no API key).

Open-Meteo provides hourly temperature and precipitation forecasts
out to 16 days with ~1km resolution.

API docs: https://open-meteo.com/en/docs
Base URL: https://api.open-meteo.com/v1/forecast

Supported Kalshi weather market types:
  - High temperature above/below X°F on a specific date
  - Low temperature above/below X°F on a specific date
  - Precipitation above/below X inches in a city/period

Known Kalshi city markets and their lat/lon:
  NYC, Chicago, LA, Miami, Dallas, Phoenix, Seattle, Boston, Denver, DC
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# Cities with known Kalshi weather markets
_CITIES: dict[str, tuple[float, float]] = {
    "new york":     (40.7128, -74.0060),
    "nyc":          (40.7128, -74.0060),
    "chicago":      (41.8781, -87.6298),
    "los angeles":  (34.0522, -118.2437),
    "la":           (34.0522, -118.2437),
    "miami":        (25.7617, -80.1918),
    "dallas":       (32.7767, -96.7970),
    "phoenix":      (33.4484, -112.0740),
    "seattle":      (47.6062, -122.3321),
    "boston":       (42.3601, -71.0589),
    "denver":       (39.7392, -104.9903),
    "washington":   (38.9072, -77.0369),
    "dc":           (38.9072, -77.0369),
    "atlanta":      (33.7490, -84.3880),
    "houston":      (29.7604, -95.3698),
    "minneapolis":  (44.9778, -93.2650),
}

_CACHE_TTL_SEC = 1800   # 30 minutes


@dataclass
class DayForecast:
    city: str
    forecast_date: date
    high_f: float          # daily max temperature °F
    low_f: float           # daily min temperature °F
    precip_in: float       # total precipitation inches
    precip_prob: float     # precipitation probability 0-1
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class WeatherClient:
    """
    Fetches weather forecasts from Open-Meteo for Kalshi weather markets.
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        # cache: city_key → list[DayForecast]
        self._cache: dict[str, list[DayForecast]] = {}
        self._last_fetch: dict[str, datetime] = {}

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    async def refresh_all(self):
        """Fetch forecasts for all known cities."""
        tasks = [
            self._fetch_city(city, lat, lon)
            for city, (lat, lon) in _CITIES.items()
            # deduplicate (nyc and new york point to same coords)
            if city not in ("nyc", "la", "dc")
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(
            f"Weather: refreshed {len(self._cache)} city forecasts"
        )

    async def _fetch_city(
        self, city: str, lat: float, lon: float
    ):
        """Fetch 16-day daily forecast for a city."""
        now = datetime.now(timezone.utc)
        last = self._last_fetch.get(city)
        if last and (now - last).total_seconds() < _CACHE_TTL_SEC:
            return

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
            ],
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": "America/New_York",
            "forecast_days": 16,
        }
        try:
            async with self._session.get(
                OPEN_METEO_BASE, params=params
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            highs = daily.get("temperature_2m_max", [])
            lows = daily.get("temperature_2m_min", [])
            precips = daily.get("precipitation_sum", [])
            precip_probs = daily.get(
                "precipitation_probability_max", []
            )

            forecasts = []
            for i, d in enumerate(dates):
                try:
                    forecasts.append(DayForecast(
                        city=city,
                        forecast_date=date.fromisoformat(d),
                        high_f=float(highs[i] or 0),
                        low_f=float(lows[i] or 0),
                        precip_in=float(precips[i] or 0),
                        precip_prob=float(
                            (precip_probs[i] or 0) / 100
                        ),
                    ))
                except (IndexError, TypeError, ValueError):
                    continue

            self._cache[city] = forecasts
            self._last_fetch[city] = now

        except Exception as e:
            logger.warning(f"Weather fetch failed for {city}: {e}")

    def get_forecast(
        self, city: str, target_date: date
    ) -> Optional[DayForecast]:
        """Get the forecast for a specific city and date."""
        # Try exact city name and common aliases
        for key in [city.lower(), *_ALIASES.get(city.lower(), [])]:
            forecasts = self._cache.get(key, [])
            for f in forecasts:
                if f.forecast_date == target_date:
                    return f
        return None

    def get_range_forecast(
        self,
        city: str,
        start_date: date,
        end_date: date,
    ) -> list[DayForecast]:
        """Get all forecasts for a city between start and end dates."""
        for key in [city.lower(), *_ALIASES.get(city.lower(), [])]:
            forecasts = self._cache.get(key, [])
            result = [
                f for f in forecasts
                if start_date <= f.forecast_date <= end_date
            ]
            if result:
                return result
        return []


# City aliases for matching
_ALIASES: dict[str, list[str]] = {
    "new york":    ["nyc", "new york city"],
    "nyc":         ["new york"],
    "los angeles": ["la"],
    "la":          ["los angeles"],
    "washington":  ["dc", "washington dc"],
    "dc":          ["washington"],
}
