"""
Strategy: Weather market signal trading.

Kalshi lists daily/weekly weather markets in major US cities:
  "Will NYC high temperature exceed 75°F on April 15?"
  "Will Chicago precipitation exceed 0.50 inches this week?"

We compare Open-Meteo's deterministic forecast + spread model against
Kalshi's implied probability and trade when they diverge.

Probability estimation:
  - Temperature markets: use forecast value + model uncertainty to
    build a normal distribution, compute P(high > threshold).
  - Precipitation markets: use the forecast precip_prob directly,
    adjusted by magnitude (if forecast is 0.1in and threshold is 0.5in,
    even with 60% chance of rain it's unlikely to hit threshold).

Question parsing:
  Kalshi weather tickers look like: HIGHNYC-26APR15, LOWCHI-26APR15
  Subtitles: "NYC high temperature above 75°F"
"""

import logging
import math
import re
from datetime import date, datetime, timezone
from typing import Optional

from api_clients.kalshi_client import Market
from api_clients.weather_client import DayForecast, WeatherClient
from strategies.signal_arb import AggregatedSignal, SignalArbConfig, ask_edge
from strategies.kelly import KellySizer

logger = logging.getLogger(__name__)

# Kalshi weather ticker prefixes and how to parse them
# HIGH{CITY}-{DATE}, LOW{CITY}-{DATE}, PRECIP{CITY}-{DATE}
_WEATHER_RE = re.compile(
    r"^(HIGH|LOW|PRECIP|RAIN|SNOW|TEMP)([A-Z]{2,6})-",
    re.IGNORECASE,
)

# Map Kalshi city codes → canonical city name for weather client
_CITY_CODES: dict[str, str] = {
    "NYC":  "new york",
    "NY":   "new york",
    "CHI":  "chicago",
    "LA":   "los angeles",
    "LAX":  "los angeles",
    "MIA":  "miami",
    "DAL":  "dallas",
    "DFW":  "dallas",
    "PHX":  "phoenix",
    "SEA":  "seattle",
    "BOS":  "boston",
    "DEN":  "denver",
    "DC":   "washington",
    "ATL":  "atlanta",
    "HOU":  "houston",
    "MSP":  "minneapolis",
    "MIN":  "minneapolis",
}

# Temperature forecast uncertainty (°F std dev)
# Accounts for model error + day-of uncertainty
_TEMP_SIGMA_BY_DAYS_OUT = [
    (0,  2.0),   # same day   ± 2°F
    (1,  3.0),   # 1 day out  ± 3°F
    (3,  4.5),   # 3 days out ± 4.5°F
    (7,  6.0),   # 1 week out ± 6°F
    (14, 8.0),   # 2 weeks    ± 8°F
]


def _temp_sigma(days_out: int) -> float:
    for threshold, sigma in reversed(_TEMP_SIGMA_BY_DAYS_OUT):
        if days_out >= threshold:
            return sigma
    return 4.0


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun approximation)."""
    a1, a2, a3, a4, a5 = (
        0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    )
    p = 0.2316419
    k = 1.0 / (1.0 + p * abs(x))
    poly = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
    cdf = 1.0 - (1.0 / (2 * math.pi) ** 0.5) * math.exp(-0.5 * x ** 2) * poly
    return float(cdf if x >= 0 else 1.0 - cdf)


def _prob_temp_exceeds(
    forecast_f: float, threshold_f: float, days_out: int
) -> float:
    """P(actual temp > threshold) given forecast and uncertainty."""
    sigma = _temp_sigma(days_out)
    z = (forecast_f - threshold_f) / sigma
    return _norm_cdf(z)


def _prob_precip_exceeds(
    forecast_precip: float,
    forecast_prob: float,
    threshold: float,
) -> float:
    """
    P(precipitation > threshold).

    Combines:
      1. forecast_prob: model's P(any precipitation)
      2. Given it rains, P(amount > threshold) using exponential distribution
    """
    if forecast_prob < 0.05:
        return 0.02   # model says almost certainly dry

    # If it does rain, model the amount as exponential with mean = forecast
    # P(X > threshold | X ~ Exp(1/mean)) = exp(-threshold/mean)
    if forecast_precip <= 0:
        # Model says some precip is possible but forecasts near zero
        return forecast_prob * 0.05

    rate = 1.0 / forecast_precip
    p_exceeds_given_rain = math.exp(-rate * threshold)
    return forecast_prob * p_exceeds_given_rain


def _parse_weather_market(
    market: Market,
) -> Optional[tuple[str, str, float, date]]:
    """
    Parse a Kalshi weather market into (city, metric, threshold, target_date).
    metric: "high_temp" | "low_temp" | "precip"

    Parses from subtitle_yes when available, falls back to question text.
    """
    text = market.subtitle_yes or market.question or ""
    tid = market.market_id.upper()

    # --- Try ticker prefix for city code ---
    city = None
    metric = None
    m = _WEATHER_RE.match(tid)
    if m:
        kind = m.group(1).upper()
        code = m.group(2).upper()
        city = _CITY_CODES.get(code)
        if kind in ("HIGH", "TEMP"):
            metric = "high_temp"
        elif kind == "LOW":
            metric = "low_temp"
        elif kind in ("PRECIP", "RAIN", "SNOW"):
            metric = "precip"

    # --- Fallback: parse city from question text ---
    if not city:
        text_lower = text.lower()
        for code, city_name in _CITY_CODES.items():
            if code.lower() in text_lower or city_name in text_lower:
                city = city_name
                break

    if not city:
        return None

    # --- Parse metric from subtitle if not already set ---
    if not metric:
        text_lower = text.lower()
        if any(w in text_lower for w in ["high", "maximum", "max"]):
            metric = "high_temp"
        elif any(w in text_lower for w in ["low", "minimum", "min"]):
            metric = "low_temp"
        elif any(w in text_lower for w in ["rain", "precip", "snow", "inch"]):
            metric = "precip"
        else:
            return None

    # --- Parse threshold (°F or inches) ---
    threshold_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:°?[Ff]|degrees?|inches?|in\b)", text
    )
    if not threshold_match:
        # Try bare number preceded by "above", "exceed", "below", "under"
        threshold_match = re.search(
            r"(?:above|exceed|below|under)\s+(\d+(?:\.\d+)?)", text
        )
    if not threshold_match:
        return None
    threshold = float(threshold_match.group(1))

    # --- Parse target date from close_time ---
    # Kalshi weather markets close at or just after the target date
    target_date = market.close_time.date()

    return city, metric, threshold, target_date


class WeatherSignalDetector:
    """
    Finds edge in Kalshi temperature and precipitation markets
    using Open-Meteo forecasts.
    """

    def __init__(
        self,
        cfg: SignalArbConfig,
        weather: WeatherClient,
        bankroll: float,
    ):
        self.cfg = cfg
        self.weather = weather
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)

    def scan(self, markets: list[Market]) -> list[AggregatedSignal]:
        results = []
        today = datetime.now(timezone.utc).date()

        for market in markets:
            try:
                if market.status != "open":
                    continue

                parsed = _parse_weather_market(market)
                if not parsed:
                    continue
                city, metric, threshold, target_date = parsed

                days_out = (target_date - today).days
                if days_out < 0 or days_out > 14:
                    continue

                forecast = self.weather.get_forecast(city, target_date)
                if not forecast:
                    continue

                model_prob = self._estimate_prob(
                    metric, threshold, forecast, days_out
                )
                if model_prob is None:
                    continue

                # Determine if market is YES = above or YES = below
                sub = (market.subtitle_yes or market.question or "").lower()
                yes_is_above = any(
                    w in sub for w in ["above", "exceed", "over", "high"]
                )
                if not yes_is_above:
                    model_prob = 1.0 - model_prob

                result = ask_edge(
                    model_prob,
                    market.yes_ask, market.no_ask, market.yes_bid,
                    self.cfg.fee_pct, self.cfg.min_edge,
                )
                if not result:
                    continue
                edge, side, entry_price = result

                # Confidence: higher for near-term, lower for far-out
                confidence = max(0.3, 1.0 - days_out * 0.05)

                trade_prob = model_prob if side == "yes" else (1 - model_prob)
                size = self.sizer.size(
                    trade_prob, entry_price, self.cfg.max_position_usd
                )

                results.append(AggregatedSignal(
                    market_id=market.market_id,
                    question=market.question or market.market_id,
                    market_prob=market.yes_price,
                    model_prob=model_prob,
                    edge=edge,
                    recommended_side=side,
                    source=f"weather:{city.replace(' ', '_')}",
                    confidence=confidence,
                    reasoning=self._reasoning(
                        metric, threshold, forecast, days_out, model_prob
                    ),
                    recommended_size_usd=size,
                ))

            except Exception as e:
                logger.debug(
                    f"WeatherSignal error on {market.market_id}: {e}"
                )

        return sorted(results, key=lambda x: abs(x.edge), reverse=True)

    def _estimate_prob(
        self,
        metric: str,
        threshold: float,
        forecast: DayForecast,
        days_out: int,
    ) -> Optional[float]:
        if metric == "high_temp":
            return _prob_temp_exceeds(forecast.high_f, threshold, days_out)
        elif metric == "low_temp":
            return _prob_temp_exceeds(forecast.low_f, threshold, days_out)
        elif metric == "precip":
            return _prob_precip_exceeds(
                forecast.precip_in, forecast.precip_prob, threshold
            )
        return None

    def _reasoning(
        self,
        metric: str,
        threshold: float,
        forecast: DayForecast,
        days_out: int,
        model_prob: float,
    ) -> str:
        if metric == "high_temp":
            return (
                f"forecast high={forecast.high_f:.1f}°F "
                f"threshold={threshold}°F "
                f"σ={_temp_sigma(days_out):.1f}°F "
                f"days_out={days_out} "
                f"model={model_prob:.1%}"
            )
        elif metric == "low_temp":
            return (
                f"forecast low={forecast.low_f:.1f}°F "
                f"threshold={threshold}°F "
                f"σ={_temp_sigma(days_out):.1f}°F "
                f"days_out={days_out} "
                f"model={model_prob:.1%}"
            )
        else:
            return (
                f"forecast precip={forecast.precip_in:.2f}in "
                f"prob={forecast.precip_prob:.0%} "
                f"threshold={threshold}in "
                f"model={model_prob:.1%}"
            )
