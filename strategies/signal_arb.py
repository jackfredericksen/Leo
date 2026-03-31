"""
External signal arbitrage — compare market implied probability to
real-world data sources to identify mispriced markets.

Sources supported:
  - Crypto price feeds (for markets like "BTC above $X by date")
  - Sports odds APIs (for sports prediction markets)
  - Poll aggregators (for political markets)
  - On-chain metrics via DeFiLlama / Coingecko

The idea:
  1. Fetch a reliable external probability estimate for the same question
  2. Compare to the market's implied probability (= YES price)
  3. If the gap exceeds a threshold (accounting for model uncertainty + fees),
     take a directional position sized by Kelly criterion

This is NOT risk-free arbitrage — it's a positive-EV bet based on
believing your signal over the market consensus. Size conservatively.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from api_clients.kalshi_client import Market
from strategies.kelly import KellySizer, implied_edge

logger = logging.getLogger(__name__)


@dataclass
class SignalArbConfig:
    # Minimum edge (after fees) to act on
    min_edge: float = 0.05
    # Model uncertainty buffer — discount our signal by this much
    uncertainty_buffer: float = 0.05
    fee_pct: float = 0.02
    max_position_usd: float = 50.0
    kelly_fraction: float = 0.10    # Conservative: 10% Kelly for signal bets


@dataclass
class SignalOpportunity:
    market_id: str
    question: str
    market_prob: float       # Market's implied probability
    signal_prob: float       # Our external estimate
    signal_source: str
    edge: float              # Net edge after fees
    recommended_side: str    # "YES" or "NO"
    recommended_size_usd: float
    detected_at: datetime


class CryptoPriceSignal:
    """
    For markets of the form 'Will X be above $Y on date Z?'
    Uses current price + simple vol model to estimate probability.

    This is a placeholder — a production version would use:
      - Historical volatility from OHLCV data
      - Black-Scholes or log-normal distribution
      - Multiple data sources
    """

    def __init__(self):
        self._prices: dict[str, float] = {}

    async def fetch_prices(self, symbols: list[str]):
        """Fetch current prices from CoinGecko (free, no auth required)."""
        ids = ",".join(s.lower() for s in symbols)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    data = await resp.json()
                    for symbol in symbols:
                        key = symbol.lower()
                        if key in data:
                            self._prices[symbol.upper()] = data[key]["usd"]
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed: {e}")

    def estimate_prob(
        self,
        symbol: str,
        target_price: float,
        days_to_resolve: float,
        annual_vol: float = 0.80,  # BTC historical ~80% annualised vol
    ) -> Optional[float]:
        """
        Log-normal probability estimate: P(S_T > K).
        Uses current spot price and annualised volatility.
        """
        import math
        spot = self._prices.get(symbol.upper())
        if not spot:
            return None

        t = days_to_resolve / 365.0
        if t <= 0:
            return 1.0 if spot > target_price else 0.0

        # Log-normal: d2 = (ln(S/K) + (μ - σ²/2)T) / (σ√T)
        # Assume μ = 0 (risk-neutral / no drift assumption)
        sigma = annual_vol
        d2 = (math.log(spot / target_price) - 0.5 * sigma**2 * t) / (sigma * math.sqrt(t))

        # Standard normal CDF approximation
        prob = _norm_cdf(d2)
        return prob


def _norm_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation of the standard normal CDF."""
    import math
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    p = 0.2316419
    k = 1.0 / (1.0 + p * abs(x))
    poly = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2) * poly
    return cdf if x >= 0 else 1.0 - cdf


class SignalArbDetector:
    def __init__(self, cfg: SignalArbConfig, bankroll: float):
        self.cfg = cfg
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)
        self.crypto_signal = CryptoPriceSignal()

    async def refresh_signals(self):
        """Pre-fetch external data before scanning."""
        await self.crypto_signal.fetch_prices(["bitcoin", "ethereum", "solana"])

    def scan(self, markets: list[Market]) -> list[SignalOpportunity]:
        opps = []
        for market in markets:
            if market.status != "open":
                continue
            opp = self._check_crypto_market(market)
            if opp:
                opps.append(opp)
        return sorted(opps, key=lambda o: o.edge, reverse=True)

    def _check_crypto_market(self, market: Market) -> Optional[SignalOpportunity]:
        """
        Parse questions like 'Will BTC be above $X by [date]?' and
        compare to our log-normal probability estimate.
        """
        import re
        from datetime import timezone

        q = market.question
        # Simple pattern match — extend with more patterns as needed
        match = re.search(
            r"(BTC|ETH|SOL|bitcoin|ethereum|solana).*?\$([0-9,]+).*?by",
            q,
            re.IGNORECASE,
        )
        if not match:
            return None

        symbol_map = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL"}
        raw_sym = match.group(1).upper()
        symbol = symbol_map.get(raw_sym.lower(), raw_sym)
        target = float(match.group(2).replace(",", ""))

        now = datetime.now(timezone.utc)
        days_left = (market.close_time - now).total_seconds() / 86400

        signal_prob = self.crypto_signal.estimate_prob(symbol, target, days_left)
        if signal_prob is None:
            return None

        # Apply uncertainty buffer
        adjusted_prob = signal_prob
        market_prob = market.yes_price

        edge = implied_edge(adjusted_prob, market_prob, self.cfg.fee_pct)

        if abs(edge) < self.cfg.min_edge + self.cfg.uncertainty_buffer:
            return None

        if edge > 0:
            side = "YES"
            size = self.sizer.size(adjusted_prob, market_prob, self.cfg.max_position_usd)
        else:
            # Our estimate is lower than market — bet NO
            no_prob = 1.0 - adjusted_prob
            no_price = market.no_price
            side = "NO"
            size = self.sizer.size(no_prob, no_price, self.cfg.max_position_usd)

        if size < 1.0:
            return None

        logger.info(
            f"Signal arb: {q!r} | signal={adjusted_prob:.3f} market={market_prob:.3f} "
            f"edge={edge:.3f} side={side} size=${size:.2f}"
        )

        return SignalOpportunity(
            market_id=market.market_id,
            question=q,
            market_prob=market_prob,
            signal_prob=adjusted_prob,
            signal_source=f"lognormal_{symbol}",
            edge=edge,
            recommended_side=side,
            recommended_size_usd=size,
            detected_at=now,
        )
