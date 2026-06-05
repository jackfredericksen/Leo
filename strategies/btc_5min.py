"""
BTC 5-Min Directional Strategy

Targets Polymarket's rolling 5-minute BTC up/down markets.

Probability model: Geometric Brownian Motion convergence
──────────────────────────────────────────────────────────
The market resolves YES if BTC is above its window-open price at close.
At any moment during the window, the true probability is:

    P(win) = N(d₂)
    d₂ = [Δ - (σ²/2)·T] / (σ·√T)

where Δ = current log-return from window open, T = time remaining (years),
σ = annualized realized vol. With T=30s and Δ=+0.10%, d₂ ≈ 1.66 → P ≈ 95%.

This replaces the tanh heuristic with a mathematically principled estimate
that naturally accounts for both the magnitude of the BTC move AND how much
time remains for a reversal.

Secondary signal adjustments (±8% max, scaled by remaining uncertainty):
  1. MTF Momentum — multi-timeframe weighted momentum, v3 weights
  2. OBI          — only in final 60s AND |imbalance| ≥ 0.15
  3. RSI (9)      — only extreme readings (>70 overbought / <30 oversold)

Entry window: 0.2 ≤ minutes_to_close ≤ 1.5
  Most edge is in the final 60–90s when direction is largely locked in.

Taker entry price cap: $0.88
  At p=0.90 as a taker, one loss requires 13+ wins to break even due to
  the asymmetric payout. Never enter at extreme prices as a taker.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from strategies.kelly import KellySizer
from strategies.signal_arb import AggregatedSignal

logger = logging.getLogger(__name__)


# Abramowitz & Stegun approximation for standard normal CDF
def _norm_cdf(x: float) -> float:
    a1, a2, a3, a4, a5 = (
        0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    )
    p = 0.2316419
    k = 1.0 / (1.0 + p * abs(x))
    poly = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2) * poly
    return float(cdf if x >= 0 else 1.0 - cdf)


@dataclass
class BTC5MinConfig:
    enabled: bool = True
    min_edge: float = 0.04
    max_position_usd: float = 50.0
    kelly_fraction: float = 0.08
    # Late entry: most edge is in the final 60–90 seconds
    min_mins_to_close: float = 0.2       # ~12s blockchain confirmation floor
    max_mins_to_close: float = 1.5
    fee_pct: float = 0.02
    # Taker entry price cap: above $0.88 the asymmetric payout breaks the math
    max_taker_entry: float = 0.88
    obi_levels: int = 10
    # B-S model fallback vol (used when realized vol is unavailable)
    default_annual_vol: float = 0.80
    # Secondary signal weights (apply as small ±8% adjustments to B-S prob)
    mtf_momentum_weight: float = 0.55
    obi_weight: float = 0.25
    rsi_weight: float = 0.20
    # OBI gate: only active in final minute with strong imbalance
    obi_late_window_mins: float = 1.0
    obi_min_signal: float = 0.15
    # RSI: period 9 (faster), extreme readings only
    rsi_period: int = 9
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    refresh_interval_sec: int = 20


def _is_btc_5min(question: str, slug: str = "") -> bool:
    """
    Match Polymarket 5-minute BTC up/down rolling markets.

    Slug check covers known prefixes (btc-updown-5m, btc-5m) then falls
    through to keyword matching so slug presence never silently blocks a match.
    """
    sl = slug.lower()
    if sl:
        if sl.startswith(("btc-updown-5m", "btc-5m", "bitcoin-5m")):
            return True
        # Slug present but didn't match known prefix — still try keywords
        if ("btc" in sl or "bitcoin" in sl) and ("5m" in sl or "5-min" in sl):
            return True
    q = question.lower()
    has_btc = "bitcoin" in q or "btc" in q
    has_5min = "5 min" in q or "5-min" in q or "5min" in q or "5 minute" in q
    return has_btc and has_5min


def _mins_to_close(close_time: datetime) -> float:
    return (close_time - datetime.now(timezone.utc)).total_seconds() / 60


class BTC5MinDetector:
    def __init__(self, cfg: BTC5MinConfig, binance, pyth=None, bankroll: float = 500.0):
        self._cfg = cfg
        self._binance = binance
        self._pyth = pyth
        self.sizer = KellySizer(bankroll=bankroll, fraction=cfg.kelly_fraction)
        self._last_signals: dict = {}
        self._window_open_prices: dict[str, float] = {}
        self._cached_slow_signals: dict = {}
        self._btc_markets: list = []

    # ──────────────────────────────────────────────────
    # B-S convergence probability
    # ──────────────────────────────────────────────────

    @staticmethod
    def _bs_prob_up(
        window_delta_frac: float,
        mins_to_close: float,
        annual_vol: float,
    ) -> float:
        """
        P(BTC remains above window-open at resolution) using GBM.

        d₂ = [Δ - (σ²/2)·T] / (σ·√T)
        P  = N(d₂)

        Example: T=30s, Δ=+0.10%, σ=0.80 annualized
          T_yr = 30 / 31_557_600 = 9.5e-7
          d₂   = (0.001 - 0.32 × 9.5e-7) / (0.80 × 9.7e-4) ≈ 1.66
          P    ≈ 0.951
        """
        T = max(mins_to_close / (365.0 * 24.0 * 60.0), 1e-10)
        sqrt_T = math.sqrt(T)
        d2 = (window_delta_frac - 0.5 * annual_vol**2 * T) / (annual_vol * sqrt_T)
        return _norm_cdf(d2)

    # ──────────────────────────────────────────────────
    # Window-open price lookup
    # ──────────────────────────────────────────────────

    async def _get_window_open_price(self, market) -> Optional[float]:
        """
        BTC price at the start of this 5-min window (close_time - 5m). Cached per market.
        Tries Pyth oracle first (matches Polymarket's resolution feed),
        falls back to Binance candle.
        """
        mid = market.market_id
        if mid in self._window_open_prices:
            return self._window_open_prices[mid]
        window_open_dt = market.close_time - timedelta(minutes=5)
        if self._pyth:
            price = await self._pyth.get_price_at(window_open_dt)
            if price and price > 0:
                self._window_open_prices[mid] = price
                return price
        price = self._binance.get_candle_close_at("BTCUSDT", window_open_dt)
        if price:
            self._window_open_prices[mid] = price
        else:
            candle_count = len(self._binance._candles.get("BTCUSDT", []))
            logger.warning(
                f"BTC5Min: no window-open price at "
                f"{window_open_dt.strftime('%H:%M:%S')} UTC "
                f"for {mid[:12]} — candle buffer has {candle_count} entries. "
                f"B-S model will fall back to secondary signals only."
            )
        return price

    # ──────────────────────────────────────────────────
    # Signal composition
    # ──────────────────────────────────────────────────

    def _compute_prob_up(
        self,
        window_delta: Optional[float],
        mtf_momentum: Optional[float],
        obi: Optional[float],
        rsi: Optional[float],
        mins_to_close: float,
        annual_vol: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Returns (composite, prob_up).

        When window_delta is available:
          - prob_bs from B-S model is the primary estimate
          - MTF/OBI/RSI apply a small ±8% secondary adjustment,
            scaled down when B-S is already highly confident
        When window_delta is unavailable:
          - Falls back to tanh-composite from the secondary signals only
        """
        cfg = self._cfg

        # ── Secondary signal scores ────────────────────────────────────────
        mtf_score = mtf_momentum if mtf_momentum is not None else 0.0
        mtf_w = cfg.mtf_momentum_weight if mtf_momentum is not None else 0.0

        rsi_score = 0.0
        if rsi is not None:
            if rsi > cfg.rsi_overbought:
                rsi_score = -((rsi - cfg.rsi_overbought) / (100.0 - cfg.rsi_overbought))
            elif rsi < cfg.rsi_oversold:
                rsi_score = (cfg.rsi_oversold - rsi) / cfg.rsi_oversold
        rsi_w = cfg.rsi_weight  # weight present; neutral zone contributes 0

        obi_active = (
            obi is not None
            and mins_to_close <= cfg.obi_late_window_mins
            and abs(obi) >= cfg.obi_min_signal
        )
        obi_score = obi if obi_active else 0.0
        obi_w = cfg.obi_weight if obi_active else 0.0

        sec_total_w = mtf_w + obi_w + rsi_w

        # ── Primary: B-S convergence model ────────────────────────────────
        if window_delta is not None:
            prob_bs = self._bs_prob_up(window_delta, mins_to_close, annual_vol)

            # Secondary adjustment: ±8% max, scaled by remaining uncertainty
            # (adjustment shrinks to zero when B-S is near 0 or 1)
            if sec_total_w > 0:
                sec_score = (
                    mtf_w * mtf_score + obi_w * obi_score + rsi_w * rsi_score
                ) / sec_total_w
            else:
                sec_score = 0.0

            # confidence_scalar: 1.0 at 50% (max uncertainty), 0.0 at certainty
            confidence_scalar = 1.0 - abs(prob_bs - 0.5) * 2.0
            adjustment = math.tanh(0.60 * sec_score) * 0.08 * confidence_scalar
            prob_up = max(0.01, min(0.99, prob_bs + adjustment))

            # composite = deviation from 50% for display / reasoning
            composite = prob_up - 0.5
            return composite, prob_up

        # ── Fallback: secondary signals only (no window delta) ─────────────
        if sec_total_w == 0:
            return None, None
        composite = (
            mtf_w * mtf_score + obi_w * obi_score + rsi_w * rsi_score
        ) / sec_total_w
        prob_up = 0.5 + math.tanh(0.80 * composite) * 0.25
        return composite, prob_up

    # ──────────────────────────────────────────────────
    # Main scan
    # ──────────────────────────────────────────────────

    async def scan(self, markets: list) -> list[AggregatedSignal]:
        cfg = self._cfg

        # Identify all BTC 5-min markets (regardless of time window)
        btc_markets = [
            m for m in markets
            if _is_btc_5min(m.question, getattr(m, "slug", ""))
        ]
        candidates = [
            m for m in btc_markets
            if cfg.min_mins_to_close
               <= _mins_to_close(m.close_time)
               <= cfg.max_mins_to_close
        ]

        self._btc_markets = btc_markets

        if btc_markets:
            mins_list = [round(_mins_to_close(m.close_time), 2) for m in btc_markets]
            logger.info(
                f"BTC5Min: {len(btc_markets)} BTC markets found "
                f"(mins-to-close: {mins_list}), "
                f"{len(candidates)} in entry window "
                f"[{cfg.min_mins_to_close}–{cfg.max_mins_to_close} min]"
            )
        else:
            logger.info(
                f"BTC5Min: 0 BTC 5-min markets detected in {len(markets)} total markets"
            )

        # Market-independent signals (fetch once)
        obi = await self._binance.compute_obi("BTCUSDT", levels=cfg.obi_levels)
        mtf_momentum = self._binance.compute_mtf_momentum("BTCUSDT")
        rsi = self._binance.compute_rsi("BTCUSDT", period=cfg.rsi_period)
        current_btc = (
            (self._pyth.get_price() if self._pyth else None)
            or self._binance.get_price("BTCUSDT")
        )
        price_source = "pyth" if (self._pyth and self._pyth.get_price()) else "binance"
        annual_vol = (
            self._binance.compute_realized_vol_blended("BTCUSDT")
            or cfg.default_annual_vol
        )

        # Cache for fast_scan() between slow-scan cycles
        self._cached_slow_signals = {
            "obi": obi,
            "mtf_momentum": mtf_momentum,
            "rsi": rsi,
            "annual_vol": annual_vol,
        }

        candle_count = len(self._binance._candles.get("BTCUSDT", []))
        logger.debug(
            f"BTC5Min signals — price={current_btc} ({price_source}), vol={annual_vol:.2f}, "
            f"rsi={rsi}, mtf={mtf_momentum}, obi={obi}, candles={candle_count}"
        )

        # Summary composite for the dashboard widget (uses first candidate or 0)
        summary_window_delta = None
        summary_mins = 0.0
        if candidates and current_btc:
            ref = await self._get_window_open_price(candidates[0])
            if ref and ref > 0:
                summary_window_delta = (current_btc - ref) / ref
            summary_mins = _mins_to_close(candidates[0].close_time)

        composite, prob_up = self._compute_prob_up(
            summary_window_delta, mtf_momentum, obi, rsi, summary_mins, annual_vol
        )

        self._last_signals = {
            "obi": obi,
            "momentum": mtf_momentum,       # "momentum" key for widget compat
            "rsi": rsi,
            "window_delta": (
                round(summary_window_delta * 100, 4)
                if summary_window_delta is not None else None
            ),  # stored as percentage (0.08 means +0.08%)
            "annual_vol": round(annual_vol, 4),
            "current_btc": current_btc,
            "price_source": price_source,
            "composite": round(composite, 4) if composite is not None else None,
            "prob_up": round(prob_up, 4) if prob_up is not None else None,
            "active_markets": len(candidates),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if not candidates:
            return []

        results = []
        for market in candidates:
            mins = _mins_to_close(market.close_time)
            window_delta = None
            if current_btc:
                ref = await self._get_window_open_price(market)
                if ref and ref > 0:
                    window_delta = (current_btc - ref) / ref

            comp, pup = self._compute_prob_up(
                window_delta, mtf_momentum, obi, rsi, mins, annual_vol
            )
            sig = self._evaluate(
                market, window_delta, mtf_momentum, obi, rsi,
                comp, pup, mins, annual_vol
            )
            if sig:
                results.append(sig)

        results.sort(key=lambda s: s.edge, reverse=True)
        return results

    # ──────────────────────────────────────────────────
    # Fast re-evaluation (no I/O)
    # ──────────────────────────────────────────────────

    def fast_scan(self) -> list[AggregatedSignal]:
        """
        Re-evaluate cached BTC markets with fresh Pyth price. No I/O.

        Called every 2s by btc_5min_fast_loop. Uses:
          - self._pyth.get_price()         — latest SSE tick (~400ms freshness)
          - self._cached_slow_signals      — OBI/MTF/RSI from last full scan()
          - self._window_open_prices       — per-market cache from scan()
          - self._btc_markets              — all BTC 5-min markets from scan()
        """
        if not self._pyth or not self._cached_slow_signals or not self._btc_markets:
            return []

        current_btc = self._pyth.get_price()
        if not current_btc:
            return []

        sig = self._cached_slow_signals
        obi = sig.get("obi")
        mtf_momentum = sig.get("mtf_momentum")
        rsi = sig.get("rsi")
        annual_vol = sig.get("annual_vol", self._cfg.default_annual_vol)

        results = []
        for market in self._btc_markets:
            mins = _mins_to_close(market.close_time)
            if not (self._cfg.min_mins_to_close <= mins <= self._cfg.max_mins_to_close):
                continue

            ref = self._window_open_prices.get(market.market_id)
            window_delta = (current_btc - ref) / ref if (ref and ref > 0) else None

            comp, pup = self._compute_prob_up(
                window_delta, mtf_momentum, obi, rsi, mins, annual_vol
            )
            result = self._evaluate(
                market, window_delta, mtf_momentum, obi, rsi,
                comp, pup, mins, annual_vol
            )
            if result:
                results.append(result)

        results.sort(key=lambda s: s.edge, reverse=True)
        return results

    # ──────────────────────────────────────────────────
    # Per-market signal evaluation
    # ──────────────────────────────────────────────────

    def _evaluate(
        self,
        market,
        window_delta: Optional[float],
        mtf_momentum: Optional[float],
        obi: Optional[float],
        rsi: Optional[float],
        composite: Optional[float],
        prob_up: Optional[float],
        mins_to_close: float,
        annual_vol: float,
    ) -> Optional[AggregatedSignal]:
        cfg = self._cfg

        if composite is None or prob_up is None:
            return None

        yes_ask = getattr(market, "yes_ask", 0.0)
        no_ask  = getattr(market, "no_ask",  0.0)
        if not yes_ask or not no_ask:
            return None

        yes_price = getattr(market, "yes_price", yes_ask)

        if prob_up >= 0.5:
            side = "yes"
            market_prob = yes_price
            true_prob = prob_up
            edge = prob_up - yes_ask - cfg.fee_pct
            entry_price = yes_ask
        else:
            side = "no"
            market_prob = yes_price
            true_prob = 1.0 - prob_up
            edge = true_prob - no_ask - cfg.fee_pct
            entry_price = no_ask

        if edge < cfg.min_edge:
            return None

        # Taker entry price cap: above $0.88 the asymmetric payout breaks the math.
        # At p=0.90, one loss erases 13+ wins — don't enter as taker at extreme prices.
        if entry_price > cfg.max_taker_entry:
            return None

        size = self.sizer.size(true_prob, entry_price, cfg.max_position_usd)

        wd_str = f"Δ={window_delta*100:+.3f}%" if window_delta is not None else "Δ=—"
        bs_str = (
            f" P_bs={self._bs_prob_up(window_delta, mins_to_close, annual_vol):.0%}"
            if window_delta is not None else ""
        )
        mtf_str = f" MTF={mtf_momentum:+.3f}" if mtf_momentum is not None else ""
        obi_str = f" OBI={obi:+.3f}" if obi is not None else ""
        rsi_str = f" RSI={rsi:.0f}" if rsi is not None else ""
        reasoning = f"{wd_str}{bs_str}{mtf_str}{obi_str}{rsi_str}"

        # Confidence reflects certainty of B-S estimate + whether we had window delta
        confidence = min(1.0, abs(composite) * 2.5)
        if window_delta is None:
            confidence *= 0.6

        return AggregatedSignal(
            market_id=market.market_id,
            question=market.question,
            market_prob=market_prob,
            model_prob=prob_up,
            edge=edge,
            recommended_side=side,
            source="btc_5min",
            confidence=confidence,
            reasoning=reasoning,
            recommended_size_usd=size,
            slug=getattr(market, "slug", ""),
            detected_at=datetime.now(timezone.utc),
        )
