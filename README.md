# Leo — Kalshi Prediction Market Trading Bot

Leo is an automated trading bot for [Kalshi](https://kalshi.com) prediction markets. It runs seven independent strategies in parallel, each finding and executing edges through different signals — price discrepancies, community forecasts, weather data, correlated market logic, and LLM analysis. All order execution goes exclusively through Kalshi; external platforms (Polymarket, Metaculus, Manifold, Binance) are used as price signals only.

---

## Strategies

### 1. Overround Arbitrage
Kalshi binary markets have a YES side and a NO side. When `YES_bid + NO_bid > 1.00`, selling both sides simultaneously locks in a guaranteed profit regardless of outcome. Leo scans all open markets every few seconds, detects these overround conditions, and executes both legs.

### 2. Crypto Price Signal
BTC, ETH, and SOL price markets on Kalshi (e.g. "Will BTC close above $90,000 this week?") are priced against a log-normal model derived from live Binance spot prices and realized volatility computed from 1-minute candles. When the market price diverges materially from the model probability, Leo trades the cheap side.

### 3. Range Straddle
Kalshi lists weekly range contracts: "Will BTC close between $90,000–$100,000?". Leo uses a Black-Scholes-style d1/d2 model — live spot + realized vol — to estimate the probability each range bracket resolves YES, and trades when the market is mispriced relative to the model.

### 4. Correlated / Logical Arbitrage
Some markets have hard logical relationships: if "BTC above $80k in January" is trading at 80%, then "BTC above $80k in February" can't rationally be lower than that. Leo auto-discovers these monotone relationships within Kalshi event groups (e.g. multiple strike levels for the same underlying) and trades the legs where prices violate the implied ordering.

### 5. News Fade
When a market spikes sharply (≥12% in one scan cycle) in response to a news event, Leo records the spike and, 1.5–4 hours later, fades it back toward fair value — betting that short-term overreaction has not been fully corrected. Spike detection is cross-cycle: the bot compares the current price to the price from the previous scan, not to internal fields from the same API snapshot.

### 6. Cross-Platform Signal (Polymarket)
Leo fetches active Polymarket binary markets and fuzzy-matches them to Kalshi markets on the same question. When the implied probabilities diverge by more than the minimum profit threshold, it executes on Kalshi using the Polymarket price as the fair-value anchor. Polymarket is a signal source only — no orders are placed there.

### 7. Forecast Aggregator (Metaculus + Manifold)
Leo pulls the 200 most active open binary questions from both Metaculus and Manifold Markets and matches them to Kalshi markets using Jaccard word-set similarity. When the community-weighted consensus diverges from the Kalshi price by more than the minimum edge, Leo trades. Match confidence is scaled by the number of forecasters and whether both sources agree.

### 8. LLM Fundamental Analysis (Claude Haiku)
For qualitative markets that rule-based models can't price — elections, policy decisions, geopolitical events — Leo asks Claude Haiku to estimate a probability. Questions are enriched with any matching Metaculus/Manifold context before being sent to the model. The response is parsed for `{probability, confidence, reasoning}`; position size is scaled by the model's stated confidence.

### 9. Weather Signal (Open-Meteo)
Kalshi lists daily temperature and precipitation markets for major US cities (e.g. "Will NYC high temperature exceed 75°F on April 15?"). Leo fetches 16-day forecasts from Open-Meteo (free, no API key) and computes:
- **Temperature**: P(actual > threshold) using a normal distribution where σ grows with days out (±2°F same-day to ±8°F at two weeks)
- **Precipitation**: P(precip > threshold) using P(any rain) × exponential distribution for the amount

---

## Architecture

```
main.py              — Orchestrator: asyncio task loop + Rich terminal UI + web dashboard
config.py            — All settings as dataclasses, loaded from environment variables
trader.py            — Order execution: execute(), execute_signal(), execute_correlated()
arbitrage.py         — Overround arb scanner
auto_correlator.py   — Discovers logical market relationships within Kalshi event groups
position_manager.py  — Tracks live positions and P&L via Kalshi portfolio API
storage.py           — SQLite persistence (leo.db): trades, positions, scan history
bot_state.py         — Shared state bridge for the web dashboard
web_gui.py           — FastAPI + WebSocket web dashboard (default port 5002)

api_clients/
  kalshi_client.py   — Kalshi REST API (RSA-PSS auth, rate limiting)
  binance_client.py  — Binance spot prices + 1-min OHLCV + realized vol
  polymarket_client.py — Polymarket Gamma API (signal only)
  forecast_client.py — Metaculus + Manifold Markets (community forecasts)
  llm_client.py      — Anthropic Claude (async, semaphore-limited, TTL cache)
  weather_client.py  — Open-Meteo 16-day forecast (30-min TTL cache)

strategies/
  signal_arb.py      — AggregatedSignal dataclass + SignalArbConfig
  crypto_signal.py   — BTC/ETH/SOL log-normal price signal
  range_straddle.py  — Crypto range bracket pricing
  correlated.py      — Logical/correlated arb execution
  news_fade.py       — News spike fade
  cross_platform.py  — Polymarket ↔ Kalshi cross-platform signal
  forecast_signal.py — Metaculus/Manifold weighted consensus
  llm_signal.py      — Claude Haiku fundamental analysis
  weather_signal.py  — Open-Meteo temperature/precipitation signal
  kelly.py           — Fractional Kelly position sizing
```

### Execution flow

Each strategy loop runs independently via `asyncio.create_task`. All loops read from a shared `state.markets` list that is refreshed every 30 seconds by a dedicated `market_refresh_loop`. When a loop finds an opportunity, it calls one of three execution paths on `Trader`:

- `trader.execute(opp)` — two-leg overround arb
- `trader.execute_signal(sig, market_map)` — single-leg signal trade
- `trader.execute_correlated(opp, market_map)` — correlated arb single leg

All execution paths respect a **30-minute per-market cooldown** to prevent multiple strategies from piling into the same market in one cycle. Position size is determined by the **fractional Kelly criterion** with a configurable fraction (default 0.10).

---

## Setup

### Requirements

- Python 3.11+
- Kalshi account with API key pair (RSA): [app.kalshi.com/profile/api-keys](https://app.kalshi.com/profile/api-keys)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p data
```

### Environment variables

Create a `.env` file in the project root:

```env
# --- Required ---
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----

# --- Mode ---
DRY_RUN=true                  # Set false to place real orders
KALSHI_USE_DEMO=false         # Set true to use Kalshi demo environment

# --- Optional: LLM strategy ---
LLM_ENABLED=false             # Set true to enable Claude Haiku analysis
ANTHROPIC_API_KEY=sk-ant-...

# --- Tuning (defaults shown) ---
MIN_PROFIT_PCT=0.02
MAX_POSITION_USD=100.0
MAX_TOTAL_EXPOSURE_USD=500.0
FEE_PCT=0.02
LOG_LEVEL=INFO
LEO_WEB_PORT=5002
```

All other parameters have sensible defaults (see `config.py`). The full list of environment variables is documented there.

### Run

```bash
python main.py
```

The terminal UI updates every second. The web dashboard is available at `http://localhost:5002`.

---

## Configuration reference

Key parameters and their defaults:

| Variable | Default | Description |
|---|---|---|
| `DRY_RUN` | `true` | Simulate trades without placing orders |
| `MAX_POSITION_USD` | `100.0` | Max USD per overround arb trade |
| `MAX_TOTAL_EXPOSURE_USD` | `500.0` | Max total open exposure |
| `MIN_PROFIT_PCT` | `0.02` | Min net profit for overround arb |
| `FEE_PCT` | `0.02` | Fee assumption for all strategies |
| `CRYPTO_SIGNAL_MIN_EDGE` | `0.05` | Min edge for crypto price signal |
| `CRYPTO_SIGNAL_MAX_USD` | `75.0` | Max size for crypto signal trades |
| `RANGE_STRADDLE_MIN_EDGE` | `0.05` | Min edge for range straddle |
| `CORR_MIN_EDGE` | `0.04` | Min edge for correlated arb |
| `CORR_MIN_CONFIDENCE` | `0.80` | Min confidence to trade correlated pair |
| `NEWS_FADE_MIN_SPIKE` | `0.12` | Min price move to register a spike (12%) |
| `NEWS_FADE_MIN_HOURS` | `1.5` | Min hours after spike to fade |
| `NEWS_FADE_MAX_HOURS` | `4.0` | Max hours after spike to still fade |
| `CROSS_ARB_MIN_PROFIT_PCT` | `0.06` | Min edge for Polymarket signal |
| `FORECAST_MIN_EDGE` | `0.05` | Min edge for forecast aggregator |
| `LLM_MIN_EDGE` | `0.07` | Min edge for LLM signal (higher: LLM is uncertain) |
| `LLM_MAX_MARKETS_PER_SCAN` | `20` | Max markets sent to Claude per cycle |
| `WEATHER_MIN_EDGE` | `0.05` | Min edge for weather signal |
| `SIGNAL_KELLY_FRACTION` | `0.10` | Fraction of Kelly criterion to use |

---

## Data & persistence

All trades and positions are logged to `data/leo.db` (SQLite). Logs are written to `data/leo.log` and stderr simultaneously.

The web dashboard at `http://localhost:5002` shows live opportunities, open positions, and trade history without requiring terminal access.

---

## Risk notes

- **`DRY_RUN=true` by default.** Leo will not place real orders until you explicitly set `DRY_RUN=false`.
- The Kelly fraction defaults to 10% (`SIGNAL_KELLY_FRACTION=0.10`), which is conservative. Increase only if you have confidence in the model calibration.
- The LLM strategy is **disabled by default** (`LLM_ENABLED=false`). Claude Haiku API calls incur cost and the strategy has higher uncertainty than rule-based approaches — set a higher `LLM_MIN_EDGE` when enabling.
- Weather, forecast, and cross-platform data sources are all free and require no API keys.
- Kalshi imposes rate limits. The default `KALSHI_RATE_LIMIT=10` (requests per 10s) respects their documented limits.
