# 🍉Support Humanitarian Efforts in Palestine🍉

The ongoing humanitarian crisis in Palestine has left millions in urgent need of aid. If you're looking to make a difference, consider supporting trusted organizations working on the ground to provide food, medical care, and essential relief:
- [UN Crisis Relief – Occupied Palestinian Territory Humanitarian Fund](https://crisisrelief.un.org/en/opt-crisis)
- [Palestine Children's Relief Fund ](https://www.pcrf.net/)
- [Doctors Without Borders](https://www.doctorswithoutborders.org/)
- [Anera (American Near East Refugee Aid)](https://www.anera.org/)
- [Save the Children](https://www.savethechildren.org/us/where-we-work/west-bank-gaza)
<br></br>


# Leo — Polymarket Trading Bot

Leo is an automated trading bot for [Polymarket](https://polymarket.com) prediction markets. It runs 14 independent strategies in parallel, finding edges through overround pricing, crypto price models, correlated market logic, community forecasts, weather data, order book imbalance, and LLM analysis. All order execution goes through Polymarket's CLOB API v2; external platforms (Kalshi, Metaculus, Manifold, Binance) are used as price signals only.

---

## Strategies

### 1. Overround Arbitrage
Polymarket binary markets have YES and NO tokens. When `YES_ask + NO_ask < $1.00`, buying both simultaneously locks in a guaranteed profit regardless of outcome. Leo scans all open markets every few seconds, detects these conditions, and executes both legs.

### 2. Crypto Price Signal
BTC, ETH, and SOL price markets (e.g. "Will BTC close above $90,000 this week?") are priced against a log-normal model built from live Binance spot prices and realized volatility from 1-minute candles. When the market price diverges materially from the model, Leo trades the cheap side.

### 3. Range Straddle
Polymarket lists weekly crypto range contracts. Leo uses a Black-Scholes-style model — live spot + realized vol — to estimate each bracket's true probability and trades when the market price diverges.

### 4. Correlated / Logical Arbitrage
Some markets have hard logical relationships: "BTC above $80k in January" can't rationally trade lower than "BTC above $80k in February." Leo auto-discovers monotone relationships within market groups and trades legs that violate the implied ordering.

### 5. News Fade
When a market spikes ≥12% sharply in response to a news event, Leo records the spike and fades it 1.5–4 hours later — betting that overreaction hasn't fully corrected. Detection is cross-cycle: the bot compares the current price to the price from the previous scan.

### 6. Forecast Aggregator
Leo fetches the 200 most active open questions from Metaculus and Manifold and matches them to Polymarket markets via Jaccard word-set similarity. When the community consensus diverges from the Polymarket price by more than the minimum edge, Leo trades.

### 7. LLM Fundamental Analysis

For qualitative markets — elections, policy decisions, geopolitical events — Leo asks Claude Haiku to estimate a probability. Questions are enriched with any matching Metaculus/Manifold context first. The model returns `{probability, confidence, reasoning}`; position size is scaled by stated confidence.

### 8. Weather Signal

Polymarket weather markets (e.g. "Will NYC high temperature exceed 75°F on April 15?") are priced against 16-day Open-Meteo forecasts (free, no API key required):

- **Temperature**: P(actual > threshold) using a normal distribution where σ grows with days out (±2°F same-day → ±8°F at two weeks)
- **Precipitation**: P(precip > threshold) using P(any rain) × exponential distribution for amount

### 9. Cross-Platform Signal (Kalshi)

Leo fetches public Kalshi market prices and fuzzy-matches them to Polymarket markets on the same question. When the implied probabilities diverge materially, Leo trades on Polymarket using Kalshi as the fair-value anchor. Kalshi is a signal source only — no orders are placed there.

### 10. Favorite Short

The favorite-longshot bias causes heavy favorites (>88¢) in thin, low-volume markets to be systematically overpriced. Leo applies a ~6% fair-value discount and shorts the overpriced YES side by buying NO. Position size is conservative (8% Kelly).

### 11. Oracle Squeeze

Polymarket markets continue trading on the CLOB after their real-world event ends while the UMA Optimistic Oracle processes resolution. During this window — which can last hours to days — prices that have converged near 0 or 1 still have a gap to capture. Leo buys the near-certain side and holds to resolution.

### 12. Semantic Arbitrage

Finds semantically identical markets listed simultaneously on Polymarket and Kalshi, then trades the persistent price gap. Matching requires: (1) same numeric thresholds within 1% tolerance, (2) keyword Jaccard overlap ≥ 65%, and (3) direction agreement ("above"/"below"). Only true semantic duplicates are traded — not lookalikes.

### 13. OFI Momentum

Computes Order Flow Imbalance (OFI) from live CLOB order book depth:

```
OFI = (bid_volume_L5 − ask_volume_L5) / total_volume_L5
```

**High-activity regime** (volume/liquidity > 2.0): follow OFI direction (momentum).  
**Low-activity regime** (volume/liquidity < 0.3): fade OFI extremes (mean reversion).  
Requires one async `get_orderbook()` call per candidate market per scan.

### 14. Market Maker

Posts resting limit buy orders on both YES and NO sides of selected markets. Earns the bid-ask spread passively when takers fill against resting orders. Polymarket CLOB v2 redistributes 100% of taker fees to makers daily — orders must stay on-book ≥ 3.5 seconds to qualify. Targets markets with real CLOB spread > 6¢ (checked via live orderbook, not the Gamma API mid-price).

---

## Architecture

```
main.py              — Orchestrator: 17 asyncio tasks + Rich terminal UI
config.py            — All settings as dataclasses, loaded from env vars
trader.py            — Order execution: execute(), execute_signal(), execute_correlated()
arbitrage.py         — Overround arb scanner
auto_correlator.py   — Discovers logical relationships within market groups
position_manager.py  — Live positions + P&L (Polymarket Data API)
storage.py           — SQLite persistence (data/leo.db): trades + history
bot_state.py         — Shared state bridge between main loop and web dashboard
web_gui.py           — FastAPI + WebSocket real-time dashboard (port 5002)

api_clients/
  polymarket_client.py — Gamma API (markets) + CLOB v2 (orders) + Data API (positions)
  binance_client.py    — Spot prices + 1-min OHLCV + realized volatility
  forecast_client.py   — Metaculus + Manifold (community forecasts)
  llm_client.py        — Anthropic Claude (async, semaphore-limited, TTL cache)
  weather_client.py    — Open-Meteo 16-day forecast (30-min TTL cache)

strategies/
  signal_arb.py          — AggregatedSignal type + ask_edge() helper
  crypto_signal.py       — BTC/ETH/SOL log-normal price signal
  range_straddle.py      — Crypto range bracket pricing (Black-Scholes)
  correlated.py          — Logical/correlated arb execution
  news_fade.py           — News spike fade
  cross_platform.py      — Kalshi ↔ Polymarket cross-platform signal
  forecast_signal.py     — Metaculus/Manifold weighted consensus
  llm_signal.py          — Claude Haiku fundamental analysis
  weather_signal.py      — Open-Meteo temperature/precipitation signal
  favorite_short.py      — Favorite-longshot bias short
  oracle_squeeze.py      — Post-close oracle resolution squeeze
  semantic_arb.py        — Number-aware cross-platform semantic matching
  orderbook_momentum.py  — OFI momentum from live CLOB depth
  market_maker.py        — Stateful resting order management + rebate targeting
  kelly.py               — Fractional Kelly position sizing
```

### Execution flow

Each strategy loop runs independently via `asyncio.create_task`. All loops share a `state.markets` list refreshed every 30 seconds by a dedicated `market_refresh_loop`. When a loop finds an opportunity it calls one of three paths on `Trader`:

- `trader.execute(opp)` — two-leg overround arb (buy YES + NO simultaneously)
- `trader.execute_signal(sig, market_map)` — single-leg signal trade
- `trader.execute_correlated(opp, market_map)` — correlated arb single leg

All execution paths respect a **30-minute per-market cooldown** to prevent multiple strategies piling into the same market. Position size is determined by the **fractional Kelly criterion** (configurable fraction, default 10%).

---

## Setup

### Requirements

- Python 3.11+
- Polymarket account with a funded Polygon wallet

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p data
```

### Polymarket credentials (one-time)

Polymarket uses EIP-712 signed orders. You need your Ethereum private key plus API credentials derived from it:

```bash
# 1. Set POLY_PRIVATE_KEY in your .env, then:
python -c "from api_clients.polymarket_client import print_api_creds; print_api_creds()"

# 2. Copy the printed values into .env:
#    POLY_API_KEY=...
#    POLY_API_SECRET=...
#    POLY_PASSPHRASE=...
```

### Environment variables

Create a `.env` file in the project root:

```env
# --- Required for live trading ---
POLY_PRIVATE_KEY=0x...            # Ethereum hex private key
POLY_WALLET_ADDRESS=0x...         # Your Polygon wallet address
POLY_API_KEY=...                  # Derived via print_api_creds()
POLY_API_SECRET=...
POLY_PASSPHRASE=...

# --- Mode ---
DRY_RUN=true                      # Set false to place real orders

# --- Optional: LLM strategy ---
LLM_ENABLED=false                 # Set true to enable Claude Haiku
ANTHROPIC_API_KEY=sk-ant-...

# --- Optional: News fade ---
NEWS_API_KEY=...                  # newsapi.org key (free tier works)

# --- Tuning (defaults shown) ---
MIN_PROFIT_PCT=0.02
MAX_POSITION_USD=100.0
MAX_TOTAL_EXPOSURE_USD=500.0
FEE_PCT=0.02
LOG_LEVEL=INFO
LEO_WEB_PORT=5002
```

All other parameters have sensible defaults — see `config.py` for the full list.

### Run

```bash
python main.py
```

The terminal UI updates every second. The web dashboard is at `http://localhost:5002`.

---

## Web dashboard

The dashboard has 6 tabs:

| Tab | Contents |
|---|---|
| **Overview** | Balance, P&L cards, scan counters, opportunity grids for all 14 strategies |
| **Signals** | Crypto, News Fade, Forecast, LLM, Weather, Range, Cross-Platform tables |
| **Arb + MM** | Market Maker active quotes, Overround Arb table, Correlated Arb table |
| **Portfolio** | Open positions with unrealized P&L, hours to resolution |
| **Trades** | Last 50 logged trades |
| **Controls** | Dry Run toggle, Pause/Resume, Stop (with confirmation), per-strategy enable/disable |

The dashboard connects via WebSocket and reconnects automatically with exponential backoff.

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `DRY_RUN` | `true` | Simulate trades without placing orders |
| `MAX_POSITION_USD` | `100.0` | Max USD per overround arb trade |
| `MAX_TOTAL_EXPOSURE_USD` | `500.0` | Max total open exposure |
| `MIN_PROFIT_PCT` | `0.02` | Min net profit % for overround arb |
| `FEE_PCT` | `0.02` | Fee assumption used across strategies |
| `CRYPTO_SIGNAL_MIN_EDGE` | `0.05` | Min edge for crypto price signal |
| `CRYPTO_SIGNAL_MAX_USD` | `75.0` | Max size for crypto signal trades |
| `RANGE_STRADDLE_MIN_EDGE` | `0.05` | Min edge for range straddle |
| `CORR_MIN_EDGE` | `0.04` | Min edge for correlated arb |
| `CORR_MIN_CONFIDENCE` | `0.80` | Min confidence to trade a correlated pair |
| `NEWS_FADE_MIN_SPIKE` | `0.12` | Min price move to register a spike |
| `NEWS_FADE_MIN_HOURS` | `1.5` | Earliest to fade after a spike |
| `NEWS_FADE_MAX_HOURS` | `4.0` | Latest to fade after a spike |
| `CROSS_ARB_MIN_PROFIT_PCT` | `0.06` | Min edge for Kalshi signal |
| `FORECAST_MIN_EDGE` | `0.05` | Min edge for forecast aggregator |
| `LLM_MIN_EDGE` | `0.07` | Min edge for LLM signal |
| `LLM_MAX_MARKETS_PER_SCAN` | `20` | Max markets sent to Claude per cycle |
| `WEATHER_MIN_EDGE` | `0.05` | Min edge for weather signal |
| `FAV_SHORT_MIN_PRICE` | `0.88` | Min YES price to consider shorting |
| `FAV_SHORT_MAX_PRICE` | `0.97` | Max YES price (skip near-certain markets) |
| `FAV_SHORT_DISCOUNT` | `0.06` | Fair-value discount for favorite-longshot bias |
| `ORACLE_SQUEEZE_MIN_GAP` | `0.08` | Min distance from 0/1 to trade squeeze |
| `ORACLE_SQUEEZE_MAX_GAP` | `0.25` | Max distance — don't trade if not yet priced in |
| `SEMANTIC_ARB_MIN_GAP` | `0.04` | Min net price gap after fees for semantic arb |
| `SEMANTIC_ARB_MIN_JACCARD` | `0.65` | Keyword overlap threshold for matching |
| `OFI_THRESHOLD` | `0.45` | Minimum abs(OFI) to generate a signal |
| `OFI_HIGH_RATIO` | `2.0` | Activity ratio above which to follow OFI |
| `OFI_LOW_RATIO` | `0.3` | Activity ratio below which to fade OFI |
| `MM_MIN_SPREAD` | `0.06` | Min real CLOB spread to quote a market |
| `MM_MAX_MARKETS` | `8` | Max simultaneous markets to quote |
| `MM_HALF_SPREAD` | `0.03` | Quote offset from mid (liquid markets) |
| `MM_THIN_HALF_SPREAD` | `0.05` | Quote offset from mid (thin markets) |
| `SIGNAL_KELLY_FRACTION` | `0.10` | Kelly fraction for most signal strategies |

---

## Data & persistence

All trades are logged to `data/leo.db` (SQLite). Logs write to `data/leo.log` and stderr simultaneously. The database is created automatically on first run.

---

## Risk notes

- **`DRY_RUN=true` by default.** Leo will not place real orders until you explicitly set `DRY_RUN=false`.
- Kelly fractions default to 8–12% depending on strategy. These are conservative. Increase only after validating model calibration on dry-run history.
- The **LLM strategy is disabled by default** (`LLM_ENABLED=false`). Claude Haiku calls incur cost and have higher uncertainty than rule-based strategies — use a higher `LLM_MIN_EDGE` and low `LLM_MAX_MARKETS_PER_SCAN` when enabling.
- The **Market Maker strategy is enabled by default** but requires a funded wallet and valid CLOB credentials to place actual orders. In dry-run mode it logs intended quotes without submitting them.
- The **Oracle Squeeze** strategy trades markets past their `close_time`. UMA occasionally resolves N/A or disputes — `ORACLE_SQUEEZE_MIN_GAP=0.08` filters out ambiguous cases but does not eliminate this risk entirely.
- Polymarket imposes CLOB rate limits. The `get_orderbook()` calls in OFI and Market Maker are the highest-frequency API consumers; the default candidate limits (`OFI_MAX_CANDIDATES=20`, `MM_MAX_MARKETS=8`) keep request volume within normal bounds.
- Weather, forecast, and Open-Meteo data sources are all free and require no API keys.
