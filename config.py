import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CoinbaseConfig:
    api_key: str = field(default_factory=lambda: os.getenv("COINBASE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("COINBASE_API_SECRET", ""))
    api_passphrase: str = field(default_factory=lambda: os.getenv("COINBASE_API_PASSPHRASE", ""))
    predictions_base_url: str = "https://api.coinbase.com/api/v1/predictions"
    advanced_trade_url: str = "https://api.coinbase.com/api/v3/brokerage"
    ws_url: str = "wss://advanced-trade-ws.coinbase.com"
    rate_limit_per_second: int = 10


@dataclass
class ArbitrageConfig:
    # Minimum profit threshold after fees to trigger a trade (e.g. 0.02 = 2%)
    min_profit_pct: float = float(os.getenv("MIN_PROFIT_PCT", "0.02"))
    # Max position size in USD per trade
    max_position_usd: float = float(os.getenv("MAX_POSITION_USD", "100.0"))
    # Max total exposure across all open positions
    max_total_exposure_usd: float = float(os.getenv("MAX_TOTAL_EXPOSURE_USD", "500.0"))
    # Fee per side (Coinbase Predictions fee, adjust as needed)
    fee_pct: float = float(os.getenv("FEE_PCT", "0.02"))
    # Minimum market liquidity (USD) to consider a market
    min_liquidity_usd: float = float(os.getenv("MIN_LIQUIDITY_USD", "1000.0"))
    # Minimum time to resolution (hours) — avoid near-expiry markets
    min_hours_to_resolve: float = float(os.getenv("MIN_HOURS_TO_RESOLVE", "1.0"))
    # Maximum time to resolution (hours) — avoid very long-dated markets
    max_hours_to_resolve: float = float(os.getenv("MAX_HOURS_TO_RESOLVE", "720.0"))
    # Over-round threshold — if YES+NO prices sum < this, flag as arbitrage
    overround_threshold: float = float(os.getenv("OVERROUND_THRESHOLD", "0.98"))
    # Poll interval in seconds
    poll_interval_sec: int = int(os.getenv("POLL_INTERVAL_SEC", "5"))


@dataclass
class StorageConfig:
    db_path: str = os.getenv("DB_PATH", "data/leo.db")


@dataclass
class Config:
    coinbase: CoinbaseConfig = field(default_factory=CoinbaseConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
