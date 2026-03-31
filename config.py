import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class KalshiConfig:
    """
    Credentials for the Kalshi API — the backend powering Coinbase
    Predictions (launched Jan 28, 2026, all 50 US states).

    Get your API key pair at: https://app.kalshi.com/profile/api-keys
      - key_id:          shown in the dashboard
      - private_key_pem: RSA PEM shown only once at key creation

    Auth: RSA-PSS signatures (NOT HMAC, NOT JWT/EC)
    Docs: https://docs.kalshi.com
    """
    key_id: str = field(
        default_factory=lambda: os.getenv("KALSHI_KEY_ID", "")
    )
    private_key_pem: str = field(
        default_factory=lambda: (
            os.getenv("KALSHI_PRIVATE_KEY", "").replace("\\n", "\n")
        )
    )
    use_demo: bool = (
        os.getenv("KALSHI_USE_DEMO", "false").lower() == "true"
    )
    rate_limit_per_10s: int = int(os.getenv("KALSHI_RATE_LIMIT", "10"))


@dataclass
class ArbitrageConfig:
    # Minimum profit % after fees to trigger a trade (e.g. 0.02 = 2%)
    min_profit_pct: float = float(os.getenv("MIN_PROFIT_PCT", "0.02"))
    # Max position size in USD per trade
    max_position_usd: float = float(
        os.getenv("MAX_POSITION_USD", "100.0")
    )
    # Max total exposure across all open positions
    max_total_exposure_usd: float = float(
        os.getenv("MAX_TOTAL_EXPOSURE_USD", "500.0")
    )
    # Kalshi fee: $0.07 × P × (1-P) per contract (~$0.02 at $0.50)
    fee_pct: float = float(os.getenv("FEE_PCT", "0.02"))
    # Minimum market liquidity (USD) to consider
    min_liquidity_usd: float = float(
        os.getenv("MIN_LIQUIDITY_USD", "500.0")
    )
    # Minimum hours to resolution — avoid near-expiry markets
    min_hours_to_resolve: float = float(
        os.getenv("MIN_HOURS_TO_RESOLVE", "1.0")
    )
    # Maximum hours to resolution — avoid very long-dated markets
    max_hours_to_resolve: float = float(
        os.getenv("MAX_HOURS_TO_RESOLVE", "720.0")
    )
    # yes_bid + no_bid > this threshold → overround arb exists
    overround_threshold: float = float(
        os.getenv("OVERROUND_THRESHOLD", "1.02")
    )
    # Seconds between market scans
    poll_interval_sec: int = int(os.getenv("POLL_INTERVAL_SEC", "5"))


@dataclass
class StorageConfig:
    db_path: str = os.getenv("DB_PATH", "data/leo.db")


@dataclass
class Config:
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
