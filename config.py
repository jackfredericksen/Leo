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
    # Seconds between market scans (poll mode)
    poll_interval_sec: int = int(os.getenv("POLL_INTERVAL_SEC", "5"))


@dataclass
class CorrelatedArbConfig:
    enabled: bool = os.getenv("CORR_ENABLED", "true").lower() == "true"
    min_edge: float = float(os.getenv("CORR_MIN_EDGE", "0.04"))
    fee_pct: float = float(os.getenv("CORR_FEE_PCT", "0.02"))
    max_position_usd: float = float(
        os.getenv("CORR_MAX_POSITION_USD", "75.0")
    )
    min_confidence: float = float(
        os.getenv("CORR_MIN_CONFIDENCE", "0.80")
    )
    poll_interval_sec: int = int(
        os.getenv("CORR_POLL_INTERVAL_SEC", "30")
    )
    min_liquidity_usd: float = float(
        os.getenv("CORR_MIN_LIQUIDITY_USD", "200.0")
    )


@dataclass
class SignalArbConfig:
    enabled: bool = os.getenv("SIGNAL_ENABLED", "true").lower() == "true"
    min_edge: float = float(os.getenv("SIGNAL_MIN_EDGE", "0.05"))
    uncertainty_buffer: float = float(os.getenv("SIGNAL_UNCERTAINTY", "0.05"))
    fee_pct: float = float(os.getenv("SIGNAL_FEE_PCT", "0.02"))
    max_position_usd: float = float(os.getenv("SIGNAL_MAX_POSITION_USD", "50.0"))
    kelly_fraction: float = float(os.getenv("SIGNAL_KELLY_FRACTION", "0.10"))
    poll_interval_sec: int = int(os.getenv("SIGNAL_POLL_INTERVAL_SEC", "60"))


@dataclass
class PolymarketConfig:
    enabled: bool = (
        os.getenv("POLYMARKET_ENABLED", "true").lower() == "true"
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("POLYMARKET_API_KEY", "")
    )
    min_profit_pct: float = float(
        os.getenv("POLYMARKET_MIN_PROFIT_PCT", "0.03")
    )
    signal_only: bool = (
        os.getenv("POLYMARKET_SIGNAL_ONLY", "true").lower() == "true"
    )
    refresh_interval_sec: int = int(
        os.getenv("POLYMARKET_REFRESH_SEC", "60")
    )


@dataclass
class PositionConfig:
    refresh_interval_sec: int = int(os.getenv("POSITION_REFRESH_SEC", "30"))


@dataclass
class StorageConfig:
    db_path: str = os.getenv("DB_PATH", "data/leo.db")


@dataclass
class CryptoSignalConfig:
    enabled: bool = os.getenv("CRYPTO_SIGNAL_ENABLED", "true").lower() == "true"
    min_edge: float = float(os.getenv("CRYPTO_SIGNAL_MIN_EDGE", "0.05"))
    refresh_interval_sec: int = int(os.getenv("CRYPTO_SIGNAL_REFRESH_SEC", "30"))
    max_position_usd: float = float(os.getenv("CRYPTO_SIGNAL_MAX_USD", "75.0"))
    kelly_fraction: float = float(os.getenv("CRYPTO_SIGNAL_KELLY", "0.10"))


@dataclass
class RangeStraddleConfig:
    enabled: bool = os.getenv("RANGE_STRADDLE_ENABLED", "true").lower() == "true"
    min_edge: float = float(os.getenv("RANGE_STRADDLE_MIN_EDGE", "0.05"))
    max_position_usd: float = float(os.getenv("RANGE_STRADDLE_MAX_USD", "75.0"))
    kelly_fraction: float = float(os.getenv("RANGE_STRADDLE_KELLY", "0.10"))


@dataclass
class CrossPlatformArbConfig:
    enabled: bool = os.getenv("CROSS_ARB_ENABLED", "true").lower() == "true"
    min_profit_pct: float = float(os.getenv("CROSS_ARB_MIN_PROFIT_PCT", "0.06"))
    max_position_usd: float = float(os.getenv("CROSS_ARB_MAX_USD", "100.0"))
    refresh_interval_sec: int = int(os.getenv("CROSS_ARB_REFRESH_SEC", "30"))
    signal_only: bool = os.getenv("CROSS_ARB_SIGNAL_ONLY", "true").lower() == "true"


@dataclass
class LogicalArbConfig:
    enabled: bool = os.getenv("LOGICAL_ARB_ENABLED", "true").lower() == "true"
    min_edge: float = float(os.getenv("LOGICAL_ARB_MIN_EDGE", "0.03"))
    max_position_usd: float = float(os.getenv("LOGICAL_ARB_MAX_USD", "100.0"))
    kelly_fraction: float = float(os.getenv("LOGICAL_ARB_KELLY", "0.15"))


@dataclass
class NewsFadeConfig:
    enabled: bool = os.getenv("NEWS_FADE_ENABLED", "true").lower() == "true"
    min_spike_pct: float = float(os.getenv("NEWS_FADE_MIN_SPIKE", "0.12"))
    min_hours_old: float = float(os.getenv("NEWS_FADE_MIN_HOURS", "1.5"))
    max_hours_old: float = float(os.getenv("NEWS_FADE_MAX_HOURS", "4.0"))
    fade_fraction: float = float(os.getenv("NEWS_FADE_FRACTION", "0.50"))
    min_liquidity_usd: float = float(os.getenv("NEWS_FADE_MIN_LIQ", "5000.0"))
    min_edge: float = float(os.getenv("NEWS_FADE_MIN_EDGE", "0.05"))
    max_position_usd: float = float(os.getenv("NEWS_FADE_MAX_USD", "50.0"))
    kelly_fraction: float = float(os.getenv("NEWS_FADE_KELLY", "0.10"))


@dataclass
class Config:
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    signal: SignalArbConfig = field(default_factory=SignalArbConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    positions: PositionConfig = field(default_factory=PositionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    crypto_signal: CryptoSignalConfig = field(default_factory=CryptoSignalConfig)
    range_straddle: RangeStraddleConfig = field(default_factory=RangeStraddleConfig)
    cross_arb: CrossPlatformArbConfig = field(default_factory=CrossPlatformArbConfig)
    logical_arb: LogicalArbConfig = field(default_factory=LogicalArbConfig)
    news_fade: NewsFadeConfig = field(default_factory=NewsFadeConfig)
    correlated: CorrelatedArbConfig = field(default_factory=CorrelatedArbConfig)
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
