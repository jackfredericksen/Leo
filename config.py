import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PolymarketConfig:
    """
    Credentials for Polymarket CLOB trading.

    One-time setup:
      1. Set POLY_PRIVATE_KEY (Ethereum hex private key)
      2. Run: python -c "from api_clients.polymarket_client import print_api_creds; print_api_creds()"
      3. Copy POLY_API_KEY / POLY_API_SECRET / POLY_PASSPHRASE into .env

    Wallet address is optional — derived automatically from private key if not set.
    Polymarket runs on Polygon; USDC is the settlement currency.
    """
    wallet_address: str = field(
        default_factory=lambda: os.getenv("POLY_WALLET_ADDRESS", "")
    )
    private_key: str = field(
        default_factory=lambda: os.getenv("POLY_PRIVATE_KEY", "")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("POLY_API_KEY", "")
    )
    api_secret: str = field(
        default_factory=lambda: os.getenv("POLY_API_SECRET", "")
    )
    api_passphrase: str = field(
        default_factory=lambda: os.getenv("POLY_PASSPHRASE", "")
    )


@dataclass
class ArbitrageConfig:
    enabled: bool = os.getenv("ARB_ENABLED", "true").lower() == "true"
    min_profit_pct: float = float(os.getenv("MIN_PROFIT_PCT", "0.02"))
    max_position_usd: float = float(
        os.getenv("MAX_POSITION_USD", "100.0")
    )
    max_total_exposure_usd: float = float(
        os.getenv("MAX_TOTAL_EXPOSURE_USD", "500.0")
    )
    fee_pct: float = float(os.getenv("FEE_PCT", "0.02"))
    min_liquidity_usd: float = float(
        os.getenv("MIN_LIQUIDITY_USD", "500.0")
    )
    min_hours_to_resolve: float = float(
        os.getenv("MIN_HOURS_TO_RESOLVE", "1.0")
    )
    max_hours_to_resolve: float = float(
        os.getenv("MAX_HOURS_TO_RESOLVE", "720.0")
    )
    overround_threshold: float = float(
        os.getenv("OVERROUND_THRESHOLD", "1.02")
    )
    poll_interval_sec: int = int(os.getenv("POLL_INTERVAL_SEC", "5"))
    live_candidates: int = int(os.getenv("ARB_LIVE_CANDIDATES", "20"))


@dataclass
class CorrelatedArbConfig:
    enabled: bool = (
        os.getenv("CORR_ENABLED", "true").lower() == "true"
    )
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
    enabled: bool = (
        os.getenv("SIGNAL_ENABLED", "true").lower() == "true"
    )
    min_edge: float = float(os.getenv("SIGNAL_MIN_EDGE", "0.05"))
    uncertainty_buffer: float = float(
        os.getenv("SIGNAL_UNCERTAINTY", "0.05")
    )
    fee_pct: float = float(os.getenv("SIGNAL_FEE_PCT", "0.02"))
    max_position_usd: float = float(
        os.getenv("SIGNAL_MAX_POSITION_USD", "50.0")
    )
    kelly_fraction: float = float(
        os.getenv("SIGNAL_KELLY_FRACTION", "0.10")
    )
    poll_interval_sec: int = int(
        os.getenv("SIGNAL_POLL_INTERVAL_SEC", "60")
    )


@dataclass
class PositionConfig:
    refresh_interval_sec: int = int(
        os.getenv("POSITION_REFRESH_SEC", "30")
    )


@dataclass
class StorageConfig:
    db_path: str = os.getenv("DB_PATH", "data/leo.db")


@dataclass
class CryptoSignalConfig:
    enabled: bool = (
        os.getenv("CRYPTO_SIGNAL_ENABLED", "true").lower() == "true"
    )
    min_edge: float = float(os.getenv("CRYPTO_SIGNAL_MIN_EDGE", "0.05"))
    refresh_interval_sec: int = int(
        os.getenv("CRYPTO_SIGNAL_REFRESH_SEC", "30")
    )
    max_position_usd: float = float(
        os.getenv("CRYPTO_SIGNAL_MAX_USD", "75.0")
    )
    kelly_fraction: float = float(
        os.getenv("CRYPTO_SIGNAL_KELLY", "0.10")
    )


@dataclass
class RangeStraddleConfig:
    enabled: bool = (
        os.getenv("RANGE_STRADDLE_ENABLED", "true").lower() == "true"
    )
    min_edge: float = float(
        os.getenv("RANGE_STRADDLE_MIN_EDGE", "0.05")
    )
    max_position_usd: float = float(
        os.getenv("RANGE_STRADDLE_MAX_USD", "75.0")
    )
    kelly_fraction: float = float(
        os.getenv("RANGE_STRADDLE_KELLY", "0.10")
    )


@dataclass
class CrossPlatformArbConfig:
    """
    Kalshi-signal strategy: fetch Kalshi public prices as a signal,
    trade the mispricing on Polymarket.
    """
    enabled: bool = (
        os.getenv("CROSS_ARB_ENABLED", "true").lower() == "true"
    )
    min_profit_pct: float = float(
        os.getenv("CROSS_ARB_MIN_PROFIT_PCT", "0.06")
    )
    max_position_usd: float = float(
        os.getenv("CROSS_ARB_MAX_USD", "100.0")
    )
    refresh_interval_sec: int = int(
        os.getenv("CROSS_ARB_REFRESH_SEC", "60")
    )



@dataclass
class ForecastConfig:
    enabled: bool = (
        os.getenv("FORECAST_ENABLED", "true").lower() == "true"
    )
    min_edge: float = float(os.getenv("FORECAST_MIN_EDGE", "0.05"))
    max_position_usd: float = float(
        os.getenv("FORECAST_MAX_USD", "75.0")
    )
    kelly_fraction: float = float(
        os.getenv("FORECAST_KELLY", "0.10")
    )
    refresh_interval_sec: int = int(
        os.getenv("FORECAST_REFRESH_SEC", "900")
    )
    min_forecasters: int = int(
        os.getenv("FORECAST_MIN_FORECASTERS", "10")
    )
    min_similarity: float = float(
        os.getenv("FORECAST_MIN_SIMILARITY", "0.25")
    )


@dataclass
class LLMConfig:
    enabled: bool = (
        os.getenv("LLM_ENABLED", "false").lower() == "true"
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL", "claude-haiku-4-5-20251001"
        )
    )
    min_edge: float = float(os.getenv("LLM_MIN_EDGE", "0.07"))
    max_concurrent: int = int(os.getenv("LLM_MAX_CONCURRENT", "3"))
    cache_ttl_min: int = int(os.getenv("LLM_CACHE_TTL_MIN", "30"))
    max_markets_per_scan: int = int(
        os.getenv("LLM_MAX_MARKETS_PER_SCAN", "20")
    )
    min_liquidity_usd: float = float(
        os.getenv("LLM_MIN_LIQUIDITY_USD", "500.0")
    )
    max_position_usd: float = float(
        os.getenv("LLM_MAX_USD", "50.0")
    )
    kelly_fraction: float = float(
        os.getenv("LLM_KELLY", "0.10")
    )
    refresh_interval_sec: int = int(
        os.getenv("LLM_REFRESH_SEC", "120")
    )


@dataclass
class WeatherConfig:
    enabled: bool = (
        os.getenv("WEATHER_ENABLED", "true").lower() == "true"
    )
    min_edge: float = float(os.getenv("WEATHER_MIN_EDGE", "0.05"))
    max_position_usd: float = float(
        os.getenv("WEATHER_MAX_USD", "75.0")
    )
    kelly_fraction: float = float(
        os.getenv("WEATHER_KELLY", "0.10")
    )
    refresh_interval_sec: int = int(
        os.getenv("WEATHER_REFRESH_SEC", "1800")
    )


@dataclass
class NewsFadeConfig:
    enabled: bool = (
        os.getenv("NEWS_FADE_ENABLED", "true").lower() == "true"
    )
    min_spike_pct: float = float(
        os.getenv("NEWS_FADE_MIN_SPIKE", "0.12")
    )
    min_hours_old: float = float(
        os.getenv("NEWS_FADE_MIN_HOURS", "1.5")
    )
    max_hours_old: float = float(
        os.getenv("NEWS_FADE_MAX_HOURS", "4.0")
    )
    fade_fraction: float = float(
        os.getenv("NEWS_FADE_FRACTION", "0.50")
    )
    min_liquidity_usd: float = float(
        os.getenv("NEWS_FADE_MIN_LIQ", "5000.0")
    )
    min_edge: float = float(os.getenv("NEWS_FADE_MIN_EDGE", "0.05"))
    max_position_usd: float = float(
        os.getenv("NEWS_FADE_MAX_USD", "50.0")
    )
    kelly_fraction: float = float(
        os.getenv("NEWS_FADE_KELLY", "0.10")
    )
    news_api_key: str = field(
        default_factory=lambda: os.getenv("NEWS_API_KEY", "")
    )
    require_news_confirmation: bool = (
        os.getenv("NEWS_REQUIRE_CONFIRM", "false").lower() == "true"
    )


@dataclass
class FavoriteShortConfig:
    enabled: bool = os.getenv("FAV_SHORT_ENABLED", "true").lower() == "true"
    min_yes_price: float = float(os.getenv("FAV_SHORT_MIN_PRICE", "0.88"))
    max_yes_price: float = float(os.getenv("FAV_SHORT_MAX_PRICE", "0.97"))
    max_volume_usd: float = float(os.getenv("FAV_SHORT_MAX_VOL", "5000.0"))
    max_open_interest_usd: float = float(os.getenv("FAV_SHORT_MAX_OI", "3000.0"))
    min_days_to_resolve: float = float(os.getenv("FAV_SHORT_MIN_DAYS", "7.0"))
    discount_factor: float = float(os.getenv("FAV_SHORT_DISCOUNT", "0.06"))
    fee_pct: float = float(os.getenv("FAV_SHORT_FEE", "0.02"))
    min_edge: float = float(os.getenv("FAV_SHORT_MIN_EDGE", "0.03"))
    max_position_usd: float = float(os.getenv("FAV_SHORT_MAX_USD", "75.0"))
    kelly_fraction: float = float(os.getenv("FAV_SHORT_KELLY", "0.08"))
    refresh_interval_sec: int = int(os.getenv("FAV_SHORT_REFRESH_SEC", "300"))


@dataclass
class OracleSqueezeConfig:
    enabled: bool = os.getenv("ORACLE_SQUEEZE_ENABLED", "true").lower() == "true"
    min_gap: float = float(os.getenv("ORACLE_SQUEEZE_MIN_GAP", "0.08"))
    max_gap: float = float(os.getenv("ORACLE_SQUEEZE_MAX_GAP", "0.25"))
    min_hours_past_close: float = float(os.getenv("ORACLE_SQUEEZE_MIN_HOURS", "0.5"))
    max_hours_past_close: float = float(os.getenv("ORACLE_SQUEEZE_MAX_HOURS", "168.0"))
    fee_pct: float = float(os.getenv("ORACLE_SQUEEZE_FEE", "0.02"))
    min_edge: float = float(os.getenv("ORACLE_SQUEEZE_MIN_EDGE", "0.03"))
    max_position_usd: float = float(os.getenv("ORACLE_SQUEEZE_MAX_USD", "100.0"))
    kelly_fraction: float = float(os.getenv("ORACLE_SQUEEZE_KELLY", "0.12"))
    refresh_interval_sec: int = int(os.getenv("ORACLE_SQUEEZE_REFRESH_SEC", "60"))


@dataclass
class SemanticArbConfig:
    enabled: bool = os.getenv("SEMANTIC_ARB_ENABLED", "true").lower() == "true"
    min_price_gap: float = float(os.getenv("SEMANTIC_ARB_MIN_GAP", "0.04"))
    min_jaccard: float = float(os.getenv("SEMANTIC_ARB_MIN_JACCARD", "0.65"))
    number_tolerance: float = float(os.getenv("SEMANTIC_ARB_NUM_TOL", "0.01"))
    max_position_usd: float = float(os.getenv("SEMANTIC_ARB_MAX_USD", "100.0"))
    fee_pct: float = float(os.getenv("SEMANTIC_ARB_FEE", "0.02"))
    kelly_fraction: float = float(os.getenv("SEMANTIC_ARB_KELLY", "0.10"))
    refresh_interval_sec: int = int(os.getenv("SEMANTIC_ARB_REFRESH_SEC", "120"))


@dataclass
class OrderbookMomentumConfig:
    enabled: bool = os.getenv("OFI_ENABLED", "true").lower() == "true"
    ofi_threshold: float = float(os.getenv("OFI_THRESHOLD", "0.45"))
    orderbook_levels: int = int(os.getenv("OFI_LEVELS", "5"))
    high_activity_ratio: float = float(os.getenv("OFI_HIGH_RATIO", "2.0"))
    low_activity_ratio: float = float(os.getenv("OFI_LOW_RATIO", "0.3"))
    min_liquidity_usd: float = float(os.getenv("OFI_MIN_LIQ", "500.0"))
    max_spread_cents: float = float(os.getenv("OFI_MAX_SPREAD", "0.12"))
    max_candidate_markets: int = int(os.getenv("OFI_MAX_CANDIDATES", "20"))
    fee_pct: float = float(os.getenv("OFI_FEE", "0.02"))
    min_edge: float = float(os.getenv("OFI_MIN_EDGE", "0.04"))
    max_position_usd: float = float(os.getenv("OFI_MAX_USD", "75.0"))
    kelly_fraction: float = float(os.getenv("OFI_KELLY", "0.10"))
    refresh_interval_sec: int = int(os.getenv("OFI_REFRESH_SEC", "120"))


@dataclass
class MarketMakerConfig:
    enabled: bool = os.getenv("MM_ENABLED", "true").lower() == "true"
    quote_half_spread: float = float(os.getenv("MM_HALF_SPREAD", "0.03"))
    thin_quote_half_spread: float = float(os.getenv("MM_THIN_HALF_SPREAD", "0.05"))
    max_inventory_usd: float = float(os.getenv("MM_MAX_INVENTORY", "200.0"))
    min_volume_usd: float = float(os.getenv("MM_MIN_VOLUME", "5000.0"))
    thin_oi_usd: float = float(os.getenv("MM_THIN_OI", "500.0"))
    min_spread_cents: float = float(os.getenv("MM_MIN_SPREAD", "0.06"))
    max_markets: int = int(os.getenv("MM_MAX_MARKETS", "8"))
    requote_threshold: float = float(os.getenv("MM_REQUOTE_THRESHOLD", "0.025"))
    requote_interval_sec: int = int(os.getenv("MM_REQUOTE_INTERVAL", "30"))
    min_order_usd: float = float(os.getenv("MM_MIN_ORDER", "10.0"))
    min_onbook_sec: float = float(os.getenv("MM_MIN_ONBOOK_SEC", "3.5"))
    refresh_interval_sec: int = int(os.getenv("MM_REFRESH_SEC", "30"))


@dataclass
class ConfluenceConfig:
    enabled: bool = os.getenv("CONFLUENCE_ENABLED", "false").lower() == "true"
    min_strategies: int = int(os.getenv("CONFLUENCE_MIN_STRATEGIES", "1"))
    max_edge_bonus: float = float(os.getenv("CONFLUENCE_MAX_BONUS", "0.02"))


@dataclass
class ExitConfig:
    enabled: bool = os.getenv("EXIT_ENABLED", "true").lower() == "true"
    exit_threshold: float = float(os.getenv("EXIT_THRESHOLD", "0.90"))
    min_contracts: float = float(os.getenv("EXIT_MIN_CONTRACTS", "1.0"))
    min_bid_ratio: float = float(os.getenv("EXIT_MIN_BID_RATIO", "0.94"))
    refresh_interval_sec: int = int(os.getenv("EXIT_REFRESH_SEC", "60"))


@dataclass
class EvolutionConfig:
    enabled: bool = os.getenv("EVOLUTION_ENABLED", "true").lower() == "true"
    review_interval_hours: int = int(os.getenv("EVOLUTION_INTERVAL_HOURS", "168"))
    min_trades_for_review: int = int(os.getenv("EVOLUTION_MIN_TRADES", "10"))
    model: str = os.getenv("EVOLUTION_MODEL", "claude-haiku-4-5-20251001")
    max_tokens: int = int(os.getenv("EVOLUTION_MAX_TOKENS", "350"))


@dataclass
class KyleLambdaConfig:
    enabled: bool = os.getenv("KYLE_LAMBDA_ENABLED", "true").lower() == "true"
    max_markets: int = int(os.getenv("KYLE_LAMBDA_MAX_MARKETS", "30"))
    refresh_interval_sec: int = int(os.getenv("KYLE_LAMBDA_REFRESH_SEC", "600"))
    high_impact_threshold: float = float(os.getenv("KYLE_LAMBDA_THRESHOLD", "0.5"))


@dataclass
class BTC5MinConfig:
    enabled: bool = os.getenv("BTC5MIN_ENABLED", "true").lower() == "true"
    min_edge: float = float(os.getenv("BTC5MIN_MIN_EDGE", "0.04"))
    max_position_usd: float = float(os.getenv("BTC5MIN_MAX_USD", "50.0"))
    kelly_fraction: float = float(os.getenv("BTC5MIN_KELLY", "0.08"))
    min_mins_to_close: float = float(os.getenv("BTC5MIN_MIN_MINS", "1.5"))
    max_mins_to_close: float = float(os.getenv("BTC5MIN_MAX_MINS", "4.0"))
    fee_pct: float = float(os.getenv("BTC5MIN_FEE", "0.02"))
    obi_levels: int = int(os.getenv("BTC5MIN_OBI_LEVELS", "10"))
    obi_weight: float = float(os.getenv("BTC5MIN_OBI_WEIGHT", "0.50"))
    momentum_lookback: int = int(os.getenv("BTC5MIN_MOM_LOOKBACK", "3"))
    momentum_weight: float = float(os.getenv("BTC5MIN_MOM_WEIGHT", "0.30"))
    rsi_period: int = int(os.getenv("BTC5MIN_RSI_PERIOD", "14"))
    rsi_weight: float = float(os.getenv("BTC5MIN_RSI_WEIGHT", "0.20"))
    refresh_interval_sec: int = int(os.getenv("BTC5MIN_REFRESH_SEC", "20"))


@dataclass
class RiskConfig:
    max_daily_usd_deployed: float = float(os.getenv("MAX_DAILY_USD", "1000.0"))
    paper_bankroll: float = float(os.getenv("PAPER_BANKROLL", "1000.0"))


@dataclass
class AlertingConfig:
    discord_webhook_url: str = field(
        default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", "")
    )
    large_fill_threshold_usd: float = float(os.getenv("ALERT_LARGE_FILL_USD", "50.0"))
    big_edge_threshold_pct: float = float(os.getenv("ALERT_BIG_EDGE_PCT", "8.0"))
    daily_loss_alert_usd: float = float(os.getenv("ALERT_DAILY_LOSS_USD", "100.0"))
    circuit_breaker_errors: int = int(os.getenv("CIRCUIT_BREAKER_ERRORS", "5"))
    circuit_breaker_pause_sec: int = int(os.getenv("CIRCUIT_BREAKER_PAUSE_SEC", "120"))


@dataclass
class Config:
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    signal: SignalArbConfig = field(default_factory=SignalArbConfig)
    positions: PositionConfig = field(default_factory=PositionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    crypto_signal: CryptoSignalConfig = field(
        default_factory=CryptoSignalConfig
    )
    range_straddle: RangeStraddleConfig = field(
        default_factory=RangeStraddleConfig
    )
    cross_arb: CrossPlatformArbConfig = field(
        default_factory=CrossPlatformArbConfig
    )
    news_fade: NewsFadeConfig = field(default_factory=NewsFadeConfig)
    correlated: CorrelatedArbConfig = field(
        default_factory=CorrelatedArbConfig
    )
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    favorite_short: FavoriteShortConfig = field(default_factory=FavoriteShortConfig)
    oracle_squeeze: OracleSqueezeConfig = field(default_factory=OracleSqueezeConfig)
    semantic_arb: SemanticArbConfig = field(default_factory=SemanticArbConfig)
    orderbook_momentum: OrderbookMomentumConfig = field(default_factory=OrderbookMomentumConfig)
    market_maker: MarketMakerConfig = field(default_factory=MarketMakerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    confluence: ConfluenceConfig = field(default_factory=ConfluenceConfig)
    position_exit: ExitConfig = field(default_factory=ExitConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    kyle_lambda: KyleLambdaConfig = field(default_factory=KyleLambdaConfig)
    btc_5min: BTC5MinConfig = field(default_factory=BTC5MinConfig)
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
