"""
Shared bot state — imported by main.py, loops.py, terminal_ui.py, and web_gui.py
so they always reference the same objects regardless of entry point.
"""

from collections import deque

from api_clients.polymarket_client import Market
from arbitrage import ArbOpportunity
from strategies.correlated import CorrelatedOpportunity
from strategies.cross_platform import CrossPlatformOpportunity
from strategies.signal_arb import AggregatedSignal


class BotState:
    def __init__(self):
        self.markets: list[Market] = []
        self.market_map: dict[str, Market] = {}
        self.extended_market_map: dict[str, Market] = {}

        self.arb_opps: list[ArbOpportunity] = []
        self.crypto_opps: list[AggregatedSignal] = []
        self.cross_opps: list[CrossPlatformOpportunity] = []
        self.range_opps: list[AggregatedSignal] = []
        self.corr_opps: list[CorrelatedOpportunity] = []
        self.fade_opps: list[AggregatedSignal] = []
        self.forecast_opps: list[AggregatedSignal] = []
        self.llm_opps: list[AggregatedSignal] = []
        self.weather_opps: list[AggregatedSignal] = []

        self.fav_opps: list[AggregatedSignal] = []
        self.squeeze_opps: list[AggregatedSignal] = []
        self.semarg_opps: list[AggregatedSignal] = []
        self.ofi_opps: list[AggregatedSignal] = []
        self.btc5min_opps: list[AggregatedSignal] = []
        self.mm_quotes: list[dict] = []

        self.arb_scans = 0
        self.crypto_scans = 0
        self.cross_scans = 0
        self.range_scans = 0
        self.corr_scans = 0
        self.fade_scans = 0
        self.forecast_scans = 0
        self.llm_scans = 0
        self.weather_scans = 0
        self.fav_scans = 0
        self.squeeze_scans = 0
        self.semarg_scans = 0
        self.ofi_scans = 0
        self.btc5min_scans = 0
        self.mm_active = 0

        self.signal_opps: list[AggregatedSignal] = []
        self.signal_scans = 0
        self.mm_scans = 0
        self.whale_signals = 0

        self.last_signal_at: dict[str, str] = {}


state = BotState()
config = None
_pos_manager_ref = None
_storage_ref = None
_resume_event = None
_stop_event = None
_kyle_lambda_ref = None
_evolution_ref = None
_confluence_ref = None
_hurst_ref = None
_trader_ref = None
_client_ref = None
_force_market_refresh = False
_log_buffer: deque = deque(maxlen=200)
_btc5min_signals: dict = {}
_last_gamma_at: str = ""
_alerter_ref = None
paused = False
health_block_trading: bool = False
health_block_reason: str = ""
last_health: dict = {}
strategy_audit: list = []