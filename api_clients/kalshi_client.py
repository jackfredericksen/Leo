"""
Kalshi API client — the backend powering Coinbase Predictions.

Coinbase Predictions launched January 28, 2026 across all 50 US states.
It is fully powered by Kalshi, a CFTC-regulated prediction market exchange.
There is no separate "Coinbase Predictions API" — all programmatic access
goes through Kalshi's REST and WebSocket APIs directly.

Authentication: RSA-PSS digital signature
  - Key ID + RSA private key from https://app.kalshi.com/profile/api-keys
  - Every request signed with: SHA-256(timestamp + METHOD + path)
  - Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE

Docs: https://docs.kalshi.com
"""

import asyncio
import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config import KalshiConfig

logger = logging.getLogger(__name__)


@dataclass
class Market:
    market_id: str          # Kalshi ticker, e.g. "INXD-24DEC31-B4800"
    event_ticker: str
    question: str
    yes_bid: float          # Best YES bid price (0–1 dollars)
    yes_ask: float          # Best YES ask price (0–1 dollars)
    no_bid: float           # Best NO bid price (0–1 dollars)
    no_ask: float           # Best NO ask price (0–1 dollars)
    yes_price: float        # Mid-price (calculated: (yes_bid + yes_ask) / 2)
    no_price: float         # Mid-price (calculated: (no_bid + no_ask) / 2)
    volume: float           # Total volume (contracts)
    volume_24h: float       # 24h volume (contracts)
    liquidity_usd: float    # Open liquidity in dollars
    open_interest: float    # Open interest (contracts)
    last_price: float       # Last traded price (dollars)
    close_time: datetime    # Market resolution time
    status: str             # "open" | "closed" | "settled"
    result: Optional[str] = None  # "yes" | "no" | None


@dataclass
class Orderbook:
    """
    Kalshi orderbooks return bids only. Asks are derived:
      YES ask at price X implies NO bid at (1.00 - X), and vice versa.

    yes_bids: list of [price_str, size_str] sorted best→worst
    no_bids:  list of [price_str, size_str] sorted best→worst
    """
    market_id: str
    yes_bids: list[list[str]]
    no_bids: list[list[str]]

    @property
    def best_yes_bid(self) -> float:
        return float(self.yes_bids[0][0]) if self.yes_bids else 0.0

    @property
    def best_no_bid(self) -> float:
        return float(self.no_bids[0][0]) if self.no_bids else 0.0

    @property
    def best_yes_ask(self) -> float:
        """YES ask = 1.00 - best NO bid (binary market identity)."""
        return round(1.0 - self.best_no_bid, 4) if self.no_bids else 1.0

    @property
    def best_no_ask(self) -> float:
        """NO ask = 1.00 - best YES bid (binary market identity)."""
        return round(1.0 - self.best_yes_bid, 4) if self.yes_bids else 1.0


@dataclass
class OrderResult:
    order_id: str
    market_id: str
    side: str           # "yes" | "no"
    action: str         # "buy" | "sell"
    size: int           # contracts
    price: int          # cents (1–99)
    status: str
    filled_count: int = 0


class KalshiClient:
    """
    Async Kalshi REST API client.

    Credentials: API key pair from https://app.kalshi.com/profile/api-keys
      - key_id: displayed in dashboard
      - private_key: RSA PEM (shown once at creation — save it!)
    """

    REST_BASE = "https://api.kalshi.com/trade-api/v2"
    WS_BASE = "wss://api.kalshi.com/trade-api/ws/v2"
    DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"

    def __init__(self, cfg: KalshiConfig):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None
        self._private_key = self._load_private_key(cfg.private_key_pem)
        self._base = self.DEMO_BASE if cfg.use_demo else self.REST_BASE

    def _load_private_key(self, pem: str):
        """Load RSA private key from PEM string."""
        pem_bytes = pem.replace("\\n", "\n").encode()
        return serialization.load_pem_private_key(pem_bytes, password=None)

    def _sign(self, method: str, path: str) -> dict:
        """
        Generate Kalshi auth headers.

        Signature = RSA-PSS(SHA-256, private_key, f"{timestamp_ms}{METHOD}{path}")
        Path must NOT include query parameters.
        Timestamp is in milliseconds.
        """
        ts_ms = str(int(time.time() * 1000))
        message = f"{ts_ms}{method.upper()}{path}".encode()

        signature_bytes = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        signature = base64.b64encode(signature_bytes).decode()

        return {
            "KALSHI-ACCESS-KEY": self.cfg.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
        }

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    async def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self._base}{path}"
        headers = self._sign("GET", path)
        async with self._session.get(url, headers=headers, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _post(self, path: str, body: dict) -> dict:
        url = f"{self._base}{path}"
        body_str = json.dumps(body)
        headers = self._sign("POST", path)
        async with self._session.post(url, headers=headers, data=body_str) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _delete(self, path: str, body: dict = None) -> dict:
        url = f"{self._base}{path}"
        body_str = json.dumps(body or {})
        headers = self._sign("DELETE", path)
        async with self._session.delete(url, headers=headers, data=body_str) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------ #
    #  Account                                                             #
    # ------------------------------------------------------------------ #

    async def get_balance(self) -> float:
        """Return account balance in dollars."""
        data = await self._get("/portfolio/balance")
        return data.get("balance", 0) / 100  # API returns cents

    # ------------------------------------------------------------------ #
    #  Markets                                                             #
    # ------------------------------------------------------------------ #

    async def get_markets(
        self,
        status: str = "open",
        limit: int = 200,
        cursor: str = None,
    ) -> tuple[list[Market], str | None]:
        """
        Paginated market listing. Returns (markets, next_cursor).
        Pass cursor to get the next page. None cursor = last page.
        """
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = await self._get("/markets", params=params)
        markets = [self._parse_market(m) for m in data.get("markets", [])]
        return markets, data.get("cursor")

    async def get_all_markets(self, status: str = "open") -> list[Market]:
        """Fetch all markets across all pages."""
        markets, cursor = await self.get_markets(status=status)
        while cursor:
            more, cursor = await self.get_markets(status=status, cursor=cursor)
            markets.extend(more)
            await asyncio.sleep(0.1)  # be polite to the rate limiter
        return markets

    async def get_market(self, ticker: str) -> Market:
        data = await self._get(f"/markets/{ticker}")
        return self._parse_market(data["market"])

    async def get_orderbook(self, ticker: str, depth: int = 10) -> Orderbook:
        data = await self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})
        ob = data.get("orderbook", data)
        return Orderbook(
            market_id=ticker,
            yes_bids=ob.get("yes", []),
            no_bids=ob.get("no", []),
        )

    async def get_events(self, status: str = "open") -> list[dict]:
        data = await self._get("/events", params={"status": status})
        return data.get("events", [])

    # ------------------------------------------------------------------ #
    #  Orders                                                              #
    # ------------------------------------------------------------------ #

    async def place_order(
        self,
        ticker: str,
        side: str,          # "yes" | "no"
        action: str,        # "buy" | "sell"
        count: int,         # number of contracts
        limit_price: int,   # cents (1–99), e.g. 42 = $0.42
        order_type: str = "limit",
    ) -> OrderResult:
        """
        Place an order.

        Kalshi contracts are priced in cents (1–99).
        Each contract pays out $1.00 if it resolves in your favour.
        Minimum order: 1 contract.

        Fee per filled contract: $0.07 × P × (1 - P)
          where P = price in dollars (e.g. 0.42)
          e.g. at $0.50: fee = $0.07 × 0.50 × 0.50 = $0.0175 ≈ $0.02/contract
        """
        body = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": order_type,
            "buy_max_cost": count * 100 if action == "buy" else None,
        }
        if order_type == "limit":
            body["yes_price"] = limit_price if side == "yes" else (100 - limit_price)
        body = {k: v for k, v in body.items() if v is not None}

        data = await self._post("/portfolio/orders", body)
        order = data["order"]
        return OrderResult(
            order_id=order["order_id"],
            market_id=ticker,
            side=side,
            action=action,
            size=count,
            price=limit_price,
            status=order["status"],
            filled_count=order.get("filled_count", 0),
        )

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await self._delete(f"/portfolio/orders/{order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel order {order_id} failed: {e}")
            return False

    async def batch_cancel_orders(self, order_ids: list[str]) -> dict:
        """Cancel up to 20 orders in one call."""
        return await self._delete(
            "/portfolio/orders/batched",
            {"ids": order_ids[:20]},
        )

    async def get_orders(self, ticker: str = None, status: str = None) -> list[dict]:
        params = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        data = await self._get("/portfolio/orders", params=params)
        return data.get("orders", [])

    async def get_positions(self, ticker: str = None) -> list[dict]:
        params = {"ticker": ticker} if ticker else {}
        data = await self._get("/portfolio/positions", params=params)
        return data.get("market_positions", [])

    # ------------------------------------------------------------------ #
    #  Parsing                                                             #
    # ------------------------------------------------------------------ #

    def _parse_market(self, m: dict) -> Market:
        # Prices come back as decimal strings in dollars (e.g. "0.42")
        yes_bid = float(m.get("yes_bid", 0) or 0)
        yes_ask_raw = m.get("yes_ask")
        no_bid = float(m.get("no_bid", 0) or 0)
        no_ask_raw = m.get("no_ask")

        # Derive asks from the binary identity if not present
        yes_ask = float(yes_ask_raw) if yes_ask_raw else round(1.0 - no_bid, 4)
        no_ask = float(no_ask_raw) if no_ask_raw else round(1.0 - yes_bid, 4)

        yes_mid = (yes_bid + yes_ask) / 2
        no_mid = (no_bid + no_ask) / 2

        close_raw = m.get("close_time") or m.get("latest_expiration_time", "")
        close_time = datetime.fromtimestamp(close_raw, tz=timezone.utc) if isinstance(close_raw, (int, float)) else datetime.fromisoformat(str(close_raw).replace("Z", "+00:00"))

        return Market(
            market_id=m["ticker"],
            event_ticker=m.get("event_ticker", ""),
            question=m.get("title") or m.get("yes_sub_title") or m.get("ticker", ""),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            yes_price=round(yes_mid, 4),
            no_price=round(no_mid, 4),
            volume=float(m.get("volume", 0) or 0),
            volume_24h=float(m.get("volume_24h", 0) or 0),
            liquidity_usd=float(m.get("liquidity", 0) or 0),
            open_interest=float(m.get("open_interest", 0) or 0),
            last_price=float(m.get("last_price", yes_mid) or yes_mid),
            close_time=close_time,
            status=m.get("status", "open"),
            result=m.get("result"),
        )

    @staticmethod
    def fee_per_contract(price_dollars: float) -> float:
        """
        Kalshi taker fee: $0.07 × P × (1 - P) per contract, rounded up to nearest cent.
        At $0.50: $0.0175 ≈ $0.02/contract
        At $0.10: $0.0063 ≈ $0.01/contract
        """
        import math
        raw = 0.07 * price_dollars * (1 - price_dollars)
        return math.ceil(raw * 100) / 100  # round up to nearest cent

    @staticmethod
    def contracts_for_usd(usd: float, price_dollars: float) -> int:
        """How many contracts can we buy for `usd` at `price_dollars` each?"""
        if price_dollars <= 0:
            return 0
        return max(0, int(usd / price_dollars))
