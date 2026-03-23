"""
Coinbase Predictions API client.

Coinbase Predictions uses a binary outcome model (YES/NO contracts),
similar to Polymarket's CLOB. Each market has:
  - A question / resolution criteria
  - YES and NO tokens, each priced 0–1 (representing probability)
  - YES + NO should always sum to ~1.00 in an efficient market

Docs: https://docs.cdp.coinbase.com/predictions/docs/welcome
"""

import hashlib
import hmac
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiohttp

from config import CoinbaseConfig

logger = logging.getLogger(__name__)


@dataclass
class Market:
    market_id: str
    question: str
    yes_price: float       # 0–1
    no_price: float        # 0–1
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume_usd: float
    liquidity_usd: float
    resolves_at: datetime
    status: str            # "open" | "closed" | "resolved"
    outcome: Optional[str] = None   # "YES" | "NO" | None


@dataclass
class OrderResult:
    order_id: str
    market_id: str
    side: str              # "YES" | "NO"
    size: float
    price: float
    status: str
    filled_at: Optional[datetime] = None


class CoinbasePredictionsClient:
    def __init__(self, cfg: CoinbaseConfig):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    def _sign(self, method: str, path: str, body: str = "") -> dict:
        """Generate Coinbase API auth headers."""
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self.cfg.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "CB-ACCESS-KEY": self.cfg.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.cfg.api_passphrase,
            "Content-Type": "application/json",
        }

    async def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.cfg.predictions_base_url}{path}"
        headers = self._sign("GET", f"/api/v1/predictions{path}")
        async with self._session.get(url, headers=headers, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _post(self, path: str, body: dict) -> dict:
        import json
        body_str = json.dumps(body)
        url = f"{self.cfg.predictions_base_url}{path}"
        headers = self._sign("POST", f"/api/v1/predictions{path}", body_str)
        async with self._session.post(url, headers=headers, data=body_str) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_markets(self, status: str = "open") -> list[Market]:
        """Fetch all open prediction markets."""
        data = await self._get("/markets", params={"status": status})
        markets = []
        for m in data.get("markets", []):
            try:
                markets.append(self._parse_market(m))
            except Exception as e:
                logger.warning(f"Failed to parse market {m.get('id')}: {e}")
        return markets

    async def get_market(self, market_id: str) -> Market:
        """Fetch a single market by ID."""
        data = await self._get(f"/markets/{market_id}")
        return self._parse_market(data["market"])

    async def get_orderbook(self, market_id: str) -> dict:
        """Fetch the order book for a market."""
        return await self._get(f"/markets/{market_id}/orderbook")

    async def place_order(
        self,
        market_id: str,
        side: str,       # "YES" | "NO"
        size_usd: float,
        price: float,    # limit price 0–1
        order_type: str = "limit",
    ) -> OrderResult:
        """Place a prediction market order."""
        body = {
            "market_id": market_id,
            "side": side,
            "size": str(size_usd),
            "price": str(price),
            "type": order_type,
        }
        data = await self._post("/orders", body)
        order = data["order"]
        return OrderResult(
            order_id=order["id"],
            market_id=market_id,
            side=side,
            size=size_usd,
            price=price,
            status=order["status"],
        )

    async def cancel_order(self, order_id: str) -> bool:
        data = await self._post(f"/orders/{order_id}/cancel", {})
        return data.get("success", False)

    async def get_positions(self) -> list[dict]:
        """Fetch open positions."""
        data = await self._get("/positions")
        return data.get("positions", [])

    def _parse_market(self, m: dict) -> Market:
        yes = m.get("yes_token", {})
        no = m.get("no_token", {})
        return Market(
            market_id=m["id"],
            question=m.get("question", ""),
            yes_price=float(yes.get("mid_price", 0.5)),
            no_price=float(no.get("mid_price", 0.5)),
            yes_bid=float(yes.get("bid", 0)),
            yes_ask=float(yes.get("ask", 1)),
            no_bid=float(no.get("bid", 0)),
            no_ask=float(no.get("ask", 1)),
            volume_usd=float(m.get("volume_24h_usd", 0)),
            liquidity_usd=float(m.get("liquidity_usd", 0)),
            resolves_at=datetime.fromisoformat(m["resolves_at"].replace("Z", "+00:00")),
            status=m.get("status", "open"),
            outcome=m.get("outcome"),
        )
