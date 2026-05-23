"""
Polymarket async trading client.

Market data:    Gamma API  (https://gamma-api.polymarket.com)  — public, no auth
Order execution: CLOB API  (https://clob.polymarket.com)       — requires Polygon wallet

One-time credential setup:
  1. Set POLY_PRIVATE_KEY (Ethereum hex private key, with or without 0x prefix)
  2. Run: python -c "from api_clients.polymarket_client import print_api_creds; print_api_creds()"
  3. Copy POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE into your .env

Polymarket runs on Polygon (USDC). Balances and sizes are in USDC.
Shares = number of outcome tokens held. Each share pays $1 USDC at resolution.

Docs: https://docs.polymarket.com/developers/clob/introduction
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"
POLY_CHAIN_ID = 137   # Polygon mainnet

_GAMMA_RETRY_ATTEMPTS = 3
_GAMMA_RETRY_BASE_SEC = 2.0
_CB_ERRORS_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_ERRORS", "5"))
_CB_PAUSE_SEC = int(os.getenv("CIRCUIT_BREAKER_PAUSE_SEC", "120"))


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Market:
    """Polymarket binary market, normalised from Gamma + CLOB APIs."""
    market_id: str          # condition_id hex (primary key, replaces Kalshi ticker)
    group_id: str           # groupItemTitle or category — used by auto_correlator
    question: str
    slug: str
    yes_token_id: str       # CLOB token ID for YES outcome (needed to place orders)
    no_token_id: str        # CLOB token ID for NO outcome
    yes_bid: float          # Best YES bid (0–1)
    yes_ask: float          # Best YES ask (0–1)
    no_bid: float           # Best NO bid (0–1)
    no_ask: float           # Best NO ask (0–1)
    yes_price: float        # YES mid price
    no_price: float         # NO mid price
    volume: float           # Total USDC volume
    volume_24h: float       # 24h USDC volume
    liquidity_usd: float    # Available USDC liquidity
    open_interest: float    # Open interest in USDC
    last_price: float       # Last traded price
    close_time: datetime    # Resolution time
    status: str             # "open" | "closed" | "settled"
    result: Optional[str] = None      # "yes" | "no" | None
    floor_strike: Optional[float] = None   # e.g. $100k for "above $100k" markets
    cap_strike: Optional[float] = None     # upper bound for range markets
    subtitle_yes: str = ""  # e.g. "Yes" or outcome label
    subtitle_no: str = ""
    category: str = ""      # e.g. "crypto", "politics", "sports"


@dataclass
class Orderbook:
    """Polymarket CLOB order book for a single outcome token."""
    token_id: str
    bids: list[list[str]]   # [[price, size], …] sorted best first (highest price)
    asks: list[list[str]]   # [[price, size], …] sorted best first (lowest price)

    @property
    def best_bid(self) -> float:
        return float(self.bids[0][0]) if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return float(self.asks[0][0]) if self.asks else 1.0


@dataclass
class OrderResult:
    order_id: str
    market_id: str          # condition_id
    token_id: str           # yes_token_id or no_token_id
    side: str               # "yes" | "no"
    action: str             # "buy" | "sell"
    size: float             # shares
    price: float            # 0.01–0.99
    status: str
    size_matched: float = 0.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PolymarketClient:
    """
    Async Polymarket trading client.

    Market discovery uses the public Gamma API (fast, paginated).
    Order execution uses the authenticated CLOB API via py-clob-client-v2
    (wrapped in asyncio.to_thread for async compatibility).

    If py-clob-client-v2 is not installed or credentials are absent,
    the client operates in read-only mode — DRY_RUN=true still works.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None
        self._clob = None   # py_clob_client_v2.ClobClient, set in __aenter__
        self._clob_errors: int = 0
        self._circuit_open_until: Optional[datetime] = None
        self._recent_cache: list[Market] = []
        self._recent_cache_at: Optional[datetime] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"Accept": "application/json"},
        )
        if self.cfg.private_key:
            try:
                await asyncio.to_thread(self._init_clob)
            except Exception as e:
                logger.warning(f"CLOB client init failed: {e} — running read-only")
        return self

    def _init_clob(self):
        from py_clob_client_v2.client import ClobClient
        from py_clob_client_v2.clob_types import ApiCreds
        creds = None
        if self.cfg.api_key and self.cfg.api_secret:
            creds = ApiCreds(
                api_key=self.cfg.api_key,
                api_secret=self.cfg.api_secret,
                api_passphrase=self.cfg.api_passphrase,
            )
        self._clob = ClobClient(
            host=CLOB_BASE,
            chain_id=POLY_CHAIN_ID,
            key=self.cfg.private_key,
            creds=creds,
        )
        logger.info("CLOB client initialised (v2)")

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_balance(self) -> float:
        """Return USDC balance on Polymarket."""
        if not self._clob:
            return 0.0
        try:
            from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
            result = await asyncio.to_thread(
                self._clob.get_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
            )
            if isinstance(result, dict):
                return float(result.get("balance", 0) or 0)
            return 0.0
        except Exception as e:
            logger.error(f"get_balance: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Circuit breaker helpers
    # ------------------------------------------------------------------

    def _circuit_ok(self) -> bool:
        """Return True if CLOB is available (circuit closed)."""
        if self._circuit_open_until:
            if datetime.now(timezone.utc) < self._circuit_open_until:
                return False
            self._circuit_open_until = None
            self._clob_errors = 0
            logger.info("CLOB circuit breaker reset — resuming order placement")
        return True

    def _on_clob_error(self) -> None:
        self._clob_errors += 1
        if self._clob_errors >= _CB_ERRORS_THRESHOLD and not self._circuit_open_until:
            self._circuit_open_until = datetime.now(timezone.utc) + timedelta(
                seconds=_CB_PAUSE_SEC
            )
            logger.warning(
                f"CLOB circuit breaker OPEN after {self._clob_errors} errors "
                f"— pausing {_CB_PAUSE_SEC}s"
            )

    def _on_clob_success(self) -> None:
        if self._clob_errors > 0:
            self._clob_errors = max(0, self._clob_errors - 1)

    @property
    def circuit_open(self) -> bool:
        return not self._circuit_ok()

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    async def get_all_markets(self, max_pages: int = 10) -> list[Market]:
        """
        Fetch active binary markets from the Gamma API.
        Ordered by 24h volume descending (most liquid first).
        Retries up to 3 times with exponential backoff on transient errors.
        """
        all_markets: list[Market] = []
        offset = 0
        limit = 100

        for _ in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": limit,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            }
            batch = await self._fetch_gamma_page(params)
            if batch is None:
                break

            for raw in batch:
                m = self._parse_market(raw)
                if m:
                    all_markets.append(m)

            if len(batch) < limit:
                break
            offset += limit
            await asyncio.sleep(0.3)

        logger.info(f"Polymarket: loaded {len(all_markets)} active markets")
        return all_markets

    async def get_recent_markets(self, limit: int = 200, ttl_sec: int = 25) -> list[Market]:
        """
        Fetch recently-opened markets sorted by startDate descending.

        Rolling short-duration markets (e.g. 5-minute BTC up/down) have very low
        24h volume and appear at offset 3000+ in the volume-sorted list.  Sorting
        by startDate descending surfaces them on the first page.

        Results are cached for `ttl_sec` seconds (default 25s) so that the
        BTC 5-min loop (refresh every 20s) doesn't make redundant API calls.
        """
        now = datetime.now(timezone.utc)
        if (
            self._recent_cache
            and self._recent_cache_at is not None
            and (now - self._recent_cache_at).total_seconds() < ttl_sec
        ):
            return self._recent_cache

        params = {
            "active": "true",
            "closed": "false",
            "limit": min(limit, 200),
            "order": "startDate",
            "ascending": "false",
        }
        batch = await self._fetch_gamma_page(params)
        if not batch:
            return self._recent_cache  # return stale on error rather than empty
        markets = []
        for raw in batch:
            m = self._parse_market(raw)
            if m:
                markets.append(m)
        self._recent_cache = markets
        self._recent_cache_at = now
        logger.debug(f"Polymarket: loaded {len(markets)} recent markets")
        return markets

    async def _fetch_gamma_page(self, params: dict) -> Optional[list]:
        """Fetch one page from the Gamma API with retry/backoff."""
        for attempt in range(_GAMMA_RETRY_ATTEMPTS):
            try:
                async with self._session.get(
                    f"{GAMMA_BASE}/markets", params=params
                ) as resp:
                    if resp.status == 429:
                        wait = _GAMMA_RETRY_BASE_SEC * (2 ** attempt)
                        logger.warning(f"Gamma API rate-limited, waiting {wait:.0f}s")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        logger.warning(f"Gamma API {resp.status}")
                        return None
                    data = await resp.json()
                return data if isinstance(data, list) else data.get("markets", [])
            except Exception as e:
                if attempt < _GAMMA_RETRY_ATTEMPTS - 1:
                    wait = _GAMMA_RETRY_BASE_SEC * (2 ** attempt)
                    logger.debug(f"Gamma page fetch error ({e}), retry in {wait:.0f}s")
                    await asyncio.sleep(wait)
                else:
                    logger.warning(f"Gamma page fetch failed: {e}")
        return None

    async def get_orderbook(self, token_id: str) -> Orderbook:
        """
        Fetch live order book for a single outcome token from the CLOB API.
        No auth required for reading order books.
        """
        try:
            async with self._session.get(
                f"{CLOB_BASE}/book", params={"token_id": token_id}
            ) as resp:
                if resp.status != 200:
                    return Orderbook(token_id, [], [])
                data = await resp.json()

            bids = [[str(b["price"]), str(b["size"])] for b in data.get("bids", [])]
            asks = [[str(a["price"]), str(a["size"])] for a in data.get("asks", [])]
            # bids sorted descending (best bid first), asks ascending (best ask first)
            bids.sort(key=lambda x: float(x[0]), reverse=True)
            asks.sort(key=lambda x: float(x[0]))
            return Orderbook(token_id=token_id, bids=bids, asks=asks)
        except Exception as e:
            logger.debug(f"Orderbook {token_id[:8]}: {e}")
            return Orderbook(token_id, [], [])

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(
        self,
        token_id: str,
        action: str,    # "buy" | "sell"
        price: float,   # 0.01–0.99
        size: float,    # shares (cost = size * price USDC when buying)
        order_type: str = "GTC",
    ) -> Optional[OrderResult]:
        """
        Place a limit order on the Polymarket CLOB.

        token_id   : yes_token_id or no_token_id for the market
        action     : "buy" to acquire shares, "sell" to close a position
        price      : limit price (0.01–0.99)
        size       : number of shares
        order_type : "GTC" (default), "FOK" (fill-or-kill)

        Polymarket fee: ~2 bps on the taker side (embedded in spread).
        """
        if not self._clob:
            raise RuntimeError(
                "CLOB client not initialised — set POLY_PRIVATE_KEY and API creds"
            )
        if not self._circuit_ok():
            logger.warning("place_order: CLOB circuit breaker open — skipping")
            return None
        try:
            from py_clob_client_v2.clob_types import OrderArgsV2, OrderType

            side_str = "BUY" if action.lower() == "buy" else "SELL"
            ot = OrderType.FOK if order_type == "FOK" else OrderType.GTC

            args = OrderArgsV2(
                token_id=token_id,
                price=round(price, 4),
                size=round(size, 2),
                side=side_str,
            )
            result = await asyncio.to_thread(
                self._clob.create_and_post_order, args, order_type=ot
            )
            if not result:
                self._on_clob_error()
                return None

            self._on_clob_success()
            order_id = result.get("orderID") or result.get("order_id", "unknown")
            status   = result.get("status", "placed")
            matched  = float(result.get("sizeMatched") or 0)

            return OrderResult(
                order_id=order_id,
                market_id="",
                token_id=token_id,
                side="yes",   # caller sets the correct side label
                action=action.lower(),
                size=size,
                price=price,
                status=status,
                size_matched=matched,
            )
        except Exception as e:
            self._on_clob_error()
            logger.error(f"place_order ({token_id[:8]}): {e}")
            return None

    async def cancel_order(self, order_id: str) -> float:
        """
        Cancel a resting order.  Returns the matched size (shares) extracted
        from the API response if available, otherwise 0.  Returns 0 on error.
        """
        if not self._clob:
            return 0.0
        try:
            from py_clob_client_v2.clob_types import OrderPayload
            result = await asyncio.to_thread(
                self._clob.cancel_order, OrderPayload(orderID=order_id)
            )
            if isinstance(result, dict):
                return float(result.get("sizeMatched") or 0)
            return 0.0
        except Exception as e:
            logger.error(f"cancel_order {order_id}: {e}")
            return 0.0

    async def get_trades(self, token_id: str, limit: int = 100) -> list[dict]:
        """
        Fetch recent trades from the CLOB API for a single outcome token.
        Used by KyleLambdaTracker to estimate price impact.
        Each item: {price, size, side, transactionHash, timestamp}.
        """
        try:
            async with self._session.get(
                f"{CLOB_BASE}/trades",
                params={"tokenID": token_id, "limit": limit},
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
            if isinstance(data, list):
                return data
            return data.get("trades", [])
        except Exception as e:
            logger.debug(f"get_trades {token_id[:8]}: {e}")
            return []

    async def get_positions(self) -> list[dict]:
        """
        Return actual token holdings from the Polymarket Data API.
        Each dict has: conditionId, asset, size, avgPrice, outcome, curPrice, etc.
        Falls back to empty list if wallet address not configured.
        """
        if not self.cfg.wallet_address or not self._session:
            return []
        try:
            url = "https://data-api.polymarket.com/positions"
            params = {"user": self.cfg.wallet_address, "sizeThreshold": "0.01"}
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.debug(f"get_positions: HTTP {resp.status}")
                    return []
                data = await resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"get_positions: {e}")
            return []

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_market(self, raw: dict) -> Optional[Market]:
        try:
            outcomes    = _json_field(raw, "outcomes", [])
            prices_raw  = _json_field(raw, "outcomePrices", [])
            tokens_list = _json_field(raw, "tokens", [])
            clob_ids    = _json_field(raw, "clobTokenIds", [])

            # Must be binary
            if len(outcomes) != 2:
                return None

            # Find YES index
            yes_idx = next(
                (i for i, o in enumerate(outcomes) if str(o).lower() in ("yes", "true")),
                0,
            )
            no_idx = 1 - yes_idx

            # Prices (last traded / mid, from Gamma API)
            yes_price_raw = float(prices_raw[yes_idx]) if len(prices_raw) > yes_idx else 0.5
            no_price_raw  = float(prices_raw[no_idx])  if len(prices_raw) > no_idx  else 0.5

            # Token IDs — prefer tokens list, fall back to clobTokenIds array
            yes_token_id = no_token_id = ""
            if tokens_list and len(tokens_list) >= 2:
                for t in tokens_list:
                    outcome = str(t.get("outcome", "")).lower()
                    tid = str(t.get("token_id", ""))
                    price = float(t.get("price", 0) or 0)
                    if outcome in ("yes", "true"):
                        yes_token_id = tid
                        yes_price_raw = price or yes_price_raw
                    else:
                        no_token_id = tid
                        no_price_raw = price or no_price_raw
            elif len(clob_ids) >= 2:
                yes_token_id = str(clob_ids[yes_idx])
                no_token_id  = str(clob_ids[no_idx])

            if not yes_token_id or not no_token_id:
                return None

            # Resolution time
            close_time = _parse_date(
                raw.get("endDate") or raw.get("end_date_iso") or raw.get("resolutionTime")
            )

            # Category / group
            tags = _json_field(raw, "tags", [])
            category = ""
            if tags:
                first = tags[0]
                category = (first.get("label", "") if isinstance(first, dict) else str(first)).lower()

            group_id = (
                raw.get("groupItemTitle")
                or raw.get("series_color")
                or category
                or ""
            )

            # Numeric strikes (for crypto above/range markets)
            question = str(raw.get("question") or raw.get("title") or "")
            floor_strike, cap_strike = _extract_strikes(question)

            # Status
            resolved = raw.get("resolved", False)
            closed   = raw.get("closed", False) or raw.get("archived", False)
            if resolved:
                status = "settled"
            elif closed:
                status = "closed"
            else:
                status = "open"

            result = None
            for key in ("result", "winner"):
                rv = str(raw.get(key, "")).lower()
                if rv in ("yes", "true"):
                    result = "yes"
                    break
                if rv in ("no", "false"):
                    result = "no"
                    break

            # Gamma API prices serve as both bid and ask (no spread info)
            yes_mid = round(yes_price_raw, 4)
            no_mid  = round(no_price_raw,  4)

            condition_id = str(raw.get("conditionId") or raw.get("condition_id") or raw.get("id", ""))

            # Event slug routes to a real Polymarket page; market slug often doesn't
            _events = raw.get("events") or []
            _first_event = _events[0] if _events else None
            _event_slug = str(_first_event.get("slug", "")) if isinstance(_first_event, dict) else ""
            _market_slug = str(raw.get("slug", ""))
            slug = _event_slug or _market_slug

            return Market(
                market_id=condition_id,
                group_id=str(group_id),
                question=question,
                slug=slug,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                yes_bid=yes_mid,
                yes_ask=yes_mid,
                no_bid=no_mid,
                no_ask=no_mid,
                yes_price=yes_mid,
                no_price=no_mid,
                volume=float(raw.get("volume", 0) or 0),
                volume_24h=float(raw.get("volume24hr", 0) or 0),
                liquidity_usd=float(raw.get("liquidity", 0) or 0),
                open_interest=float(raw.get("openInterest", 0) or 0),
                last_price=yes_mid,
                close_time=close_time,
                status=status,
                result=result,
                floor_strike=floor_strike,
                cap_strike=cap_strike,
                subtitle_yes=str(outcomes[yes_idx]),
                subtitle_no=str(outcomes[no_idx]),
                category=category,
            )
        except Exception as e:
            logger.debug(f"Market parse: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def usdc_to_shares(usdc: float, price: float) -> float:
        """How many shares can we buy for `usdc` USDC at `price` per share?"""
        if price <= 0:
            return 0.0
        return max(0.0, usdc / price)

    @staticmethod
    def fee_estimate(price: float, size: float) -> float:
        """
        Estimated Polymarket taker fee.
        Polymarket charges ~2 bps (0.02%) of notional on taker orders.
        For simplicity we model as 1% of cost (conservative).
        """
        return price * size * 0.01


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _json_field(raw: dict, key: str, default):
    val = raw.get(key, default)
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return default
    return val if val is not None else default


def _parse_date(raw) -> datetime:
    _now = datetime.now(timezone.utc)
    default = _now.replace(year=_now.year + 1)
    if not raw:
        return default
    try:
        s = str(raw).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return default


_STRIKE_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*([kKmM]?)", re.IGNORECASE)


def _extract_strikes(question: str) -> tuple[Optional[float], Optional[float]]:
    """Extract floor / cap strike prices from a market question string."""
    nums: list[float] = []
    for m in _STRIKE_RE.finditer(question):
        try:
            val = float(m.group(1).replace(",", ""))
            suffix = m.group(2).lower()
            if suffix == "k":
                val *= 1_000
            elif suffix == "m":
                val *= 1_000_000
            nums.append(val)
        except ValueError:
            pass

    q = question.lower()
    if ("between" in q or " to " in q or "–" in q or "-" in q) and len(nums) >= 2:
        return (min(nums), max(nums))
    if any(w in q for w in ("above", "over", "exceed", "more than", "higher than")) and nums:
        return (max(nums), None)
    if any(w in q for w in ("below", "under", "less than", "lower than")) and nums:
        return (None, min(nums))
    if len(nums) == 1:
        return (nums[0], None)
    return (None, None)


# ---------------------------------------------------------------------------
# One-time credential helper (run manually)
# ---------------------------------------------------------------------------

def print_api_creds():
    """
    Derive and print Polymarket API credentials from your Ethereum private key.
    Run once, then set the printed values in your .env file.

    Usage:
        python -c "from api_clients.polymarket_client import print_api_creds; print_api_creds()"

    Requires POLY_PRIVATE_KEY set in environment or .env.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("POLY_PRIVATE_KEY", "")
    if not key:
        print("ERROR: POLY_PRIVATE_KEY not set")
        return
    try:
        from py_clob_client_v2.client import ClobClient
        c = ClobClient(host=CLOB_BASE, chain_id=POLY_CHAIN_ID, key=key)
        creds = c.create_or_derive_api_key()
        print("Add these to your .env:")
        print(f"POLY_API_KEY={creds.api_key}")
        print(f"POLY_API_SECRET={creds.api_secret}")
        print(f"POLY_PASSPHRASE={creds.api_passphrase}")
    except ImportError:
        print("Install py-clob-client-v2: pip install py-clob-client-v2")
    except Exception as e:
        print(f"Error: {e}")
