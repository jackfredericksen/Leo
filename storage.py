"""
SQLite storage for Leo — trades, opportunities, and P&L tracking.
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional

from config import StorageConfig

logger = logging.getLogger(__name__)


class Storage:
    def __init__(self, cfg: StorageConfig):
        self.db_path = cfg.db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id       TEXT NOT NULL,
                    question        TEXT,
                    arb_type        TEXT,
                    leg_yes_price   REAL,
                    leg_no_price    REAL,
                    size_usd        REAL,
                    net_profit_pct  REAL,
                    dry_run         INTEGER DEFAULT 1,
                    status          TEXT,
                    yes_order_id    TEXT,
                    no_order_id     TEXT,
                    created_at      TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS opportunities (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id       TEXT NOT NULL,
                    question        TEXT,
                    arb_type        TEXT,
                    gross_profit_pct REAL,
                    net_profit_pct  REAL,
                    acted_on        INTEGER DEFAULT 0,
                    detected_at     TEXT
                );
            """)

    def log_trade(
        self,
        market_id: str,
        question: str,
        arb_type: str,
        leg_yes_price: Optional[float],
        leg_no_price: Optional[float],
        size_usd: float,
        net_profit_pct: float,
        dry_run: bool,
        status: str,
        yes_order_id: str = None,
        no_order_id: str = None,
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades
                (market_id, question, arb_type, leg_yes_price, leg_no_price,
                 size_usd, net_profit_pct, dry_run, status, yes_order_id, no_order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id, question, arb_type, leg_yes_price, leg_no_price,
                    size_usd, net_profit_pct, int(dry_run), status,
                    yes_order_id, no_order_id,
                ),
            )

    def log_opportunity(self, market_id: str, question: str, arb_type: str,
                        gross_pct: float, net_pct: float, acted_on: bool,
                        detected_at: datetime):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO opportunities
                (market_id, question, arb_type, gross_profit_pct, net_profit_pct,
                 acted_on, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (market_id, question, arb_type, gross_pct, net_pct,
                 int(acted_on), detected_at.isoformat()),
            )

    def get_recent_trades(self, limit: int = 50) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()

    def get_pnl_summary(self) -> dict:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN dry_run=0 THEN size_usd ELSE 0 END) as total_deployed_usd,
                    AVG(net_profit_pct) as avg_net_profit_pct,
                    SUM(CASE WHEN dry_run=0 THEN size_usd * net_profit_pct ELSE 0 END) as estimated_pnl_usd
                FROM trades
                WHERE status IN ('placed', 'simulated')
            """).fetchone()
            return dict(row) if row else {}
