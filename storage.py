"""
SQLite storage for Leo — trades, opportunities, and P&L tracking.
"""

import sqlite3
import logging
from datetime import datetime
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
                    side            TEXT,
                    yes_price       REAL,
                    no_price        REAL,
                    contracts       INTEGER,
                    net_profit_pct  REAL,
                    size_usd        REAL,
                    dry_run         INTEGER DEFAULT 1,
                    status          TEXT,
                    yes_order_id    TEXT,
                    no_order_id     TEXT,
                    outcome         TEXT DEFAULT 'pending',
                    created_at      TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS opportunities (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id        TEXT NOT NULL,
                    question         TEXT,
                    arb_type         TEXT,
                    gross_profit_pct REAL,
                    net_profit_pct   REAL,
                    acted_on         INTEGER DEFAULT 0,
                    detected_at      TEXT
                );

                CREATE TABLE IF NOT EXISTS pnl_history (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    taken_at     TEXT DEFAULT (datetime('now')),
                    balance_usd  REAL,
                    unrealized   REAL,
                    realized     REAL,
                    total_pnl    REAL
                );
            """)
            # Migrations for existing databases
            for migration in [
                "ALTER TABLE trades ADD COLUMN outcome TEXT DEFAULT 'pending'",
                "ALTER TABLE trades ADD COLUMN side TEXT",
                "ALTER TABLE trades ADD COLUMN size_usd REAL",
            ]:
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError as e:
                    logger.debug(f"Migration skipped: {e}")

    def log_trade(
        self,
        market_id: str,
        question: str,
        arb_type: str,
        yes_price: float,
        no_price: float,
        contracts: int,
        net_profit_pct: float,
        dry_run: bool,
        status: str,
        side: Optional[str] = None,
        yes_order_id: Optional[str] = None,
        no_order_id: Optional[str] = None,
        size_usd: Optional[float] = None,
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades
                (market_id, question, arb_type, side, yes_price, no_price,
                 contracts, net_profit_pct, dry_run, status,
                 yes_order_id, no_order_id, size_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id, question, arb_type, side, yes_price, no_price,
                    contracts, net_profit_pct, int(dry_run), status,
                    yes_order_id, no_order_id, size_usd,
                ),
            )

    def log_opportunity(
        self,
        market_id: str,
        question: str,
        arb_type: str,
        gross_pct: float,
        net_pct: float,
        acted_on: bool,
        detected_at: datetime,
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO opportunities
                (market_id, question, arb_type, gross_profit_pct,
                 net_profit_pct, acted_on, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id, question, arb_type, gross_pct, net_pct,
                    int(acted_on), detected_at.isoformat(),
                ),
            )

    def get_recent_trades(self, limit: int = 50) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def log_pnl_snapshot(
        self,
        balance: float,
        unrealized: float,
        realized: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO pnl_history (balance_usd, unrealized, realized, total_pnl) "
                "VALUES (?, ?, ?, ?)",
                (balance, unrealized, realized, unrealized + realized),
            )

    def get_pnl_history(self, hours: int = 24) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT taken_at, balance_usd, unrealized, realized, total_pnl "
                "FROM pnl_history "
                "WHERE taken_at > datetime('now', ?) "
                "ORDER BY taken_at ASC",
                (f"-{hours} hours",),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_estimated_pnl_history(self, hours: int = 24) -> list[dict]:
        """
        Cumulative estimated P&L from trades over time.
        Used as a fallback when the position manager returns all-zero snapshots
        (e.g. dry-run mode or no live Polymarket positions).
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                WITH ordered AS (
                    SELECT
                        created_at,
                        COALESCE(
                            size_usd,
                            contracts * CASE
                                WHEN side = 'both' THEN COALESCE(yes_price,0) + COALESCE(no_price,0)
                                WHEN side = 'no'   THEN COALESCE(no_price, 0.5)
                                ELSE                    COALESCE(yes_price, 0.5)
                            END
                        ) * net_profit_pct AS est_pnl
                    FROM trades
                    WHERE status IN ('placed', 'simulated')
                      AND created_at > datetime('now', ?)
                )
                SELECT
                    created_at AS taken_at,
                    SUM(est_pnl) OVER (
                        ORDER BY created_at
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS total_pnl
                FROM ordered
                ORDER BY created_at ASC
                """,
                (f"-{hours} hours",),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_strategy_pnl(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    arb_type,
                    COUNT(*)        AS trades,
                    AVG(net_profit_pct) AS avg_edge,
                    SUM(
                        COALESCE(
                            size_usd,
                            contracts * CASE
                                WHEN side = 'both' THEN COALESCE(yes_price,0) + COALESCE(no_price,0)
                                WHEN side = 'no'   THEN COALESCE(no_price, 0.5)
                                ELSE                    COALESCE(yes_price, 0.5)
                            END
                        ) * net_profit_pct
                    )               AS est_pnl
                FROM trades
                WHERE status IN ('placed', 'simulated')
                GROUP BY arb_type
                ORDER BY est_pnl DESC
                """,
            ).fetchall()
        return [dict(r) for r in rows]

    def update_trade_outcome(self, trade_id: int, outcome: str) -> None:
        """Mark a trade as 'won', 'lost', or 'pending'."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET outcome = ? WHERE id = ?",
                (outcome, trade_id),
            )

    def get_unresolved_trades(self) -> list[sqlite3.Row]:
        """Return all trades (live and paper) still marked pending."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, market_id, question, arb_type, side, yes_price, no_price "
                "FROM trades WHERE outcome = 'pending' "
                "AND status IN ('placed', 'simulated') ORDER BY created_at DESC LIMIT 200"
            ).fetchall()

    def get_strategy_win_rates(self) -> dict[str, dict]:
        """
        Return per-strategy win rate computed from resolved trades.
        Only includes trades with outcome in ('won', 'lost').
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    arb_type,
                    COUNT(*) AS total,
                    SUM(CASE WHEN outcome = 'won' THEN 1 ELSE 0 END) AS wins,
                    CAST(SUM(CASE WHEN outcome = 'won' THEN 1 ELSE 0 END) AS FLOAT)
                      / NULLIF(COUNT(CASE WHEN outcome IN ('won','lost') THEN 1 END), 0)
                      AS win_rate
                FROM trades
                WHERE outcome IN ('won', 'lost')
                GROUP BY arb_type
                """
            ).fetchall()
        return {
            row["arb_type"]: {
                "total": row["total"],
                "wins": row["wins"],
                "win_rate": row["win_rate"] or 0.0,
            }
            for row in rows
        }

    def get_pnl_summary(self) -> dict:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(contracts) as total_contracts,
                    AVG(net_profit_pct) as avg_net_profit_pct,
                    SUM(
                        COALESCE(
                            size_usd,
                            contracts * CASE
                                WHEN side = 'both' THEN COALESCE(yes_price,0) + COALESCE(no_price,0)
                                WHEN side = 'no'   THEN COALESCE(no_price, 0.5)
                                ELSE                    COALESCE(yes_price, 0.5)
                            END
                        ) * net_profit_pct
                    ) as estimated_pnl_usd
                FROM trades
                WHERE status IN ('placed', 'simulated')
            """).fetchone()
            return dict(row) if row else {}
