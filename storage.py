"""
SQLite storage for Leo — trades, opportunities, and P&L tracking.
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional

from config import StorageConfig

logger = logging.getLogger(__name__)


def _trade_stake_usd(row: sqlite3.Row) -> float:
    """USD deployed on a trade row."""
    if row["size_usd"]:
        return float(row["size_usd"])
    contracts = float(row["contracts"] or 0)
    side = (row["side"] or "").lower()
    if side == "both":
        return contracts * (
            float(row["yes_price"] or 0) + float(row["no_price"] or 0)
        )
    if side == "no":
        return contracts * float(row["no_price"] or 0.5)
    return contracts * float(row["yes_price"] or 0.5)


def trade_realized_pnl(row: sqlite3.Row) -> float:
    """
    Actual P&L for a resolved trade. Pending trades return 0.

    Dry-run rows are paper-only — this uses real win/loss math once
    outcome_tracker marks them won/lost against Polymarket settlement.
    """
    outcome = row["outcome"]
    if outcome not in ("won", "lost"):
        return 0.0

    stake = _trade_stake_usd(row)
    if stake <= 0:
        return 0.0

    side = (row["side"] or "").lower()
    if outcome == "lost":
        return -stake

    if side == "both":
        # Overround arb: net_profit_pct is the locked-in edge at entry
        return stake * float(row["net_profit_pct"] or 0)

    if side == "yes":
        price = float(row["yes_price"] or 0)
        if price <= 0:
            return 0.0
        return stake * ((1.0 - price) / price)

    if side == "no":
        price = float(row["no_price"] or 0)
        if price <= 0:
            return 0.0
        return stake * ((1.0 - price) / price)

    return 0.0


def trade_theoretical_pnl(row: sqlite3.Row) -> float:
    """Optimistic edge×stake for pending paper trades (UI feedback only)."""
    if row["outcome"] in ("won", "lost"):
        return 0.0
    stake = _trade_stake_usd(row)
    if stake <= 0:
        return 0.0
    return stake * float(row["net_profit_pct"] or 0)


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
        Cumulative paper P&L from resolved trades only (dry-run path).
        Pending simulated trades contribute $0 until outcome_tracker settles them.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trades
                WHERE status IN ('placed', 'simulated')
                  AND outcome IN ('won', 'lost')
                  AND created_at > datetime('now', ?)
                ORDER BY created_at ASC
                """,
                (f"-{hours} hours",),
            ).fetchall()

        cumulative = 0.0
        history = []
        for row in rows:
            cumulative += trade_realized_pnl(row)
            history.append({
                "taken_at": row["created_at"],
                "total_pnl": round(cumulative, 2),
            })
        return history

    def get_strategy_pnl(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trades
                WHERE status IN ('placed', 'simulated')
                ORDER BY created_at DESC
                """,
            ).fetchall()

        by_type: dict[str, dict] = {}
        for row in rows:
            key = row["arb_type"] or "unknown"
            if key not in by_type:
                by_type[key] = {"arb_type": key, "trades": 0, "avg_edge": 0.0, "est_pnl": 0.0, "_edges": []}
            bucket = by_type[key]
            bucket["trades"] += 1
            bucket["_edges"].append(float(row["net_profit_pct"] or 0))
            bucket["est_pnl"] += trade_realized_pnl(row)

        result = []
        for bucket in by_type.values():
            edges = bucket.pop("_edges")
            bucket["avg_edge"] = sum(edges) / len(edges) if edges else 0.0
            bucket["est_pnl"] = round(bucket["est_pnl"], 2)
            result.append(bucket)
        result.sort(key=lambda x: x["est_pnl"], reverse=True)
        return result

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

    def get_calibration_report(self) -> dict:
        """
        Per-strategy calibration: predicted edge vs realized win rate.
        Useful for dry-run validation before going live.
        """
        with self._connect() as conn:
            strategies = conn.execute(
                """
                SELECT
                    arb_type,
                    COUNT(*) AS total,
                    SUM(CASE WHEN outcome IN ('won','lost') THEN 1 ELSE 0 END) AS resolved,
                    SUM(CASE WHEN outcome = 'won' THEN 1 ELSE 0 END) AS wins,
                    AVG(net_profit_pct) AS avg_predicted_edge,
                    AVG(CASE WHEN outcome = 'won' THEN net_profit_pct END) AS avg_edge_wins,
                    AVG(CASE WHEN outcome = 'lost' THEN net_profit_pct END) AS avg_edge_losses,
                    SUM(
                        COALESCE(
                            size_usd,
                            contracts * CASE
                                WHEN side = 'both' THEN COALESCE(yes_price,0) + COALESCE(no_price,0)
                                WHEN side = 'no'   THEN COALESCE(no_price, 0.5)
                                ELSE                    COALESCE(yes_price, 0.5)
                            END
                        ) * net_profit_pct
                    ) AS est_pnl
                FROM trades
                WHERE status IN ('placed', 'simulated')
                GROUP BY arb_type
                ORDER BY total DESC
                """
            ).fetchall()

            buckets = conn.execute(
                """
                SELECT
                    arb_type,
                    CASE
                        WHEN net_profit_pct < 0.02 THEN '0-2%'
                        WHEN net_profit_pct < 0.04 THEN '2-4%'
                        WHEN net_profit_pct < 0.06 THEN '4-6%'
                        WHEN net_profit_pct < 0.08 THEN '6-8%'
                        WHEN net_profit_pct < 0.10 THEN '8-10%'
                        ELSE '10%+'
                    END AS edge_bucket,
                    COUNT(*) AS n,
                    SUM(CASE WHEN outcome = 'won' THEN 1 ELSE 0 END) AS wins,
                    AVG(net_profit_pct) AS avg_edge
                FROM trades
                WHERE outcome IN ('won', 'lost')
                GROUP BY arb_type, edge_bucket
                ORDER BY arb_type, edge_bucket
                """
            ).fetchall()

            summary_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_trades,
                    SUM(CASE WHEN outcome IN ('won','lost') THEN 1 ELSE 0 END) AS resolved,
                    SUM(CASE WHEN outcome = 'won' THEN 1 ELSE 0 END) AS wins,
                    AVG(net_profit_pct) AS avg_edge
                FROM trades WHERE status IN ('placed', 'simulated')
                """
            ).fetchone()

            trade_rows = conn.execute(
                "SELECT * FROM trades WHERE status IN ('placed', 'simulated')"
            ).fetchall()

        pnl_by_type: dict[str, float] = {}
        for tr in trade_rows:
            key = tr["arb_type"] or "unknown"
            pnl_by_type[key] = pnl_by_type.get(key, 0.0) + trade_realized_pnl(tr)

        by_strategy = []
        for row in strategies:
            resolved = row["resolved"] or 0
            wins = row["wins"] or 0
            win_rate = (wins / resolved) if resolved else None
            avg_edge = row["avg_predicted_edge"] or 0.0
            calibration_gap = (
                (win_rate - 0.5) - avg_edge if win_rate is not None else None
            )
            by_strategy.append({
                "arb_type": row["arb_type"],
                "total": row["total"],
                "resolved": resolved,
                "wins": wins,
                "win_rate": win_rate,
                "avg_predicted_edge": avg_edge,
                "avg_edge_wins": row["avg_edge_wins"],
                "avg_edge_losses": row["avg_edge_losses"],
                "calibration_gap": calibration_gap,
                "est_pnl": round(pnl_by_type.get(row["arb_type"], 0.0), 2),
            })

        edge_buckets = [
            {
                "arb_type": r["arb_type"],
                "bucket": r["edge_bucket"],
                "count": r["n"],
                "wins": r["wins"],
                "win_rate": (r["wins"] / r["n"]) if r["n"] else 0.0,
                "avg_edge": r["avg_edge"] or 0.0,
            }
            for r in buckets
        ]

        return {
            "strategies": by_strategy,
            "edge_buckets": edge_buckets,
            "summary": dict(summary_row) if summary_row else {},
        }

    def get_pnl_summary(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trades
                WHERE status IN ('placed', 'simulated')
                """,
            ).fetchall()

        if not rows:
            return {}

        total_trades = len(rows)
        total_contracts = sum(float(r["contracts"] or 0) for r in rows)
        edges = [float(r["net_profit_pct"] or 0) for r in rows]
        realized = sum(trade_realized_pnl(r) for r in rows)
        theoretical = sum(trade_theoretical_pnl(r) for r in rows)
        pending = sum(1 for r in rows if r["outcome"] == "pending")
        resolved = sum(1 for r in rows if r["outcome"] in ("won", "lost"))

        return {
            "total_trades": total_trades,
            "total_contracts": total_contracts,
            "avg_net_profit_pct": sum(edges) / len(edges) if edges else 0.0,
            "estimated_pnl_usd": round(realized, 2),
            "theoretical_pnl_usd": round(theoretical, 2),
            "pending_trades": pending,
            "resolved_trades": resolved,
        }
