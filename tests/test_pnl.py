"""Unit tests for paper/live P&L accounting."""

import sqlite3

from storage import trade_realized_pnl, trade_theoretical_pnl


def _row(**kwargs) -> sqlite3.Row:
    defaults = {
        "size_usd": 100.0,
        "contracts": 100,
        "side": "yes",
        "yes_price": 0.50,
        "no_price": 0.50,
        "net_profit_pct": 0.08,
        "outcome": "pending",
    }
    defaults.update(kwargs)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE t (size_usd REAL, contracts INT, side TEXT, "
        "yes_price REAL, no_price REAL, net_profit_pct REAL, outcome TEXT)"
    )
    conn.execute(
        "INSERT INTO t VALUES (?,?,?,?,?,?,?)",
        tuple(defaults[k] for k in (
            "size_usd", "contracts", "side", "yes_price",
            "no_price", "net_profit_pct", "outcome",
        )),
    )
    return conn.execute("SELECT * FROM t").fetchone()


class TestTradeRealizedPnl:
    def test_pending_is_zero(self):
        assert trade_realized_pnl(_row(outcome="pending")) == 0.0

    def test_yes_win(self):
        # $100 at 0.50 → profit $100
        pnl = trade_realized_pnl(_row(outcome="won", side="yes", yes_price=0.50))
        assert abs(pnl - 100.0) < 0.01

    def test_yes_loss(self):
        pnl = trade_realized_pnl(_row(outcome="lost", side="yes"))
        assert pnl == -100.0

    def test_arb_win_uses_edge(self):
        pnl = trade_realized_pnl(_row(
            outcome="won", side="both", net_profit_pct=0.05, size_usd=200.0
        ))
        assert abs(pnl - 10.0) < 0.01


class TestTradeTheoreticalPnl:
    def test_pending_uses_edge_times_stake(self):
        pnl = trade_theoretical_pnl(_row(outcome="pending", net_profit_pct=0.10))
        assert abs(pnl - 10.0) < 0.01

    def test_resolved_is_zero(self):
        assert trade_theoretical_pnl(_row(outcome="won")) == 0.0