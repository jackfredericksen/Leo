"""
Rich terminal UI for Leo — tables and live dashboard panel.
"""

from datetime import datetime, timezone

from rich import box
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from bot_state import state
from config import config
from position_manager import PositionManager
from storage import Storage
from strategies.correlated import CorrelatedOpportunity
from strategies.signal_arb import AggregatedSignal

def _mode() -> str:
    return (
        "[bold red]LIVE[/]"
        if not config.dry_run
        else "[bold yellow]DRY RUN[/]"
    )


def _hrs(close_time: datetime) -> str:
    h = (close_time - datetime.now(timezone.utc)).total_seconds() / 3600
    return f"{h:.1f}h"


def _arb_table() -> Table:
    t = Table(
        "Market", "Sum", "Net%", "Closes",
        box=box.SIMPLE_HEAD, title="[cyan]Overround Arb[/]",
        title_style="bold", min_width=50,
    )
    for o in state.arb_opps[:5]:
        q = o.question[:26] + ("…" if len(o.question) > 26 else "")
        t.add_row(
            q,
            f"[green]{o.yes_ask + o.no_ask:.4f}[/]",
            f"[bold green]{o.net_profit_pct:.1%}[/]",
            _hrs(o.close_time),
        )
    if not state.arb_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _crypto_table() -> Table:
    t = Table(
        "Market", "Spot→Model", "Edge", "Side",
        box=box.SIMPLE_HEAD, title="[yellow]Crypto Signal[/]",
        title_style="bold", min_width=54,
    )
    for o in state.crypto_opps[:5]:
        clr = "green" if o.edge > 0 else "red"
        q = o.question[:24] + ("…" if len(o.question) > 24 else "")
        t.add_row(
            q,
            f"{o.market_prob:.0%}→{o.model_prob:.0%}",
            f"[{clr}]{o.edge:+.1%}[/]",
            o.recommended_side,
        )
    if not state.crypto_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _cross_table() -> Table:
    t = Table(
        "Market", "Buy@", "Ref@", "Net%",
        box=box.SIMPLE_HEAD, title="[blue]Cross-Platform (Kalshi→Poly)[/]",
        title_style="bold", min_width=54,
    )
    for o in state.cross_opps[:5]:
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        t.add_row(
            q,
            f"PM {o.buy_price:.3f}",
            f"KL {o.sell_price:.3f}",
            f"[bold blue]{o.net_profit_pct:.1%}[/]",
        )
    if not state.cross_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _range_table() -> Table:
    t = Table(
        "Market", "Model", "Mkt", "Edge",
        box=box.SIMPLE_HEAD, title="[magenta]Range/Logic Arb[/]",
        title_style="bold", min_width=50,
    )
    for o in (state.range_opps + state.corr_opps)[:5]:
        if isinstance(o, AggregatedSignal):
            clr = "green" if o.edge > 0 else "red"
            q = o.question[:24] + ("…" if len(o.question) > 24 else "")
            t.add_row(q, f"{o.model_prob:.0%}", f"{o.market_prob:.0%}", f"[{clr}]{o.edge:+.1%}[/]")
        else:
            q = o.question_mispriced[:24] + ("…" if len(o.question_mispriced) > 24 else "")
            t.add_row(q, o.relation.value, f"{o.edge:.1%}", o.recommended_side)
    if not state.range_opps and not state.corr_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _fade_table() -> Table:
    t = Table(
        "Market", "Dir", "Edge",
        box=box.SIMPLE_HEAD, title="[orange3]News Fade[/]",
        title_style="bold", min_width=46,
    )
    for o in state.fade_opps[:5]:
        clr = "green" if o.edge > 0 else "red"
        q = o.question[:30] + ("…" if len(o.question) > 30 else "")
        t.add_row(q, o.recommended_side, f"[{clr}]{o.edge:+.1%}[/]")
    if not state.fade_opps:
        t.add_row("[dim]none[/]", "", "")
    return t


def _forecast_table() -> Table:
    t = Table(
        "Market", "Src", "Model", "Mkt", "Edge",
        box=box.SIMPLE_HEAD, title="[bright_cyan]Forecast Aggregator[/]",
        title_style="bold", min_width=58,
    )
    for o in state.forecast_opps[:5]:
        clr = "green" if o.edge > 0 else "red"
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        t.add_row(q, (o.source or "")[:10], f"{o.model_prob:.0%}", f"{o.market_prob:.0%}", f"[{clr}]{o.edge:+.1%}[/]")
    if not state.forecast_opps:
        t.add_row("[dim]none[/]", "", "", "", "")
    return t


def _llm_table() -> Table:
    t = Table(
        "Market", "Model", "Mkt", "Edge", "Conf",
        box=box.SIMPLE_HEAD, title="[bright_green]LLM Analysis[/]",
        title_style="bold", min_width=58,
    )
    for o in state.llm_opps[:5]:
        clr = "green" if o.edge > 0 else "red"
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        t.add_row(q, f"{o.model_prob:.0%}", f"{o.market_prob:.0%}", f"[{clr}]{o.edge:+.1%}[/]", f"{o.confidence:.0%}" if o.confidence else "—")
    if not state.llm_opps:
        t.add_row("[dim]none[/]", "", "", "", "")
    return t


def _weather_table() -> Table:
    t = Table(
        "Market", "Forecast", "Mkt", "Edge",
        box=box.SIMPLE_HEAD, title="[bright_blue]Weather Signal[/]",
        title_style="bold", min_width=54,
    )
    for o in state.weather_opps[:5]:
        clr = "green" if o.edge > 0 else "red"
        q = o.question[:24] + ("…" if len(o.question) > 24 else "")
        t.add_row(q, f"{o.model_prob:.0%}", f"{o.market_prob:.0%}", f"[{clr}]{o.edge:+.1%}[/]")
    if not state.weather_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _fav_table() -> Table:
    t = Table(
        "Market", "Mkt", "Fair", "Edge",
        box=box.SIMPLE_HEAD, title="[bold red]Favorite Short[/]",
        title_style="bold", min_width=52,
    )
    for o in state.fav_opps[:5]:
        q = o.question[:24] + ("…" if len(o.question) > 24 else "")
        t.add_row(q, f"{o.market_prob:.0%}", f"{o.model_prob:.0%}", f"[green]{o.edge:+.1%}[/]")
    if not state.fav_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _squeeze_table() -> Table:
    t = Table(
        "Market", "Price", "Target", "Edge",
        box=box.SIMPLE_HEAD, title="[bold yellow]Oracle Squeeze[/]",
        title_style="bold", min_width=52,
    )
    for o in state.squeeze_opps[:5]:
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        tgt = "1.00" if o.recommended_side == "yes" else "0.00"
        t.add_row(q, f"{o.market_prob:.2f}", tgt, f"[bold green]{o.edge:+.1%}[/]")
    if not state.squeeze_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _semarg_table() -> Table:
    t = Table(
        "Market", "Poly", "Ext", "Gap",
        box=box.SIMPLE_HEAD, title="[bold blue]Semantic Arb[/]",
        title_style="bold", min_width=52,
    )
    for o in state.semarg_opps[:5]:
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        t.add_row(q, f"{o.market_prob:.0%}", f"{o.model_prob:.0%}", f"[bold blue]{o.edge:+.1%}[/]")
    if not state.semarg_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _ofi_table() -> Table:
    t = Table(
        "Market", "OFI", "Regime", "Edge",
        box=box.SIMPLE_HEAD, title="[bold magenta]OFI Momentum[/]",
        title_style="bold", min_width=52,
    )
    for o in state.ofi_opps[:5]:
        q = o.question[:22] + ("…" if len(o.question) > 22 else "")
        regime = "follow" if "follow" in o.source else "fade"
        clr = "green" if o.edge > 0 else "red"
        t.add_row(q, o.recommended_side, regime, f"[{clr}]{o.edge:+.1%}[/]")
    if not state.ofi_opps:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _mm_table() -> Table:
    t = Table(
        "Market", "YES@", "NO@", "Mid",
        box=box.SIMPLE_HEAD, title=f"[bold cyan]Market Maker ({state.mm_active} active)[/]",
        title_style="bold", min_width=52,
    )
    for q in state.mm_quotes[:5]:
        mkt = q.get("question", "")[:22]
        yes_b = f"{q['yes_bid']:.2f}" if q.get("yes_bid") else "—"
        no_b  = f"{q['no_bid']:.2f}"  if q.get("no_bid")  else "—"
        mid   = f"{q.get('fair_value', 0):.2f}"
        t.add_row(mkt, yes_b, no_b, mid)
    if not state.mm_quotes:
        t.add_row("[dim]none[/]", "", "", "")
    return t


def _stats_table(storage: Storage, pos_manager: PositionManager) -> Table:
    trade_stats = storage.get_pnl_summary()
    pos = pos_manager.summary()
    t = Table(box=box.SIMPLE, show_header=False, min_width=26)
    t.add_column(style="dim", no_wrap=True)
    t.add_column(justify="right")
    t.add_row("Mode", _mode())
    t.add_row("Balance (USDC)", f"${pos.get('balance_usd', 0):.2f}")
    t.add_row("Positions", str(pos.get("open_positions", 0)))
    t.add_row("Unrealized", f"${pos.get('unrealized_pnl', 0):.2f}")
    t.add_row("Realized", f"${pos.get('realized_pnl', 0):.2f}")
    t.add_row("─" * 10, "─" * 7)
    t.add_row("Markets", str(len(state.markets)))
    t.add_row("Arb scans", str(state.arb_scans))
    t.add_row("Corr scans", str(state.corr_scans))
    t.add_row("Crypto scans", str(state.crypto_scans))
    t.add_row("Cross scans", str(state.cross_scans))
    t.add_row("Range scans", str(state.range_scans))
    t.add_row("Fade scans", str(state.fade_scans))
    t.add_row("Forecast scans", str(state.forecast_scans))
    t.add_row("LLM scans", str(state.llm_scans))
    t.add_row("Weather scans", str(state.weather_scans))
    t.add_row("Fav scans", str(state.fav_scans))
    t.add_row("Squeeze scans", str(state.squeeze_scans))
    t.add_row("Sem arb scans", str(state.semarg_scans))
    t.add_row("OFI scans", str(state.ofi_scans))
    t.add_row("BTC 5min scans", str(state.btc5min_scans))
    t.add_row("MM active", str(state.mm_active))
    t.add_row("─" * 10, "─" * 7)
    t.add_row("Trades", str(trade_stats.get("total_trades", 0)))
    t.add_row("Est. P&L", f"${trade_stats.get('estimated_pnl_usd') or 0:.2f}")
    return t


def build_ui(storage: Storage, pos_manager: PositionManager) -> Panel:
    row1 = Columns([_arb_table(), _crypto_table()], equal=True)
    row2 = Columns([_cross_table(), _range_table()], equal=True)
    row3 = Columns([_fade_table(), _forecast_table()], equal=True)
    row4 = Columns([_llm_table(), _weather_table()], equal=True)
    row5 = Columns([_fav_table(), _squeeze_table()], equal=True)
    row6 = Columns([_semarg_table(), _ofi_table()], equal=True)
    row7 = Columns([_mm_table(), _stats_table(storage, pos_manager)], equal=True)
    return Panel(
        Group(row1, row2, row3, row4, row5, row6, row7),
        title=(
            "[bold]Leo — Polymarket Trading Bot[/bold]  "
            + datetime.now().strftime("%H:%M:%S")
        ),
        border_style="cyan",
    )
