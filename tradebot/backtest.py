"""Walk-forward backtest: replays history bar by bar through the *same* BotCore used live.

The model starts untrained, learns from the first ``warmup_bars`` without trading, and then keeps
learning while it trades - so the result shows how the self-adapting bot would actually have done,
with no peeking at the future.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .brokers.sim import SimBroker
from .config import BotConfig
from .core import BotCore
from .journal import Journal
from .types import ClosedTrade, SymbolInfo

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    symbol: str
    trades: List[ClosedTrade]
    equity: pd.DataFrame  # time, equity
    stats: dict = field(default_factory=dict)


def compute_stats(trades: List[ClosedTrade], equity: pd.DataFrame, initial: float, r_by_ticket: dict) -> dict:
    profits = np.array([t.profit for t in trades], dtype=float)
    wins = profits[profits > 0]
    losses = profits[profits <= 0]
    eq = equity["equity"].to_numpy(dtype=float) if len(equity) else np.array([initial])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    daily = equity.set_index("time")["equity"].resample("1D").last().dropna() if len(equity) else pd.Series(dtype=float)
    rets = daily.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if len(rets) > 2 and rets.std() > 0 else float("nan")
    rs = np.array([r_by_ticket.get(t.position_id, np.nan) for t in trades], dtype=float)
    streak = worst = 0
    for p in profits:
        streak = streak + 1 if p <= 0 else 0
        worst = max(worst, streak)
    final = float(eq[-1])
    return {
        "trades": int(len(trades)),
        "win_rate_pct": round(100.0 * len(wins) / len(trades), 2) if len(trades) else 0.0,
        "net_profit": round(float(profits.sum()), 2),
        "return_pct": round(100.0 * (final / initial - 1.0), 2),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 3) if len(losses) and losses.sum() < 0 else None,
        "avg_trade": round(float(profits.mean()), 2) if len(trades) else 0.0,
        "avg_r": round(float(np.nanmean(rs)), 3) if np.isfinite(rs).any() else None,
        "max_drawdown_pct": round(max(0.0, float(-dd.min() * 100.0)), 2),
        "sharpe_daily": round(sharpe, 2) if math.isfinite(sharpe) else None,
        "longest_losing_streak": int(worst),
        "exits": {k: int(v) for k, v in pd.Series([t.reason for t in trades]).value_counts().items()} if trades else {},
        "final_equity": round(final, 2),
    }


def run_backtest(cfg: BotConfig, bars: pd.DataFrame, symbol: str, info: SymbolInfo,
                 context: Optional[Dict[str, pd.DataFrame]] = None,
                 out_dir: Optional[str] = None, warmup_bars: Optional[int] = None,
                 log_decisions: bool = True, progress: bool = True) -> BacktestResult:
    """``bars``: the trading timeframe; ``context``: bars of every ``model.context_timeframes`` chart."""
    bars = bars.reset_index(drop=True)
    n = len(bars)
    warmup = cfg.backtest.warmup_bars if warmup_bars is None else warmup_bars
    warmup = max(warmup, 300)
    if n < warmup + 50:
        raise ValueError(f"Need more than {warmup + 50} bars, got {n}")

    sim = SimBroker({symbol: bars}, {symbol: info}, cfg.backtest.initial_balance,
                    cfg.trading.commission_per_lot, cfg.backtest.default_spread_points)
    journal = Journal(out_dir, log_decisions=log_decisions, auto_flush=False)
    core = BotCore(cfg, sim, journal=journal, state_dir=None, dry_run=False)
    core.infos[symbol] = info

    t0 = time.time()
    # Features/labels are computed once for speed; both are causal (verified by the tests), and
    # BotCore only ever learns labels whose outcome window ended before the current bar.
    prep = core.prepare(symbol, bars, context)
    core.learn_range(symbol, prep, warmup - 1, progress=progress)

    equity_rows = []
    r_by_ticket = {}
    last_report = time.time()
    sim.set_index(symbol, warmup)
    for i in range(warmup, n - 1):
        core.on_bar(symbol, prep, i)
        sim.advance(symbol)
        acct = sim.account()
        equity_rows.append((bars["time"].iat[i + 1], acct.equity))
        for tid, meta in core.open_risk.items():
            r_by_ticket.setdefault(tid, meta["risk_money"])
        if progress and time.time() - last_report > 5:
            last_report = time.time()
            log.info("backtest %s: %5.1f%%  equity %.2f  trades %d", symbol,
                     100.0 * (i - warmup) / (n - 1 - warmup), acct.equity, len(sim.all_closed))
    for pos in sim.positions(symbol):
        sim.close_position(pos, comment="end-of-test")
    core.sync_closed(bars["time"].iat[-1] + core.tf)
    journal.close()

    trades = sim.all_closed
    r_mult = {t.position_id: t.profit / r_by_ticket[t.position_id]
              for t in trades if r_by_ticket.get(t.position_id, 0) > 0}
    equity = pd.DataFrame(equity_rows, columns=["time", "equity"])
    stats = compute_stats(trades, equity, cfg.backtest.initial_balance, r_mult)
    stats["bars_traded"] = n - 1 - warmup
    stats["period"] = f"{bars['time'].iat[warmup]} -> {bars['time'].iat[-1]}"
    stats["seconds"] = round(time.time() - t0, 1)
    brain = core.brains[symbol]
    stats["model"] = brain.summary()
    stats["outlooks"] = brain.outlook_summary()
    stats["learning_history"] = brain.history[-5:]
    if out_dir:
        out = Path(out_dir)
        equity.to_csv(out / "equity.csv", index=False)
        (out / "summary.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
        _plot(equity, out / "equity.png", symbol)
    return BacktestResult(symbol, trades, equity, stats)


def _plot(equity: pd.DataFrame, path: Path, symbol: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if equity.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity["time"], equity["equity"], lw=1.2)
    ax.set_title(f"{symbol} walk-forward backtest equity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
