"""Is the demo record good enough to risk real money? (``python -m tradebot readiness``)

The brain itself does not learn faster on demo or live - it learns from market prices, which are
the same. What a demo period proves is that the *whole system* (signals, costs, fills, your broker,
your PC/VPS) makes money. That needs enough trades to tell skill from luck, which takes weeks, not
a day. This report checks the bot's own demo trades against the thresholds in ``readiness:``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import BotConfig


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Readiness:
    ready: bool
    checks: List[Check]
    trades: int
    days: float

    def report(self) -> str:
        lines = ["Live-trading readiness (based on this bot's demo trades):"]
        for c in self.checks:
            lines.append(f"  [{'OK' if c.ok else '  '}] {c.name}: {c.detail}")
        if self.ready:
            lines.append("READY by your criteria. Start live with a small risk_per_trade_pct and keep watching.")
        else:
            lines.append("NOT READY yet - keep it on demo. Going live now would be betting on luck.")
        return "\n".join(lines)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def assess(cfg: BotConfig, log_dir: Optional[str] = None, account: Optional[int] = None,
           symbol: Optional[str] = None) -> Readiness:
    r = cfg.readiness
    d = Path(log_dir or cfg.paths.log_dir)
    closed = _read(d / "closed.csv")
    decisions = _read(d / "decisions.csv")
    if account is not None and "account" in closed:
        closed = closed[closed["account"].astype(str) == str(account)]
    if symbol is not None:
        if "symbol" in closed:
            closed = closed[closed["symbol"] == symbol]
        if "symbol" in decisions:
            decisions = decisions[decisions["symbol"] == symbol]

    days = 0.0
    if len(decisions):
        t = pd.to_datetime(decisions["time"])
        days = (t.max() - t.min()).total_seconds() / 86400.0
    n = len(closed)
    profit = closed["profit"].astype(float).to_numpy() if n else np.array([])
    rs = closed["r_multiple"].astype(float).to_numpy() if n else np.array([])
    gains, losses = profit[profit > 0].sum(), -profit[profit < 0].sum()
    pf = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)
    avg_r = float(np.mean(rs)) if n else 0.0
    cum_r = np.cumsum(rs) if n else np.array([0.0])
    dd_r = float(np.max(np.maximum.accumulate(np.r_[0.0, cum_r]) - np.r_[0.0, cum_r]))
    dd_pct = dd_r * cfg.risk.risk_per_trade_pct  # 1R = risk_per_trade_pct of equity

    checks = [
        Check("time on demo", days >= r.min_days, f"{days:.1f} days (need {r.min_days})"),
        Check("number of trades", n >= r.min_trades, f"{n} (need {r.min_trades})"),
        Check("profit factor", n > 0 and pf >= r.min_profit_factor,
              f"{pf:.2f} (need {r.min_profit_factor})" if n else "no trades yet"),
        Check("average result", n > 0 and avg_r >= r.min_avg_r,
              f"{avg_r:+.2f}R per trade (need >= {r.min_avg_r:+.2f}R)" if n else "no trades yet"),
        Check("worst drawdown", dd_pct <= r.max_drawdown_pct,
              f"{dd_r:.1f}R ~ {dd_pct:.1f}% of equity (limit {r.max_drawdown_pct}%)"),
    ]
    return Readiness(all(c.ok for c in checks), checks, n, days)
