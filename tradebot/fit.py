"""Can this account afford the bot's trades? (printed by ``check`` and at live start-up)

The bot never risks more than ``risk_per_trade_pct`` of equity on a trade. Every broker has a
minimum trade size (0.01 lot), and on gold 0.01 lot of a standard account is 1 ounce, so a $5
stop already costs $5. On a small account that is far more than a few percent, so the bot skips
the trade. This module turns that into numbers for your account, and says what to change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import pandas as pd

from .config import TradeStyle
from .features import atr as atr_series
from .types import SymbolInfo


@dataclass
class StyleFit:
    style: str
    stop: float  # typical stop distance now, in price units
    min_lot: float
    min_risk: float  # money lost by the smallest possible trade if the stop is hit
    min_risk_pct: float  # ... as % of equity
    fits: bool
    digits: int = 2
    small_mode: bool = False  # fits only thanks to small-account mode


def latest_atr(bars_by_tf: Mapping[str, pd.DataFrame], period: int = 14) -> Dict[str, float]:
    out = {}
    for tf, bars in bars_by_tf.items():
        if bars is None or len(bars) < period + 2:
            continue
        a = atr_series(bars["high"].astype(float), bars["low"].astype(float), bars["close"].astype(float), period)
        val = float(a.iloc[-1])
        if math.isfinite(val) and val > 0:
            out[tf] = val
    return out


def assess_fit(info: SymbolInfo, styles: List[TradeStyle], atr_by_tf: Mapping[str, float],
               equity: float, risk_pct: float, min_lot_cap_pct: Optional[float] = None) -> List[StyleFit]:
    budget = equity * risk_pct / 100.0
    cap = equity * min_lot_cap_pct / 100.0 if min_lot_cap_pct else 0.0
    rows = []
    for s in styles:
        a = atr_by_tf.get(s.timeframe)
        if a is None:
            continue
        stop = s.sl_atr * a
        min_risk = stop * info.value_per_price_unit * info.volume_min
        pct = 100.0 * min_risk / equity if equity > 0 else float("inf")
        normal = min_risk <= budget + 1e-9
        small = not normal and min_risk <= cap + 1e-9
        rows.append(StyleFit(s.name, stop, info.volume_min, min_risk, pct, normal or small, info.digits, small))
    return rows


def fit_report(rows: List[StyleFit], equity: float, currency: str, risk_pct: float,
               min_lot_margin: Optional[float] = None, free_margin: Optional[float] = None) -> str:
    budget = equity * risk_pct / 100.0
    lines = [f"Account fit: equity {equity:,.2f} {currency}, risk per trade {risk_pct:g}% = "
             f"{budget:,.2f} {currency}"]
    for r in rows:
        lines.append(f"  {r.style:<24} stop now ~{r.stop:,.{r.digits}f} -> smallest trade ({r.min_lot:g} lot) risks "
                     f"{r.min_risk:,.2f} {currency} = {r.min_risk_pct:.1f}% of equity  "
                     f"[{'OK (small-account mode)' if r.small_mode else 'OK' if r.fits else 'TOO BIG - will be skipped'}]")
    margin_ok = True
    if min_lot_margin is not None and free_margin is not None:
        margin_ok = min_lot_margin <= free_margin * 0.9
        lines.append(f"  margin for the smallest trade: {min_lot_margin:,.2f} {currency} "
                     f"(free margin {free_margin:,.2f})" + ("" if margin_ok else "  [NOT ENOUGH]"))
    if any(r.small_mode for r in rows) and margin_ok:
        worst = max(r.min_risk_pct for r in rows if r.fits)
        lines.append(f"  -> SMALL-ACCOUNT MODE: the bot trades the minimum lot. One losing trade can cost up to "
                     f"{worst:.0f}% of the account.")
    if rows and all(r.fits for r in rows) and margin_ok:
        lines.append("  -> every trade style fits this account.")
        return "\n".join(lines)
    if rows and any(r.fits for r in rows) and margin_ok:
        lines.append("  -> only the styles marked OK can be traded; the others are skipped.")
        return "\n".join(lines)
    cheapest = min(rows, key=lambda r: r.min_risk) if rows else None
    lines.append("  -> THE BOT WILL NOT OPEN TRADES ON THIS ACCOUNT: even the smallest trade is too big for it.")
    if cheapest is not None:
        need = cheapest.min_risk / (risk_pct / 100.0)
        need5 = cheapest.min_risk / 0.05
        lines.append("     Options:")
        lines.append("       (1) use a CENT account: balance shown in US cents and trades 100x smaller "
                     "($20 = 2,000 USC), which usually fits;")
        lines.append(f"       (2) have about {need:,.0f} {currency} of equity at your {risk_pct:g}% risk;")
        lines.append(f"       (3) raise risk.risk_per_trade_pct (5% max): needs about {need5:,.0f} {currency}, "
                     "but then each losing trade costs 5% of the account;")
        lines.append(f"       (4) small-account mode: allow the minimum lot even though one losing trade costs about "
                     f"{cheapest.min_risk_pct:.0f}% or more of the account (run 1-INSTALL-AND-SETUP.bat again).")
    if not margin_ok:
        lines.append("     The broker would also reject the order for lack of margin.")
    return "\n".join(lines)
