"""Outcome labelling ("triple barrier"): what would have happened to a trade opened after bar t.

For every bar ``t`` and every *arm* (direction x stop-loss/take-profit profile) we simulate a trade
entered at the open of bar ``t+1`` with SL/TP placed at multiples of ATR(t), held for at most
``horizon`` bars. The label says whether the take-profit was hit first, and ``r`` is the realised
result in R (multiples of the initial risk), net of spread and assumed costs.

These labels are what the model learns from, so it learns exactly the decision it has to make:
"if I open this trade now with these exits, is it likely to pay?".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import ExitProfile
from .types import Side


@dataclass(frozen=True)
class Arm:
    side: Side
    sl_atr: float
    tp_atr: float

    @property
    def key(self) -> str:
        return f"{self.side.label}:sl{self.sl_atr:g}_tp{self.tp_atr:g}"


def make_arms(profiles: List[ExitProfile]) -> List[Arm]:
    return [Arm(side, p.sl_atr, p.tp_atr) for p in profiles for side in (Side.BUY, Side.SELL)]


@dataclass
class ArmLabels:
    win: np.ndarray  # 1.0 if the take-profit was hit first, else 0.0; NaN if not resolvable yet
    loss: np.ndarray  # 1.0 if the stop-loss was hit first, else 0.0 (neither = time-out)
    r: np.ndarray  # realised R multiple, NaN if not resolvable yet


def spread_prices(bars: pd.DataFrame, point: float, default_spread_points: float) -> np.ndarray:
    if "spread" in bars.columns:
        spr = bars["spread"].to_numpy(dtype=np.float64)
        spr = np.where(np.isfinite(spr) & (spr > 0), spr, default_spread_points)
    else:
        spr = np.full(len(bars), float(default_spread_points))
    return spr * point


def label_arm(
    bars: pd.DataFrame,
    atr: np.ndarray,
    arm: Arm,
    horizon: int,
    point: float,
    default_spread_points: float = 10.0,
    extra_cost: float = 0.0,
) -> ArmLabels:
    """Vectorised triple-barrier outcome for one arm. Bars are bid prices (MT5 convention)."""
    n = len(bars)
    win = np.full(n, np.nan)
    loss = np.full(n, np.nan)
    r = np.full(n, np.nan)
    m = n - horizon  # rows t = 0..m-1 have all bars t+1..t+horizon available
    if m <= 0:
        return ArmLabels(win, loss, r)

    o = bars["open"].to_numpy(dtype=np.float64)
    h = bars["high"].to_numpy(dtype=np.float64)
    lo = bars["low"].to_numpy(dtype=np.float64)
    c = bars["close"].to_numpy(dtype=np.float64)
    spr = spread_prices(bars, point, default_spread_points)

    t = np.arange(m)
    a = atr[:m]
    sl_dist = arm.sl_atr * a
    tp_dist = arm.tp_atr * a
    # windows of bars t+1 .. t+horizon -> shape (m, horizon)
    wo = sliding_window_view(o[1:], horizon)[:m]
    wh = sliding_window_view(h[1:], horizon)[:m]
    wl = sliding_window_view(lo[1:], horizon)[:m]
    ws = sliding_window_view(spr[1:], horizon)[:m]

    if arm.side is Side.BUY:
        entry = o[t + 1] + spr[t + 1] + extra_cost  # buy at ask
        sl_px = entry - sl_dist
        tp_px = entry + tp_dist
        hit_sl = wl <= sl_px[:, None]
        hit_tp = wh >= tp_px[:, None]
        sl_exit = np.minimum(wo, sl_px[:, None])  # gap through the stop fills at the open
        tp_exit = np.maximum(wo, tp_px[:, None])
        timeout_exit = c[t + horizon]
        sign = 1.0
    else:
        entry = o[t + 1] - extra_cost  # sell at bid
        sl_px = entry + sl_dist
        tp_px = entry - tp_dist
        hit_sl = (wh + ws) >= sl_px[:, None]  # shorts are closed at the ask
        hit_tp = (wl + ws) <= tp_px[:, None]
        sl_exit = np.maximum(wo + ws, sl_px[:, None])
        tp_exit = np.minimum(wo + ws, tp_px[:, None])
        timeout_exit = c[t + horizon] + spr[t + horizon]
        sign = -1.0

    any_hit = hit_sl | hit_tp
    has_hit = any_hit.any(axis=1)
    first = np.argmax(any_hit, axis=1)
    rows = np.arange(m)
    # If SL and TP are both inside the same bar we cannot know the order: assume the worst (SL).
    sl_first = hit_sl[rows, first]
    exit_px = np.where(
        has_hit,
        np.where(sl_first, sl_exit[rows, first], tp_exit[rows, first]),
        timeout_exit,
    )
    won = has_hit & ~sl_first
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = sign * (exit_px - entry) / sl_dist
    ok = np.isfinite(rr) & np.isfinite(a) & (a > 0)
    win[:m] = np.where(ok, won.astype(np.float64), np.nan)
    loss[:m] = np.where(ok, (has_hit & sl_first).astype(np.float64), np.nan)
    r[:m] = np.where(ok, rr, np.nan)
    return ArmLabels(win, loss, r)


def label_all(
    bars: pd.DataFrame,
    atr: np.ndarray,
    arms: List[Arm],
    horizon: int,
    point: float,
    default_spread_points: float = 10.0,
    extra_cost: float = 0.0,
) -> Dict[str, ArmLabels]:
    return {
        arm.key: label_arm(bars, atr, arm, horizon, point, default_spread_points, extra_cost)
        for arm in arms
    }
