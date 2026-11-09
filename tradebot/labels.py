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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import TradeStyle
from .types import Side


@dataclass(frozen=True)
class Arm:
    """One candidate trade: direction x style (stop/target in ATR of ``tf``, max holding time)."""

    side: Side
    tf: str
    sl_atr: float
    tp_atr: float
    horizon: int  # max holding time in trading-timeframe bars
    trail_atr: Optional[float] = None  # trailing-stop distance in ATR (None = fixed stop)
    trail_start_r: float = 1.0  # trailing starts once the trade is this many R in profit

    @property
    def key(self) -> str:
        trail = f"_tr{self.trail_atr:g}" if self.trail_atr else ""
        return f"{self.side.label}:{self.tf}:sl{self.sl_atr:g}_tp{self.tp_atr:g}{trail}"


def make_arms(styles: List[TradeStyle]) -> List[Arm]:
    return [Arm(side, s.timeframe, s.sl_atr, s.tp_atr, s.max_hold_bars, s.trail_atr, s.trail_start_r)
            for s in styles for side in (Side.BUY, Side.SELL)]


@dataclass
class ArmLabels:
    # 1.0 if the trade ended in profit at the take-profit (or, for trailing styles, at a stop that
    # had been trailed into profit), else 0.0; NaN if not resolvable yet
    win: np.ndarray
    loss: np.ndarray  # 1.0 if a stop-loss closed it without profit, else 0.0 (neither = time-out)
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
    """Vectorised triple-barrier outcome for one arm. Bars are bid prices (MT5 convention).
    ``atr`` is the ATR (price units) that sizes this arm's stop and target at every bar."""
    n = len(bars)
    win = np.full(n, np.nan)
    loss = np.full(n, np.nan)
    r = np.full(n, np.nan)
    m = n - horizon  # rows t = 0..m-1 have all bars t+1..t+horizon available
    if m <= 0:
        return ArmLabels(win, loss, r)
    if arm.trail_atr:
        return _label_trailing(bars, atr, arm, horizon, point, default_spread_points, extra_cost, m)

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


def _label_trailing(bars, atr, arm: Arm, horizon: int, point: float, default_spread_points: float,
                    extra_cost: float, m: int) -> ArmLabels:
    """Outcome of a trailing-stop trade, replayed candle by candle exactly as the live bot manages
    it: inside each candle the stop set at the previous close is checked first (worst case), then
    the target; at the candle's close the stop is moved to ``trail`` behind the best price so far
    once the trade is ``trail_start_r`` R in profit, never backwards."""
    n = len(bars)
    o = bars["open"].to_numpy(dtype=np.float64)
    h = bars["high"].to_numpy(dtype=np.float64)
    lo = bars["low"].to_numpy(dtype=np.float64)
    c = bars["close"].to_numpy(dtype=np.float64)
    spr = spread_prices(bars, point, default_spread_points)
    t = np.arange(m)
    a = atr[:m]
    sl_dist = arm.sl_atr * a
    tp_dist = arm.tp_atr * a
    trail = arm.trail_atr * a
    start = arm.trail_start_r * sl_dist
    buy = arm.side is Side.BUY
    if buy:
        entry = o[t + 1] + spr[t + 1] + extra_cost
        sl, tp = entry - sl_dist, entry + tp_dist
        best = np.full(m, -np.inf)
    else:
        entry = o[t + 1] - extra_cost
        sl, tp = entry + sl_dist, entry - tp_dist
        best = np.full(m, np.inf)
    alive = np.ones(m, dtype=bool)
    exit_px = np.full(m, np.nan)
    kind = np.zeros(m, dtype=np.int8)  # 0 timeout, 1 take-profit, 2 stop
    with np.errstate(invalid="ignore"):
        for k in range(1, horizon + 1):
            j = t + k
            if buy:
                bo, bh, bl = o[j], h[j], lo[j]
                hit_sl = alive & (bl <= sl)
                hit_tp = alive & ~hit_sl & (bh >= tp)
                exit_px = np.where(hit_sl, np.minimum(bo, sl), exit_px)
                exit_px = np.where(hit_tp, np.maximum(bo, tp), exit_px)
            else:
                s = spr[j]
                bo, bh, bl = o[j] + s, h[j] + s, lo[j] + s  # shorts are closed at the ask
                hit_sl = alive & (bh >= sl)
                hit_tp = alive & ~hit_sl & (bl <= tp)
                exit_px = np.where(hit_sl, np.maximum(bo, sl), exit_px)
                exit_px = np.where(hit_tp, np.minimum(bo, tp), exit_px)
            kind[hit_sl] = 2
            kind[hit_tp] = 1
            alive &= ~(hit_sl | hit_tp)
            # candle closed: move the stop (applies from the next candle)
            if buy:
                best = np.maximum(best, bh)
                active = alive & (best - entry >= start)
                sl = np.where(active, np.maximum(sl, best - trail), sl)
            else:
                best = np.minimum(best, bl)
                active = alive & (entry - best >= start)
                sl = np.where(active, np.minimum(sl, best + trail), sl)
        timeout = c[t + horizon] if buy else c[t + horizon] + spr[t + horizon]
        exit_px = np.where(kind == 0, timeout, exit_px)
        sign = 1.0 if buy else -1.0
        rr = sign * (exit_px - entry) / sl_dist
    ok = np.isfinite(rr) & np.isfinite(a) & (a > 0)
    win = np.full(n, np.nan)
    loss = np.full(n, np.nan)
    r = np.full(n, np.nan)
    won = (kind == 1) | ((kind == 2) & (rr > 0))
    lost = (kind == 2) & ~(rr > 0)
    win[:m] = np.where(ok, won.astype(np.float64), np.nan)
    loss[:m] = np.where(ok, lost.astype(np.float64), np.nan)
    r[:m] = np.where(ok, rr, np.nan)
    return ArmLabels(win, loss, r)


def label_all(
    bars: pd.DataFrame,
    atr_tf: Dict[str, np.ndarray],
    arms: List[Arm],
    point: float,
    default_spread_points: float = 10.0,
    extra_cost: float = 0.0,
) -> Dict[str, ArmLabels]:
    return {
        arm.key: label_arm(bars, atr_tf[arm.tf], arm, arm.horizon, point, default_spread_points, extra_cost)
        for arm in arms
    }


def direction_labels(bars: pd.DataFrame, horizon: int) -> np.ndarray:
    """1.0 if the close ``horizon`` bars later is higher than the close of bar t, else 0.0
    (NaN where the future is not known yet)."""
    c = bars["close"].to_numpy(dtype=np.float64)
    out = np.full(len(c), np.nan)
    if len(c) > horizon:
        out[: len(c) - horizon] = (c[horizon:] > c[:-horizon]).astype(np.float64)
    return out
