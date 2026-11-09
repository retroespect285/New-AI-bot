"""Chart reading: turn raw OHLCV bars into scale-free features the model can learn from.

The bot reads several charts at once:

* the **trading timeframe** (M5 by default) in full detail (~58 indicators and candle patterns);
* **context timeframes** (M1, M30, H1, H4 by default) with a compact set of ~20 indicators each.
  M1 shows the small movements inside each 5-minute candle, M30/H1/H4 the bigger trend;
* the **candle still forming** on each higher timeframe (e.g. "the current H1 candle is up 0.8 ATR
  and near its high"), rebuilt from the trading-timeframe bars.

Every feature at bar ``i`` only uses information available when bar ``i`` closed: a context
candle is used only once it has closed (no look-ahead). ``tests/test_features.py`` verifies this.
Prices are normalised by ATR so the same model works in quiet and in wild markets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

FEATURE_VERSION = 2
# Bars needed before all indicators are defined (EMA200 / ATR100 plus margin).
MIN_HISTORY = 260
MIN_CONTEXT_HISTORY = 60  # context charts use indicators up to 50 bars long
MAX_CONTEXT_GAP = pd.Timedelta(days=3)  # older context data (missing history) is treated as unknown


def _wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return _wilder(true_range(high, low, close), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0.0), n)
    loss = _wilder(-delta.clip(upper=0.0), n)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(loss != 0.0, 100.0).where(gain.notna())


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_n = _wilder(true_range(high, low, close), n)
    plus_di = 100.0 * _wilder(plus_dm, n) / atr_n
    minus_di = 100.0 * _wilder(minus_dm, n) / atr_n
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return _wilder(dx, n), plus_di, minus_di


@dataclass
class FeatureSet:
    X: np.ndarray  # (n_bars, n_features) float64, NaN where undefined
    names: list
    atr: np.ndarray  # raw ATR of the trading timeframe, in price units
    valid: np.ndarray  # bool mask of rows with every feature defined
    frame: pd.DataFrame  # same data as X, for inspection / logging
    # ATR of every chart, aligned to the trading bars (for stop / target distances)
    atr_tf: Dict[str, np.ndarray] = field(default_factory=dict)


def build_features(bars: pd.DataFrame, point: float, atr_period: int = 14) -> FeatureSet:
    """Features of a single chart (the trading timeframe only)."""
    frame, a = base_frame(bars, point, atr_period)
    X = frame.to_numpy(dtype=np.float64)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(a) & (a > 0)
    return FeatureSet(X=X, names=list(frame.columns), atr=a, valid=valid, frame=frame)


def base_frame(bars: pd.DataFrame, point: float, atr_period: int = 14):
    """Full indicator set for the trading timeframe. Returns (DataFrame, ATR array)."""
    o = bars["open"].astype(float)
    h = bars["high"].astype(float)
    lo = bars["low"].astype(float)
    c = bars["close"].astype(float)
    a = atr(h, lo, c, atr_period)
    safe_a = a.replace(0.0, np.nan)
    f = {}

    # --- momentum: move over k bars measured in ATRs
    for k in (1, 3, 6, 12, 24, 48):
        f[f"mom_{k}"] = (c - c.shift(k)) / safe_a

    # --- trend: distance from EMAs and their slopes
    emas = {n: _ema(c, n) for n in (9, 21, 50, 100, 200)}
    for n, e in emas.items():
        f[f"ema_dist_{n}"] = (c - e) / safe_a
    f["ema_9_21"] = (emas[9] - emas[21]) / safe_a
    f["ema_21_50"] = (emas[21] - emas[50]) / safe_a
    f["ema_50_200"] = (emas[50] - emas[200]) / safe_a
    f["ema_slope_21"] = (emas[21] - emas[21].shift(5)) / safe_a
    f["ema_slope_50"] = (emas[50] - emas[50].shift(5)) / safe_a
    f["ema_slope_200"] = (emas[200] - emas[200].shift(12)) / safe_a

    # --- oscillators
    f["rsi_7"] = (rsi(c, 7) - 50.0) / 50.0
    f["rsi_14"] = (rsi(c, 14) - 50.0) / 50.0
    macd = _ema(c, 12) - _ema(c, 26)
    signal = _ema(macd, 9)
    f["macd"] = macd / safe_a
    f["macd_hist"] = (macd - signal) / safe_a
    low14 = lo.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = 100.0 * (c - low14) / (high14 - low14).replace(0.0, np.nan)
    f["stoch_k"] = (stoch_k - 50.0) / 50.0
    f["stoch_d"] = (stoch_k.rolling(3).mean() - 50.0) / 50.0
    adx_v, pdi, mdi = adx(h, lo, c, 14)
    f["adx"] = adx_v / 100.0
    f["di_diff"] = (pdi - mdi) / 100.0

    # --- volatility regime
    mid20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0)
    f["bb_pos"] = (c - mid20) / (2.0 * std20).replace(0.0, np.nan)
    f["bb_width"] = std20 / safe_a
    f["atr_ratio"] = a / atr(h, lo, c, 100) - 1.0
    f["atr_bps"] = np.log(a / c * 1e4)
    f["range_atr"] = (h - lo) / safe_a

    # --- support / resistance and breakouts
    for n in (20, 50, 100):
        hh = h.rolling(n).max()
        ll = lo.rolling(n).min()
        f[f"chan_pos_{n}"] = (c - ll) / (hh - ll).replace(0.0, np.nan)
        f[f"to_high_{n}"] = (hh - c) / safe_a
        f[f"to_low_{n}"] = (c - ll) / safe_a

    # --- candlestick anatomy and patterns
    rng = (h - lo).replace(0.0, np.nan)
    body = c - o
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - lo
    f["body"] = body / rng
    f["upper_wick"] = upper / rng
    f["lower_wick"] = lower / rng
    f["body_1"] = f["body"].shift(1)
    f["body_2"] = f["body"].shift(2)
    prev_o, prev_c = o.shift(1), c.shift(1)
    f["bull_engulf"] = ((prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)).astype(float)
    f["bear_engulf"] = ((prev_c > prev_o) & (c < o) & (c <= prev_o) & (o >= prev_c)).astype(float)
    abs_body = body.abs()
    f["hammer"] = ((lower > 2.0 * abs_body) & (upper < abs_body + 0.1 * rng)).astype(float)
    f["shooting_star"] = ((upper > 2.0 * abs_body) & (lower < abs_body + 0.1 * rng)).astype(float)
    f["doji"] = (abs_body < 0.1 * rng).astype(float)
    f["inside_bar"] = ((h < h.shift(1)) & (lo > lo.shift(1))).astype(float)
    f["outside_bar"] = ((h > h.shift(1)) & (lo < lo.shift(1))).astype(float)
    direction = np.sign(body)
    streak_id = (direction != direction.shift(1)).cumsum()
    f["streak"] = direction * direction.groupby(streak_id).cumcount().add(1) / 10.0

    # --- activity
    if "tick_volume" in bars.columns:
        vol = bars["tick_volume"].astype(float)
        f["vol_ratio"] = np.log((vol + 1.0) / (vol.rolling(50).mean() + 1.0))
    else:
        f["vol_ratio"] = pd.Series(0.0, index=bars.index)
    if "spread" in bars.columns:
        f["spread_atr"] = bars["spread"].astype(float) * point / safe_a
    else:
        f["spread_atr"] = pd.Series(0.0, index=bars.index)

    # --- session / calendar (broker server time)
    t = pd.to_datetime(bars["time"])
    hour = t.dt.hour + t.dt.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    f["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 5.0)
    f["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 5.0)

    frame = pd.DataFrame(f, index=bars.index).replace([np.inf, -np.inf], np.nan)
    # A flat bar / flat market makes some ratios undefined; use the neutral value instead of
    # discarding the whole row.
    neutral = {"body": 0.0, "upper_wick": 0.0, "lower_wick": 0.0, "body_1": 0.0, "body_2": 0.0,
               "bb_pos": 0.0, "stoch_k": 0.0, "stoch_d": 0.0,
               "chan_pos_20": 0.5, "chan_pos_50": 0.5, "chan_pos_100": 0.5}
    frame = frame.fillna(value=neutral)
    frame.iloc[:MIN_HISTORY] = np.nan  # indicators still warming up
    return frame, a.to_numpy(dtype=np.float64)


def context_frame(bars: pd.DataFrame, prefix: str, atr_period: int = 14):
    """Compact indicator set for a context chart. Returns (DataFrame, ATR array)."""
    o = bars["open"].astype(float)
    h = bars["high"].astype(float)
    lo = bars["low"].astype(float)
    c = bars["close"].astype(float)
    a = atr(h, lo, c, atr_period)
    sa = a.replace(0.0, np.nan)
    f = {}
    for k in (1, 3, 12):
        f[f"mom_{k}"] = (c - c.shift(k)) / sa
    e20, e50 = _ema(c, 20), _ema(c, 50)
    f["ema_dist_20"] = (c - e20) / sa
    f["ema_dist_50"] = (c - e50) / sa
    f["ema_20_50"] = (e20 - e50) / sa
    f["ema_slope_20"] = (e20 - e20.shift(3)) / sa
    f["rsi_14"] = (rsi(c, 14) - 50.0) / 50.0
    macd = _ema(c, 12) - _ema(c, 26)
    f["macd_hist"] = (macd - _ema(macd, 9)) / sa
    adx_v, pdi, mdi = adx(h, lo, c, 14)
    f["adx"] = adx_v / 100.0
    f["di_diff"] = (pdi - mdi) / 100.0
    mid20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0)
    f["bb_pos"] = (c - mid20) / (2.0 * std20).replace(0.0, np.nan)
    f["atr_ratio"] = a / atr(h, lo, c, 50) - 1.0
    for n in (20, 50):
        hh, ll = h.rolling(n).max(), lo.rolling(n).min()
        f[f"chan_pos_{n}"] = (c - ll) / (hh - ll).replace(0.0, np.nan)
    rng = (h - lo).replace(0.0, np.nan)
    body = c - o
    f["body"] = body / rng
    f["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    f["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - lo) / rng
    po, pc = o.shift(1), c.shift(1)
    bull = (pc < po) & (c > o) & (c >= po) & (o <= pc)
    bear = (pc > po) & (c < o) & (c <= po) & (o >= pc)
    f["engulf"] = bull.astype(float) - bear.astype(float)
    direction = np.sign(body)
    f["streak"] = direction * direction.groupby((direction != direction.shift(1)).cumsum()).cumcount().add(1) / 10.0
    frame = pd.DataFrame(f, index=bars.index).replace([np.inf, -np.inf], np.nan)
    frame = frame.fillna(value={"body": 0.0, "upper_wick": 0.0, "lower_wick": 0.0, "bb_pos": 0.0,
                                "chan_pos_20": 0.5, "chan_pos_50": 0.5})
    frame.iloc[:MIN_CONTEXT_HISTORY] = np.nan
    frame.columns = [f"{prefix}_{col}" for col in frame.columns]
    return frame, a.to_numpy(dtype=np.float64)


def _close_ns(times: pd.Series, minutes: int) -> np.ndarray:
    t = pd.to_datetime(times).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return t + minutes * 60_000_000_000


def align_asof(base_close_ns: np.ndarray, ctx_close_ns: np.ndarray) -> np.ndarray:
    """For every trading bar, the index of the last context candle that had *closed* by the time
    the trading bar closed (-1 if none, or if that candle is too old to be meaningful)."""
    idx = np.searchsorted(ctx_close_ns, base_close_ns, side="right") - 1
    ok = idx >= 0
    gap = np.where(ok, base_close_ns - ctx_close_ns[np.clip(idx, 0, None)], 0)
    idx[gap > MAX_CONTEXT_GAP.value] = -1
    return idx


def _take(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = values[np.clip(idx, 0, None)].astype(np.float64)
    out[idx < 0] = np.nan
    return out


def forming_frame(bars: pd.DataFrame, base_minutes: int, tf: str, tf_minutes: int, tf_atr: np.ndarray) -> pd.DataFrame:
    """The higher-timeframe candle that is still forming, rebuilt from trading bars <= i."""
    t = pd.to_datetime(bars["time"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    period = t - (t % (tf_minutes * 60_000_000_000))
    grp = pd.Series(period, index=bars.index)
    first_open = bars["open"].astype(float).groupby(grp).transform("first")
    hi = bars["high"].astype(float).groupby(grp).cummax()
    lo = bars["low"].astype(float).groupby(grp).cummin()
    c = bars["close"].astype(float)
    a = pd.Series(tf_atr, index=bars.index).replace(0.0, np.nan)
    elapsed = ((t - period) / 60_000_000_000 + base_minutes) / tf_minutes
    p = tf.lower()
    out = pd.DataFrame({
        f"{p}_now_body": (c - first_open) / a,
        f"{p}_now_pos": ((c - lo) / (hi - lo).replace(0.0, np.nan)).fillna(0.5),
        f"{p}_now_elapsed": elapsed,
    }, index=bars.index)
    return out.replace([np.inf, -np.inf], np.nan)


def build_mtf_features(bars: pd.DataFrame, context: Mapping[str, pd.DataFrame], base_tf: str,
                       tf_minutes: Mapping[str, int], point: float, atr_period: int = 14,
                       required_atr: Optional[list] = None) -> FeatureSet:
    """Features from the trading chart plus every context chart, aligned without look-ahead.

    ``context`` maps a timeframe name (e.g. ``"H1"``) to its bars. ``required_atr`` lists the
    timeframes whose ATR must be known for a row to be usable (the trade-style timeframes).
    """
    base_min = tf_minutes[base_tf]
    frame, base_atr = base_frame(bars, point, atr_period)
    parts = [frame]
    atr_tf = {base_tf: base_atr}
    base_close = _close_ns(bars["time"], base_min)
    for tf, cbars in context.items():
        cmin = tf_minutes[tf]
        cframe, catr = context_frame(cbars, tf.lower(), atr_period)
        idx = align_asof(base_close, _close_ns(cbars["time"], cmin))
        aligned = pd.DataFrame({col: _take(cframe[col].to_numpy(), idx) for col in cframe.columns},
                               index=bars.index)
        atr_tf[tf] = _take(catr, idx)
        parts.append(aligned)
        if cmin > base_min:
            parts.append(forming_frame(bars, base_min, tf, cmin, atr_tf[tf]))
    frame = pd.concat(parts, axis=1)
    X = frame.to_numpy(dtype=np.float64)
    valid = np.isfinite(X).all(axis=1)
    for tf in (required_atr or [base_tf]):
        a = atr_tf[tf]
        valid &= np.isfinite(a) & (a > 0)
    return FeatureSet(X=X, names=list(frame.columns), atr=base_atr, valid=valid, frame=frame, atr_tf=atr_tf)
