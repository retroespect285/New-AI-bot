"""Loading / saving bar data, and a synthetic market generator for tests and demos."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .brokers.base import BAR_COLUMNS
from .config import TIMEFRAME_MINUTES
from .types import SymbolInfo

log = logging.getLogger(__name__)

_ALIASES = {
    "date": "date", "time": "time", "datetime": "time", "timestamp": "time",
    "open": "open", "high": "high", "low": "low", "close": "close",
    "tickvol": "tick_volume", "tick_volume": "tick_volume", "volume": "tick_volume",
    "vol": "real_volume", "real_volume": "real_volume", "spread": "spread",
}


def load_bars(path: str) -> pd.DataFrame:
    """Read bars from CSV. Accepts this bot's own export format *and* the file produced by the MT5
    terminal (right-click a chart symbol -> Bars -> Export, tab separated <DATE> <TIME> ...)."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [_ALIASES.get(c.strip().strip("<>").lower(), c.strip().strip("<>").lower()) for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        df["time"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"])
    else:
        df["time"] = pd.to_datetime(df["time"])
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "tick_volume" not in df:
        df["tick_volume"] = 0
    if "spread" not in df:
        df["spread"] = 0
    df["time"] = df["time"].astype("datetime64[ns]")
    df = df[BAR_COLUMNS].sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def save_bars(df: pd.DataFrame, path: str, info: Optional[SymbolInfo] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df[BAR_COLUMNS].to_csv(p, index=False)
    if info is not None:
        p.with_suffix(".json").write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")


def load_symbol_info(csv_path: str) -> Optional[SymbolInfo]:
    p = Path(csv_path).with_suffix(".json")
    if not p.exists():
        return None
    return SymbolInfo(**json.loads(p.read_text(encoding="utf-8")))


def resample_bars(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Build higher-timeframe candles (aligned like MT5: to the server day) from lower ones."""
    t = pd.to_datetime(bars["time"])
    key = t.dt.floor(f"{minutes}min")
    g = bars.assign(_k=key.values).groupby("_k", sort=True)
    out = pd.DataFrame({
        "time": g["time"].first().index.astype("datetime64[ns]"),
        "open": g["open"].first().to_numpy(),
        "high": g["high"].max().to_numpy(),
        "low": g["low"].min().to_numpy(),
        "close": g["close"].last().to_numpy(),
        "tick_volume": g["tick_volume"].sum().to_numpy() if "tick_volume" in bars else 0,
        "spread": np.round(g["spread"].mean().to_numpy()).astype(int) if "spread" in bars else 0,
    })
    return out.reset_index(drop=True)


def bars_path(directory: str, symbol: str, timeframe: str) -> Path:
    return Path(directory) / f"{symbol}_{timeframe}.csv"


def load_context(base_csv: str, base_tf: str, timeframes: List[str], base_bars: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Find the other charts next to ``<SYMBOL>_<TF>.csv`` (as written by ``tradebot export``).
    Missing higher timeframes are rebuilt from the base bars; missing lower ones are an error."""
    p = Path(base_csv)
    stem = p.stem
    prefix = stem[: -(len(base_tf) + 1)] if stem.upper().endswith("_" + base_tf) else stem
    out = {}
    for tf in timeframes:
        candidate = p.parent / f"{prefix}_{tf}.csv"
        if candidate.exists():
            out[tf] = load_bars(str(candidate))
        elif TIMEFRAME_MINUTES[tf] > TIMEFRAME_MINUTES[base_tf]:
            log.warning("%s not found - building %s candles from %s", candidate.name, tf, p.name)
            out[tf] = resample_bars(base_bars, TIMEFRAME_MINUTES[tf])
        else:
            raise FileNotFoundError(
                f"{candidate} is missing. The bot reads the {tf} chart too: run `python -m tradebot export` "
                f"to download every timeframe, or remove {tf} from model.context_timeframes.")
    return out


def synthetic_market(n_base: int, base_tf: str, timeframes: List[str], seed: int = 0,
                     price: float = 2350.0, point: float = 0.01, start: str = "2024-01-01"):
    """Synthetic gold-like market: 1-minute bars resampled into every timeframe needed.
    Returns (base bars, {timeframe: bars}). For testing the pipeline only."""
    base_min = TIMEFRAME_MINUTES[base_tf]
    m1 = synthetic_bars(n_base * base_min + 60, seed=seed, start=start, minutes=1, price=price, point=point,
                        vol=0.00026, spread_points=25)
    base = resample_bars(m1, base_min).iloc[:n_base].reset_index(drop=True)
    end = base["time"].iat[-1] + pd.Timedelta(minutes=base_min)
    m1 = m1[m1["time"] < end].reset_index(drop=True)
    ctx = {}
    for tf in timeframes:
        ctx[tf] = m1 if tf == "M1" else resample_bars(m1, TIMEFRAME_MINUTES[tf])
    return base, ctx


def synthetic_bars(n: int = 20_000, seed: int = 0, start: str = "2024-01-01", minutes: int = 5,
                   price: float = 1.10, point: float = 1e-5, vol: float = 0.00025,
                   spread_points: float = 8.0) -> pd.DataFrame:
    """Regime-switching random walk (trending / mean-reverting phases, volatility clustering,
    intraday volatility pattern, weekends skipped). Only for testing the pipeline - real markets are
    much harder to predict."""
    rng = np.random.default_rng(seed)
    times = []
    t = pd.Timestamp(start)
    step = pd.Timedelta(minutes=minutes)
    while len(times) < n:
        if t.dayofweek < 5:
            times.append(t)
        t += step
    times = pd.DatetimeIndex(times)
    hours = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    intraday = 0.6 + 0.8 * np.exp(-((hours - 14.0) ** 2) / 18.0)

    sub = 6  # sub-steps per bar to build realistic highs and lows
    ret = np.empty(n)
    base_vol = vol
    regime_scale = np.sqrt(minutes / 5.0)  # keep regime drift comparable across bar sizes
    drift = 0.0
    phi = 0.0
    prev = 0.0
    regime_left = 0
    for k in range(n):
        if regime_left <= 0:
            regime_left = int(rng.integers(150, 900) * 5 / minutes)
            kind = rng.choice(["trend", "revert", "noise"], p=[0.35, 0.35, 0.30])
            drift = rng.normal(0, 0.16 * base_vol) * regime_scale ** 2 if kind == "trend" else 0.0
            phi = {"trend": 0.15, "revert": -0.2, "noise": 0.0}[kind]
        regime_left -= 1
        vol = np.sqrt(0.97 * vol**2 + 0.03 * (base_vol * rng.lognormal(0, 0.35)) ** 2)
        r = drift + phi * prev + vol * intraday[k] * rng.standard_t(5) / np.sqrt(5 / 3)
        ret[k] = r
        prev = r

    close = price * np.exp(np.cumsum(ret))
    open_ = np.r_[price, close[:-1]]
    paths = open_[:, None] * np.exp(
        np.cumsum(rng.normal(0, 1, (n, sub)), axis=1) / np.sqrt(sub) * (np.abs(ret)[:, None] + vol * 0.5)
    )
    paths[:, -1] = close
    high = np.maximum(paths.max(axis=1), np.maximum(open_, close))
    low = np.minimum(paths.min(axis=1), np.minimum(open_, close))
    spread = np.clip(np.round(rng.normal(spread_points, spread_points / 4, n) / intraday),
                     max(1, spread_points / 3), spread_points * 5)
    digits = int(round(-np.log10(point)))
    return pd.DataFrame({
        "time": times.astype("datetime64[ns]"),
        "open": np.round(open_, digits),
        "high": np.round(high, digits),
        "low": np.round(low, digits),
        "close": np.round(close, digits),
        "tick_volume": rng.poisson(200 * intraday).astype(int),
        "spread": spread.astype(int),
    })
