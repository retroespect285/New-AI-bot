"""Loading / saving bar data, and a synthetic market generator for tests and demos."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .brokers.base import BAR_COLUMNS
from .types import SymbolInfo

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


def synthetic_bars(n: int = 20_000, seed: int = 0, start: str = "2024-01-01", minutes: int = 5,
                   price: float = 1.10, point: float = 1e-5) -> pd.DataFrame:
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
    vol = 0.00025
    drift = 0.0
    phi = 0.0
    prev = 0.0
    regime_left = 0
    for k in range(n):
        if regime_left <= 0:
            regime_left = int(rng.integers(150, 900))
            kind = rng.choice(["trend", "revert", "noise"], p=[0.35, 0.35, 0.30])
            drift = rng.normal(0, 0.00004) if kind == "trend" else 0.0
            phi = {"trend": 0.15, "revert": -0.2, "noise": 0.0}[kind]
        regime_left -= 1
        vol = np.sqrt(0.97 * vol**2 + 0.03 * (0.00025 * rng.lognormal(0, 0.35)) ** 2)
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
    spread = np.clip(np.round(rng.normal(8, 2, n) / intraday), 3, 40)
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
