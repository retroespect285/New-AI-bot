import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradebot.config import BotConfig, TradeStyle  # noqa: E402
from tradebot.types import SymbolInfo  # noqa: E402

EURUSD = SymbolInfo(name="EURUSD", point=1e-5, digits=5, tick_size=1e-5, tick_value=1.0,
                    contract_size=100000, volume_min=0.01, volume_max=100, volume_step=0.01)


@pytest.fixture
def eurusd() -> SymbolInfo:
    return EURUSD


def small_config() -> BotConfig:
    """Fast settings so tests run in seconds (not suitable for real trading)."""
    cfg = BotConfig()
    m = cfg.model
    m.styles = [TradeStyle("M5", 1.0, 1.5, 12), TradeStyle("M5", 1.5, 3.0, 12)]
    m.context_timeframes = []  # single chart: fast; multi-chart tests use mtf_config()
    m.outlooks = ["M5", "M30"]
    m.min_train_samples = 400
    m.min_eval_samples = 150
    m.gbm_window = 800
    m.gbm_refit_every = 400
    m.gbm_max_iter = 30
    cfg.trading.symbols = ["EURUSD"]
    cfg.trading.lookback_bars = 600
    cfg.trading.friday_cutoff_hour = None
    cfg.trading.session.start_hour, cfg.trading.session.end_hour = 0, 24
    return cfg.validate()


def mtf_config() -> BotConfig:
    """Small multi-timeframe setup: trades M5 and H1 styles, reads M1 / M30 / H1."""
    cfg = small_config()
    cfg.model.context_timeframes = ["M1", "M30", "H1"]
    cfg.model.styles = [TradeStyle("M5", 1.0, 1.5, 12), TradeStyle("H1", 1.0, 1.5, 36)]
    cfg.model.outlooks = ["M5", "M30", "H1"]
    return cfg.validate()


@pytest.fixture
def cfg() -> BotConfig:
    return small_config()


def make_bars(rows, start="2024-01-01 00:00", minutes=5, spread=0):
    """rows: list of (open, high, low, close)."""
    times = pd.date_range(start, periods=len(rows), freq=f"{minutes}min")
    o, h, lo, c = np.array(rows, dtype=float).T
    return pd.DataFrame({"time": times, "open": o, "high": h, "low": lo, "close": c,
                         "tick_volume": 100, "spread": spread})
