import numpy as np

from tradebot.data import synthetic_bars
from tradebot.features import MIN_HISTORY, build_features


def test_features_are_defined_after_warmup():
    bars = synthetic_bars(1500, seed=3)
    fs = build_features(bars, 1e-5)
    assert fs.X.shape == (1500, len(fs.names))
    assert not fs.valid[:MIN_HISTORY].any()
    assert fs.valid[MIN_HISTORY + 10:].mean() > 0.99
    assert np.isfinite(fs.X[fs.valid]).all()
    assert (fs.atr[fs.valid] > 0).all()


def test_features_never_look_ahead():
    """A feature at bar i must be identical whether or not later bars exist."""
    bars = synthetic_bars(1600, seed=4)
    full = build_features(bars, 1e-5)
    for cut in (900, 1200, 1599):
        part = build_features(bars.iloc[: cut + 1].reset_index(drop=True), 1e-5)
        np.testing.assert_allclose(part.X[cut], full.X[cut], rtol=1e-9, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(part.atr[cut], full.atr[cut], rtol=1e-12)


def test_window_start_does_not_matter_much():
    """Live trading computes features on a rolling window; values must match the full history."""
    bars = synthetic_bars(4000, seed=5)
    full = build_features(bars, 1e-5)
    window = build_features(bars.iloc[-1500:].reset_index(drop=True), 1e-5)
    np.testing.assert_allclose(window.X[-1], full.X[-1], rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------- multi-timeframe
import pandas as pd  # noqa: E402

from tradebot.config import TIMEFRAME_MINUTES  # noqa: E402
from tradebot.data import synthetic_market  # noqa: E402
from tradebot.features import align_asof, build_mtf_features  # noqa: E402

TFS = ["M1", "M30", "H1", "H4"]


def _mtf(base, ctx):
    return build_mtf_features(base, ctx, "M5", TIMEFRAME_MINUTES, 0.01, 14, required_atr=["M5", "H1", "H4"])


def test_mtf_features_cover_every_chart():
    base, ctx = synthetic_market(4500, "M5", TFS, seed=5)
    fs = _mtf(base, ctx)
    for prefix in ("m1_", "m30_", "h1_", "h4_", "m30_now_", "h1_now_", "h4_now_"):
        assert any(n.startswith(prefix) for n in fs.names), prefix
    assert fs.valid[-500:].all()
    assert set(fs.atr_tf) == {"M5", "M1", "M30", "H1", "H4"}
    # gold-like volatility: higher timeframes have bigger candles
    med = {tf: float(np.nanmedian(a[fs.valid])) for tf, a in fs.atr_tf.items()}
    assert med["M1"] < med["M5"] < med["M30"] < med["H1"] < med["H4"]


def test_mtf_features_never_look_ahead():
    """Cut all charts at a point in time, as MT5 would deliver them live (the forming candle of
    every chart is not closed yet): every feature of the last trading bar must be identical."""
    base, ctx = synthetic_market(4500, "M5", TFS, seed=6)
    full = _mtf(base, ctx)
    for cut in (3800, 4100, 4499):
        now = base["time"].iat[cut] + pd.Timedelta(minutes=5)
        part_ctx = {tf: d[d["time"] + pd.Timedelta(minutes=TIMEFRAME_MINUTES[tf]) <= now] for tf, d in ctx.items()}
        part = _mtf(base.iloc[: cut + 1], part_ctx)
        assert part.valid[cut] == full.valid[cut]
        np.testing.assert_allclose(part.X[cut], full.X[cut], rtol=1e-9, atol=1e-12, equal_nan=True)
        for tf in full.atr_tf:
            np.testing.assert_allclose(part.atr_tf[tf][cut], full.atr_tf[tf][cut], rtol=1e-12)


def test_context_candle_is_used_only_after_it_closed():
    t = lambda s: np.array([pd.Timestamp(x).value for x in s], dtype=np.int64)  # noqa: E731
    h1_close = t(["2024-01-01 01:00", "2024-01-01 02:00"])
    base_close = t(["2024-01-01 00:55", "2024-01-01 01:00", "2024-01-01 01:55", "2024-01-01 02:00"])
    assert list(align_asof(base_close, h1_close)) == [-1, 0, 0, 1]
