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
