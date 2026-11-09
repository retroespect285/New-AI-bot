import pandas as pd
import pytest

from tradebot.data import load_context, resample_bars, save_bars, synthetic_bars


def test_resample_matches_mt5_alignment():
    m1 = synthetic_bars(600, seed=2, start="2024-01-02 00:00", minutes=1, price=2350, point=0.01)
    h1 = resample_bars(m1, 60)
    assert list(h1["time"][:3]) == [pd.Timestamp("2024-01-02 00:00"), pd.Timestamp("2024-01-02 01:00"),
                                    pd.Timestamp("2024-01-02 02:00")]
    first = m1.iloc[:60]
    row = h1.iloc[0]
    assert row["open"] == first["open"].iat[0] and row["close"] == first["close"].iat[-1]
    assert row["high"] == first["high"].max() and row["low"] == first["low"].min()
    assert row["tick_volume"] == first["tick_volume"].sum()
    h4 = resample_bars(m1, 240)
    assert (h4["time"].dt.hour % 4 == 0).all()


def test_load_context_finds_exported_files_and_rebuilds_missing_higher_timeframes(tmp_path):
    m1 = synthetic_bars(3000, seed=3, minutes=1, price=2350, point=0.01)
    m5 = resample_bars(m1, 5)
    save_bars(m5, str(tmp_path / "XAUUSD_M5.csv"))
    save_bars(m1, str(tmp_path / "XAUUSD_M1.csv"))
    ctx = load_context(str(tmp_path / "XAUUSD_M5.csv"), "M5", ["M1", "H1"], m5)
    assert len(ctx["M1"]) == 3000
    pd.testing.assert_frame_equal(ctx["H1"], resample_bars(m5, 60), check_dtype=False)
    (tmp_path / "XAUUSD_M1.csv").unlink()
    with pytest.raises(FileNotFoundError, match="export"):
        load_context(str(tmp_path / "XAUUSD_M5.csv"), "M5", ["M1"], m5)
