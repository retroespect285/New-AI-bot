import json

import pandas as pd

from tests.conftest import EURUSD, small_config
from tradebot.backtest import run_backtest
from tradebot.data import load_bars, save_bars, synthetic_bars


def test_backtest_runs_and_writes_reports(tmp_path):
    cfg = small_config()
    bars = synthetic_bars(2600, seed=21)
    res = run_backtest(cfg, bars, "EURUSD", EURUSD, out_dir=str(tmp_path), warmup_bars=1000, progress=False)
    s = res.stats
    for key in ("trades", "win_rate_pct", "return_pct", "max_drawdown_pct", "profit_factor", "model"):
        assert key in s
    assert len(res.equity) == len(bars) - 1 - 1000
    assert (tmp_path / "summary.json").exists() and (tmp_path / "equity.csv").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["trades"] == s["trades"]
    decisions = pd.read_csv(tmp_path / "decisions.csv")
    assert len(decisions) == len(bars) - 1 - 1000
    assert s["final_equity"] == round(res.equity["equity"].iat[-1], 2)


def test_backtest_has_no_look_ahead(tmp_path):
    """Every decision must be identical whether or not the future bars are in the data set."""
    cfg = small_config()
    bars = synthetic_bars(2800, seed=22)
    short, full = tmp_path / "short", tmp_path / "full"
    run_backtest(cfg, bars.iloc[:2300].reset_index(drop=True), "EURUSD", EURUSD, out_dir=str(short),
                 warmup_bars=1000, progress=False)
    run_backtest(cfg, bars, "EURUSD", EURUSD, out_dir=str(full), warmup_bars=1000, progress=False)
    a = pd.read_csv(short / "decisions.csv")
    b = pd.read_csv(full / "decisions.csv").iloc[: len(a)]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    oa = pd.read_csv(short / "orders.csv") if (short / "orders.csv").exists() else pd.DataFrame()
    ob = pd.read_csv(full / "orders.csv") if (full / "orders.csv").exists() else pd.DataFrame()
    if len(oa):
        pd.testing.assert_frame_equal(oa.reset_index(drop=True), ob.iloc[: len(oa)].reset_index(drop=True))


def test_csv_roundtrip_and_mt5_terminal_format(tmp_path):
    bars = synthetic_bars(50, seed=1)
    path = tmp_path / "EURUSD_M5.csv"
    save_bars(bars, str(path), EURUSD)
    back = load_bars(str(path))
    pd.testing.assert_frame_equal(back, bars, check_dtype=False)
    # the format written by MT5's "Export Bars" dialog
    mt5 = tmp_path / "mt5.csv"
    mt5.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2024.01.02\t00:00:00\t1.10380\t1.10400\t1.10370\t1.10390\t120\t0\t8\n"
        "2024.01.02\t00:05:00\t1.10390\t1.10410\t1.10350\t1.10360\t98\t0\t9\n"
    )
    df = load_bars(str(mt5))
    assert list(df.columns) == ["time", "open", "high", "low", "close", "tick_volume", "spread"]
    assert df["time"].iat[1] == pd.Timestamp("2024-01-02 00:05") and df["spread"].iat[1] == 9
