import numpy as np

from tests.conftest import make_bars
from tradebot.labels import Arm, label_arm
from tradebot.types import Side

PT = 1e-5


def flat_then(rows_after, n_before=3, price=1.0):
    return [(price, price, price, price)] * n_before + rows_after


def test_buy_take_profit_hit_first():
    # atr is given explicitly as 0.001 (100 points); entry at open of bar t+1 = 1.0
    rows = flat_then([(1.0, 1.0016, 0.9995, 1.0015), (1.0015, 1.0016, 1.001, 1.001)] + [(1.001,) * 4] * 5)
    bars = make_bars(rows)
    atr = np.full(len(bars), 0.001)
    lab = label_arm(bars, atr, Arm(Side.BUY, 1.0, 1.5), horizon=3, point=PT, default_spread_points=0)
    t = 2  # entry at open of bar 3
    assert lab.win[t] == 1.0
    assert np.isclose(lab.r[t], 1.5)


def test_same_bar_sl_and_tp_counts_as_loss():
    rows = flat_then([(1.0, 1.002, 0.998, 1.0)] + [(1.0,) * 4] * 5)
    bars = make_bars(rows)
    atr = np.full(len(bars), 0.001)
    lab = label_arm(bars, atr, Arm(Side.BUY, 1.0, 1.5), horizon=3, point=PT, default_spread_points=0)
    assert lab.win[2] == 0.0
    assert np.isclose(lab.r[2], -1.0)


def test_timeout_gives_partial_result():
    rows = flat_then([(1.0, 1.0004, 0.9999, 1.0003)] * 4 + [(1.0003,) * 4] * 3)
    bars = make_bars(rows)
    atr = np.full(len(bars), 0.001)
    lab = label_arm(bars, atr, Arm(Side.BUY, 1.0, 1.5), horizon=3, point=PT, default_spread_points=0)
    assert lab.win[2] == 0.0
    assert np.isclose(lab.r[2], 0.3)


def test_sell_pays_the_spread_on_exit():
    # price falls 150 points; with 20 points spread a short's TP needs ask (= bid + spread) to fall
    rows = flat_then([(1.0, 1.0, 0.9985, 0.9985)] + [(0.9985,) * 4] * 5)
    bars = make_bars(rows, spread=20)
    atr = np.full(len(bars), 0.001)
    arm = Arm(Side.SELL, 1.0, 1.5)
    lab = label_arm(bars, atr, arm, horizon=3, point=PT)
    assert lab.win[2] == 0.0  # 0.9985 + 0.0002 > TP 0.9985
    rows2 = flat_then([(1.0, 1.0, 0.9982, 0.9982)] + [(0.9982,) * 4] * 5)
    lab2 = label_arm(make_bars(rows2, spread=20), atr, arm, horizon=3, point=PT)
    assert lab2.win[2] == 1.0


def test_unresolved_tail_is_nan():
    bars = make_bars([(1.0,) * 4] * 20)
    atr = np.full(20, 0.001)
    lab = label_arm(bars, atr, Arm(Side.BUY, 1.0, 1.5), horizon=5, point=PT)
    assert np.isfinite(lab.win[:15]).all()
    assert np.isnan(lab.win[15:]).all()
