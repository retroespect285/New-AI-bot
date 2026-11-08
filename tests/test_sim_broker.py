import pytest

from tests.conftest import make_bars
from tradebot.brokers.sim import SimBroker
from tradebot.types import Side


def _broker(rows, info, spread=0):
    bars = make_bars(rows, spread=spread)
    b = SimBroker({"EURUSD": bars}, {"EURUSD": info}, initial_balance=10_000, default_spread_points=0)
    b.set_index("EURUSD", 0)
    return b


def test_buy_fills_at_next_open_and_hits_take_profit(eurusd):
    rows = [(1.0, 1.0, 1.0, 1.0), (1.0010, 1.0012, 1.0008, 1.0011), (1.0011, 1.0031, 1.0009, 1.0030)]
    b = _broker(rows, eurusd)
    res = b.open_position("EURUSD", Side.BUY, 1.0, sl=0.9990, tp=1.0030, magic=1)
    assert res.ok and res.price == pytest.approx(1.0010)
    b.advance("EURUSD")
    assert len(b.positions()) == 1
    b.advance("EURUSD")
    assert not b.positions()
    t = b.all_closed[0]
    assert t.reason == "tp" and t.profit == pytest.approx(200.0)
    assert b.account().balance == pytest.approx(10_200.0)


def test_stop_wins_when_both_levels_are_inside_one_bar(eurusd):
    rows = [(1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0), (1.0, 1.0030, 0.9980, 1.0)]
    b = _broker(rows, eurusd)
    b.open_position("EURUSD", Side.BUY, 1.0, sl=0.9990, tp=1.0020, magic=1)
    b.advance("EURUSD")
    b.advance("EURUSD")
    assert b.all_closed[0].reason == "sl"
    assert b.all_closed[0].profit == pytest.approx(-100.0)


def test_sell_closes_at_ask_and_gap_fills_at_open(eurusd):
    rows = [(1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0), (1.0030, 1.0035, 1.0025, 1.0030)]
    b = _broker(rows, eurusd, spread=10)
    res = b.open_position("EURUSD", Side.SELL, 0.5, sl=1.0010, tp=0.9980, magic=1)
    assert res.price == pytest.approx(1.0)  # sells fill at the bid
    b.advance("EURUSD")
    b.advance("EURUSD")  # gaps above the stop -> filled at the (ask) open, worse than the stop
    t = b.all_closed[0]
    assert t.reason == "sl"
    assert t.price_close == pytest.approx(1.0031)
    assert t.profit == pytest.approx(-155.0)


def test_rejects_invalid_stops(eurusd):
    b = _broker([(1.0,) * 4] * 3, eurusd)
    assert not b.open_position("EURUSD", Side.BUY, 1.0, sl=1.001, tp=1.002, magic=1).ok
