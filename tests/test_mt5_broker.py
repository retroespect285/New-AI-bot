import pandas as pd
import pytest

from tests import fake_mt5
from tradebot.brokers.mt5 import MT5Broker
from tradebot.config import MT5Config
from tradebot.types import Position, Side


@pytest.fixture
def broker():
    fake_mt5.state.reset()
    b = MT5Broker(MT5Config(login=123, password="pw", server="Demo-Server", path="C:/mt5/terminal64.exe"),
                  module=fake_mt5)
    b.connect()
    return b


def test_connect_passes_credentials(broker):
    args, kwargs = fake_mt5.state.initialized_with
    assert args == ("C:/mt5/terminal64.exe",)
    assert kwargs["login"] == 123 and kwargs["password"] == "pw" and kwargs["server"] == "Demo-Server"


def test_rates_drop_the_forming_bar(broker):
    df = broker.get_rates("EURUSD", "M5", 50)
    assert len(df) == 50
    assert list(df.columns) == ["time", "open", "high", "low", "close", "tick_volume", "spread"]
    assert df["time"].iat[-1] == pd.Timestamp(fake_mt5.state.now - 300, unit="s")


def test_symbol_info_mapping(broker):
    info = broker.symbol_info("EURUSD")
    assert info.point == 1e-5 and info.digits == 5 and info.stops_level == 10
    assert info.value_per_price_unit == pytest.approx(100000.0)


def test_open_position_builds_request_and_falls_back_on_filling_mode(broker):
    res = broker.open_position("EURUSD", Side.BUY, 0.37, sl=1.0990012, tp=1.1020049, magic=42,
                               comment="x" * 50, deviation=15)
    assert res.ok and res.ticket == 555
    first, second = fake_mt5.state.requests
    assert first["type_filling"] == fake_mt5.ORDER_FILLING_FOK  # rejected by the "broker"
    assert second["type_filling"] == fake_mt5.ORDER_FILLING_IOC
    assert second["type"] == fake_mt5.ORDER_TYPE_BUY and second["price"] == 1.10012
    assert second["sl"] == 1.0990 and second["tp"] == 1.1020  # rounded to the symbol's digits
    assert second["magic"] == 42 and second["deviation"] == 15 and len(second["comment"]) <= 31
    # the accepted filling mode is remembered for the next order
    broker.open_position("EURUSD", Side.SELL, 0.1, sl=1.11, tp=1.09, magic=42)
    assert fake_mt5.state.requests[-1]["type_filling"] == fake_mt5.ORDER_FILLING_IOC
    assert fake_mt5.state.requests[-1]["price"] == 1.10000


def test_close_and_modify(broker):
    pos = Position(ticket=9, symbol="EURUSD", side=Side.SELL, volume=0.2, price_open=1.1, sl=1.11, tp=1.09,
                   time_open=pd.Timestamp("2024-01-01"), magic=42)
    assert broker.close_position(pos, comment="time-exit").ok
    req = fake_mt5.state.requests[-1]
    assert req["position"] == 9 and req["type"] == fake_mt5.ORDER_TYPE_BUY and req["volume"] == 0.2
    assert broker.modify_position(pos, 1.1, 1.09).ok
    assert fake_mt5.state.requests[-1]["action"] == fake_mt5.TRADE_ACTION_SLTP


def test_positions_filtered_by_magic(broker):
    fake_mt5.state.positions = [
        fake_mt5.Pos(1, "EURUSD", 0, 42, 0.1, 1.1, 1.09, 1.12, 1_700_000_000, 5.0, ""),
        fake_mt5.Pos(2, "EURUSD", 1, 7, 0.1, 1.1, 1.11, 1.09, 1_700_000_000, -5.0, "manual"),
    ]
    mine = broker.positions("EURUSD", magic=42)
    assert [p.ticket for p in mine] == [1] and mine[0].side is Side.BUY


def test_demo_detection(broker):
    assert broker.account().is_demo
    fake_mt5.state.trade_mode = fake_mt5.ACCOUNT_TRADE_MODE_REAL
    assert not broker.account().is_demo


def test_closed_trades_aggregate_deals(broker):
    t_in, t_out = 1_700_000_000, 1_700_003_000
    fake_mt5.state.deals = [
        fake_mt5.Deal(1, 555, "EURUSD", fake_mt5.DEAL_TYPE_BUY, fake_mt5.DEAL_ENTRY_IN, 42, 0.3, 1.1, 0.0, -1.0, 0.0, 0.0, t_in, 3),
        fake_mt5.Deal(2, 555, "EURUSD", fake_mt5.DEAL_TYPE_SELL, fake_mt5.DEAL_ENTRY_OUT, 42, 0.3, 1.101, 30.0, -1.0, -0.5, 0.0, t_out, 5),
        fake_mt5.Deal(3, 999, "EURUSD", fake_mt5.DEAL_TYPE_BUY, fake_mt5.DEAL_ENTRY_OUT, 7, 0.1, 1.1, 99.0, 0.0, 0.0, 0.0, t_out, 0),
    ]
    trades = broker.closed_trades(pd.Timestamp(t_in, unit="s"), magic=42)
    assert len(trades) == 1
    t = trades[0]
    assert t.position_id == 555 and t.side is Side.BUY and t.reason == "tp"
    assert t.profit == pytest.approx(27.5)  # profit + commissions + swap


def test_wrong_tick_value_is_corrected_by_the_profit_calculator(broker, caplog):
    # e.g. a gold symbol advertising tick value 10.0 while real P/L is 1.0 per tick
    fake_mt5.state.tick_value = 10.0
    info = broker.symbol_info("EURUSD")
    assert info.tick_value == pytest.approx(1.0)
    assert info.value_per_price_unit == pytest.approx(100000.0)
    assert "profit calculator" in caplog.text


def test_correct_tick_value_is_kept(broker):
    fake_mt5.state.tick_value = 1.0
    assert broker.symbol_info("EURUSD").tick_value == 1.0
