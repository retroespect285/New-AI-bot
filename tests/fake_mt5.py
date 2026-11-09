"""A minimal in-memory stand-in for the ``MetaTrader5`` package (which only exists on Windows),
so the MT5 broker adapter can be tested anywhere."""
from collections import namedtuple

import numpy as np

TIMEFRAME_M1 = 1
TIMEFRAME_M5 = 5
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 16385
TIMEFRAME_H4 = 16388
_TF_SECONDS = {1: 60, 5: 300, 30: 1800, 16385: 3600, 16388: 14400}
TRADE_ACTION_DEAL = 1
TRADE_ACTION_SLTP = 6
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2
SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_INVALID_FILL = 10030
ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_REAL = 2
POSITION_TYPE_BUY = 0
DEAL_ENTRY_IN = 0
DEAL_ENTRY_OUT = 1
DEAL_TYPE_BUY = 0
DEAL_TYPE_SELL = 1

SymInfo = namedtuple("SymInfo", "point digits trade_tick_size trade_tick_value trade_contract_size volume_min "
                                "volume_max volume_step trade_stops_level trade_freeze_level filling_mode")
TickT = namedtuple("TickT", "bid ask time")
Acc = namedtuple("Acc", "login server balance equity margin_free currency trade_mode trade_allowed")
Term = namedtuple("Term", "connected trade_allowed")
Result = namedtuple("Result", "retcode order deal volume price comment")
Pos = namedtuple("Pos", "ticket symbol type magic volume price_open sl tp time profit comment")
SymT = namedtuple("SymT", "name description")
Deal = namedtuple("Deal", "ticket position_id symbol type entry magic volume price profit commission swap fee time reason")


class State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.initialized_with = None
        self.requests = []
        self.accepted_filling = ORDER_FILLING_IOC
        self.trade_mode = ACCOUNT_TRADE_MODE_DEMO
        self.positions = []
        self.deals = []
        self.filling_mode = SYMBOL_FILLING_FOK | SYMBOL_FILLING_IOC
        self.now = 1_700_000_100  # open time of the M5 bar that is currently forming
        self.margin_per_lot = 220.0
        self.tick_value = 1.0  # what symbol_info reports
        self.profit_per_unit = 100000.0  # what the profit calculator really pays per 1.0 move per lot
        self.symbols = [SymT("EURUSD", "Euro vs US Dollar"), SymT("XAUUSD", "Gold vs US Dollar"),
                        SymT("XAUUSDc", "Gold vs US Dollar (cent)"), SymT("US30", "Dow Jones")]
        self.init_ok = True
        self.equity = 10050.0


state = State()


def initialize(*args, **kwargs):
    state.initialized_with = (args, kwargs)
    return state.init_ok


def symbols_get(group=None):
    return tuple(state.symbols)


def shutdown():
    return None


def last_error():
    return (0, "ok")


def terminal_info():
    return Term(True, True)


def account_info():
    return Acc(123, "Demo-Server", state.equity, state.equity, state.equity * 0.9, "USD", state.trade_mode, True)


def order_calc_margin(action, symbol, volume, price):
    return state.margin_per_lot * volume


def symbol_select(symbol, enable):
    return True


def symbol_info(symbol):
    return SymInfo(1e-5, 5, 1e-5, state.tick_value, 100000.0, 0.01, 50.0, 0.01, 10, 0, state.filling_mode)


def order_calc_profit(action, symbol, volume, price_open, price_close):
    sign = 1 if action == ORDER_TYPE_BUY else -1
    return sign * (price_close - price_open) * state.profit_per_unit * volume


def symbol_info_tick(symbol):
    return TickT(1.10000, 1.10012, state.now + 5)


def copy_rates_from_pos(symbol, timeframe, start, count):
    dtype = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
             ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")]
    n = count
    sec = _TF_SECONDS[timeframe]
    t0 = (state.now // sec) * sec - (n - 1) * sec  # last element = the candle still forming
    arr = np.zeros(n, dtype=dtype)
    arr["time"] = t0 + np.arange(n) * sec
    wave = 0.002 * np.sin(arr["time"] / 7_000.0)
    arr["open"] = 1.1 + wave
    arr["close"] = 1.1 + wave + 0.0001
    arr["high"] = arr["close"] + 0.0004
    arr["low"] = arr["open"] - 0.0004
    arr["tick_volume"] = 100
    arr["spread"] = 12
    return arr


def order_send(request):
    state.requests.append(dict(request))
    if request["action"] == TRADE_ACTION_DEAL and request.get("type_filling") != state.accepted_filling:
        return Result(TRADE_RETCODE_INVALID_FILL, 0, 0, 0.0, 0.0, "Unsupported filling mode")
    return Result(TRADE_RETCODE_DONE, 555, 777, request.get("volume", 0.0), request.get("price", 0.0), "done")


def positions_get(symbol=None):
    return tuple(p for p in state.positions if symbol is None or p.symbol == symbol)


def history_deals_get(date_from, date_to):
    return tuple(state.deals)
