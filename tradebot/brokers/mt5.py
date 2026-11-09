"""MetaTrader 5 broker, using the official ``MetaTrader5`` Python package.

Requirements: Windows (the package talks to a locally installed MT5 terminal), the terminal logged
in to your broker, and *Algo Trading* enabled in the terminal toolbar.
"""
from __future__ import annotations

import importlib
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..config import MT5Config
from ..types import AccountInfo, ClosedTrade, OrderResult, Position, Side, SymbolInfo, Tick
from .base import BAR_COLUMNS, Broker, BrokerError

log = logging.getLogger(__name__)

# Numeric values from the MQL5 docs, used if the package does not expose a constant.
_DEFAULTS = {
    "TRADE_ACTION_DEAL": 1,
    "TRADE_ACTION_SLTP": 6,
    "ORDER_TYPE_BUY": 0,
    "ORDER_TYPE_SELL": 1,
    "ORDER_TIME_GTC": 0,
    "ORDER_FILLING_FOK": 0,
    "ORDER_FILLING_IOC": 1,
    "ORDER_FILLING_RETURN": 2,
    "SYMBOL_FILLING_FOK": 1,
    "SYMBOL_FILLING_IOC": 2,
    "TRADE_RETCODE_PLACED": 10008,
    "TRADE_RETCODE_DONE": 10009,
    "TRADE_RETCODE_DONE_PARTIAL": 10010,
    "TRADE_RETCODE_INVALID_FILL": 10030,
    "TRADE_RETCODE_REQUOTE": 10004,
    "TRADE_RETCODE_PRICE_CHANGED": 10020,
    "TRADE_RETCODE_PRICE_OFF": 10021,
    "DEAL_ENTRY_IN": 0,
    "DEAL_ENTRY_OUT": 1,
    "DEAL_ENTRY_INOUT": 2,
    "DEAL_ENTRY_OUT_BY": 3,
    "DEAL_TYPE_BUY": 0,
    "DEAL_TYPE_SELL": 1,
    "ACCOUNT_TRADE_MODE_DEMO": 0,
    "ACCOUNT_TRADE_MODE_CONTEST": 1,
    "ACCOUNT_TRADE_MODE_REAL": 2,
    "POSITION_TYPE_BUY": 0,
    "POSITION_TYPE_SELL": 1,
}
_DEAL_REASONS = {0: "manual", 1: "manual", 2: "manual", 3: "bot", 4: "sl", 5: "tp", 6: "stop_out"}
_COMMENT_MAX = 31


class MT5Broker(Broker):
    name = "mt5"

    def __init__(self, cfg: MT5Config, module=None):
        self.cfg = cfg
        self._mt5 = module
        self._info_cache: Dict[str, SymbolInfo] = {}
        self._filling_cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def mt5(self):
        if self._mt5 is None:
            try:
                self._mt5 = importlib.import_module("MetaTrader5")
            except ImportError as exc:  # pragma: no cover - depends on platform
                raise BrokerError(
                    "The MetaTrader5 package is not installed. It only exists for Windows: "
                    "run `pip install MetaTrader5` on the Windows machine that runs the MT5 terminal."
                ) from exc
        return self._mt5

    def c(self, name: str) -> int:
        return getattr(self.mt5, name, _DEFAULTS[name])

    def _err(self) -> str:
        try:
            return str(self.mt5.last_error())
        except Exception:  # pragma: no cover
            return "unknown error"

    # ------------------------------------------------------------------ connection
    def connect(self) -> None:
        kwargs = {"timeout": int(self.cfg.timeout_ms)}
        if self.cfg.path:
            kwargs["path"] = self.cfg.path
        if self.cfg.login:
            kwargs["login"] = int(self.cfg.login)
            if self.cfg.password:
                kwargs["password"] = str(self.cfg.password)
            if self.cfg.server:
                kwargs["server"] = str(self.cfg.server)
        path = kwargs.pop("path", None)
        ok = self.mt5.initialize(path, **kwargs) if path else self.mt5.initialize(**kwargs)
        if not ok:
            raise BrokerError(f"MT5 initialize() failed: {self._err()}. Is the terminal installed and logged in?")
        term = self.mt5.terminal_info()
        acc = self.mt5.account_info()
        if acc is None:
            raise BrokerError(f"MT5 connected but no account is logged in: {self._err()}")
        if term is not None and not getattr(term, "trade_allowed", True):
            log.warning("Algo Trading is DISABLED in the MT5 terminal - orders will be rejected. "
                        "Click the 'Algo Trading' button in the terminal toolbar.")
        log.info("Connected to MT5: account %s on %s (%s)", acc.login, getattr(acc, "server", "?"),
                 "DEMO" if self._is_demo(acc) else "REAL")

    def shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:  # pragma: no cover
            pass

    def ensure_connected(self) -> bool:
        term = self.mt5.terminal_info()
        if term is not None and getattr(term, "connected", True):
            return True
        log.warning("MT5 terminal disconnected (%s); reconnecting...", self._err())
        self.shutdown()
        time.sleep(1.0)
        try:
            self.connect()
            return True
        except BrokerError as exc:
            log.error("Reconnect failed: %s", exc)
            return False

    def _is_demo(self, acc) -> bool:
        return getattr(acc, "trade_mode", 0) != self.c("ACCOUNT_TRADE_MODE_REAL")

    # ------------------------------------------------------------------ market data
    def _timeframe(self, timeframe: str) -> int:
        tf = getattr(self.mt5, f"TIMEFRAME_{timeframe}", None)
        if tf is None:
            raise BrokerError(f"Unknown timeframe {timeframe}")
        return tf

    def symbol_info(self, symbol: str) -> SymbolInfo:
        if not self.mt5.symbol_select(symbol, True):
            raise BrokerError(f"Symbol {symbol} not available: {self._err()}")
        si = self.mt5.symbol_info(symbol)
        if si is None:
            raise BrokerError(f"symbol_info({symbol}) failed: {self._err()}")
        info = SymbolInfo(
            name=symbol,
            point=float(si.point),
            digits=int(si.digits),
            tick_size=float(si.trade_tick_size or si.point),
            tick_value=float(si.trade_tick_value),
            contract_size=float(si.trade_contract_size),
            volume_min=float(si.volume_min),
            volume_max=float(si.volume_max),
            volume_step=float(si.volume_step),
            stops_level=int(si.trade_stops_level),
            freeze_level=int(getattr(si, "trade_freeze_level", 0)),
        )
        self._info_cache[symbol] = info
        modes = []
        fm = int(getattr(si, "filling_mode", 0))
        if fm & self.c("SYMBOL_FILLING_FOK"):
            modes.append(self.c("ORDER_FILLING_FOK"))
        if fm & self.c("SYMBOL_FILLING_IOC"):
            modes.append(self.c("ORDER_FILLING_IOC"))
        modes.append(self.c("ORDER_FILLING_RETURN"))
        self._filling_cache[symbol] = modes
        if info.tick_value <= 0:
            raise BrokerError(f"{symbol}: tick value is 0 - open a chart for it in MT5 and retry")
        return info

    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos(symbol, self._timeframe(timeframe), 0, int(count) + 1)
        if rates is None or len(rates) == 0:
            raise BrokerError(f"No bars for {symbol} {timeframe}: {self._err()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s").astype("datetime64[ns]")
        for col in BAR_COLUMNS:
            if col not in df:
                df[col] = 0
        # the last bar is still forming - only closed bars are used for decisions
        return df[BAR_COLUMNS].iloc[:-1].reset_index(drop=True)

    def get_tick(self, symbol: str) -> Tick:
        t = self.mt5.symbol_info_tick(symbol)
        if t is None:
            raise BrokerError(f"No tick for {symbol}: {self._err()}")
        return Tick(bid=float(t.bid), ask=float(t.ask), time=pd.Timestamp(int(t.time), unit="s"))

    # ------------------------------------------------------------------ account
    def account(self) -> AccountInfo:
        a = self.mt5.account_info()
        if a is None:
            raise BrokerError(f"account_info() failed: {self._err()}")
        return AccountInfo(
            login=int(a.login), balance=float(a.balance), equity=float(a.equity),
            margin_free=float(a.margin_free), currency=str(a.currency), is_demo=self._is_demo(a),
            trade_allowed=bool(getattr(a, "trade_allowed", True)),
        )

    def positions(self, symbol: Optional[str] = None, magic: Optional[int] = None) -> List[Position]:
        raw = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        if raw is None:
            return []
        out = []
        for p in raw:
            if magic is not None and int(p.magic) != int(magic):
                continue
            side = Side.BUY if int(p.type) == self.c("POSITION_TYPE_BUY") else Side.SELL
            out.append(Position(
                ticket=int(p.ticket), symbol=str(p.symbol), side=side, volume=float(p.volume),
                price_open=float(p.price_open), sl=float(p.sl), tp=float(p.tp),
                time_open=pd.Timestamp(int(p.time), unit="s"), profit=float(p.profit),
                magic=int(p.magic), comment=str(getattr(p, "comment", "")),
            ))
        return out

    # ------------------------------------------------------------------ orders
    def _fillings(self, symbol: str) -> List[int]:
        if symbol not in self._filling_cache:
            self.symbol_info(symbol)
        return self._filling_cache[symbol]

    def _send(self, request: dict, symbol: str) -> OrderResult:
        done = {self.c("TRADE_RETCODE_DONE"), self.c("TRADE_RETCODE_PLACED"), self.c("TRADE_RETCODE_DONE_PARTIAL")}
        retry_price = {self.c("TRADE_RETCODE_REQUOTE"), self.c("TRADE_RETCODE_PRICE_CHANGED"), self.c("TRADE_RETCODE_PRICE_OFF")}
        result = None
        modes = self._fillings(symbol)
        for attempt in range(3):
            for mode in modes:
                request["type_filling"] = mode
                result = self.mt5.order_send(request)
                if result is None:
                    return OrderResult(False, message=f"order_send failed: {self._err()}")
                if result.retcode != self.c("TRADE_RETCODE_INVALID_FILL"):
                    break
            if result.retcode in done:
                if mode != modes[0]:  # remember the filling mode this broker accepts
                    self._filling_cache[symbol] = [mode] + [m for m in modes if m != mode]
                return OrderResult(True, ticket=int(result.order) or None, price=float(result.price),
                                   volume=float(result.volume), retcode=int(result.retcode),
                                   message=str(result.comment))
            if result.retcode in retry_price and request.get("action") == self.c("TRADE_ACTION_DEAL"):
                tick = self.get_tick(symbol)
                buy = request["type"] == self.c("ORDER_TYPE_BUY")
                request["price"] = tick.ask if buy else tick.bid
                continue
            break
        return OrderResult(False, retcode=int(result.retcode), message=f"rejected: {result.comment} (retcode {result.retcode})")

    def open_position(self, symbol: str, side: Side, volume: float, sl: float, tp: float,
                      magic: int, comment: str = "", deviation: int = 20) -> OrderResult:
        info = self._info_cache.get(symbol) or self.symbol_info(symbol)
        tick = self.get_tick(symbol)
        request = {
            "action": self.c("TRADE_ACTION_DEAL"),
            "symbol": symbol,
            "volume": float(volume),
            "type": self.c("ORDER_TYPE_BUY") if side is Side.BUY else self.c("ORDER_TYPE_SELL"),
            "price": tick.ask if side is Side.BUY else tick.bid,
            "sl": info.round_price(sl),
            "tp": info.round_price(tp),
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": comment[:_COMMENT_MAX],
            "type_time": self.c("ORDER_TIME_GTC"),
        }
        return self._send(request, symbol)

    def modify_position(self, position: Position, sl: float, tp: float) -> OrderResult:
        info = self._info_cache.get(position.symbol) or self.symbol_info(position.symbol)
        request = {
            "action": self.c("TRADE_ACTION_SLTP"),
            "position": int(position.ticket),
            "symbol": position.symbol,
            "sl": info.round_price(sl),
            "tp": info.round_price(tp),
            "magic": int(position.magic),
        }
        result = self.mt5.order_send(request)
        if result is None:
            return OrderResult(False, message=f"order_send failed: {self._err()}")
        ok = result.retcode == self.c("TRADE_RETCODE_DONE")
        return OrderResult(ok, ticket=position.ticket, retcode=int(result.retcode), message=str(result.comment))

    def close_position(self, position: Position, comment: str = "", deviation: int = 20) -> OrderResult:
        tick = self.get_tick(position.symbol)
        closing_buy = position.side is Side.SELL
        request = {
            "action": self.c("TRADE_ACTION_DEAL"),
            "symbol": position.symbol,
            "volume": float(position.volume),
            "type": self.c("ORDER_TYPE_BUY") if closing_buy else self.c("ORDER_TYPE_SELL"),
            "position": int(position.ticket),
            "price": tick.ask if closing_buy else tick.bid,
            "deviation": int(deviation),
            "magic": int(position.magic),
            "comment": comment[:_COMMENT_MAX],
            "type_time": self.c("ORDER_TIME_GTC"),
        }
        return self._send(request, position.symbol)

    def margin_required(self, symbol: str, side: Side, volume: float, price: float) -> Optional[float]:
        order_type = self.c("ORDER_TYPE_BUY") if side is Side.BUY else self.c("ORDER_TYPE_SELL")
        try:
            margin = self.mt5.order_calc_margin(order_type, symbol, float(volume), float(price))
        except Exception:  # pragma: no cover - older terminals
            return None
        return None if margin is None else float(margin)

    # ------------------------------------------------------------------ history
    def closed_trades(self, since: pd.Timestamp, magic: int, position_ids: Iterable[int] = ()) -> List[ClosedTrade]:
        known = {int(p) for p in position_ids}
        # Deal times are broker server time; query a generous window so time-zone offsets between
        # this PC and the server can never hide a deal, then filter precisely below.
        start = datetime(since.year, since.month, since.day, tzinfo=timezone.utc) - timedelta(days=2)
        end = datetime.now(timezone.utc) + timedelta(days=2)
        deals = self.mt5.history_deals_get(start, end)
        if deals is None:
            return []
        out_entries = {self.c("DEAL_ENTRY_OUT"), self.c("DEAL_ENTRY_OUT_BY"), self.c("DEAL_ENTRY_INOUT")}
        by_pos: Dict[int, list] = {}
        for d in deals:
            by_pos.setdefault(int(d.position_id), []).append(d)
        trades = []
        for pid, ds in by_pos.items():
            if pid == 0:
                continue
            ours = pid in known or any(int(d.magic) == int(magic) for d in ds)
            outs = [d for d in ds if int(d.entry) in out_entries]
            if not ours or not outs:
                continue
            last = max(outs, key=lambda d: int(d.time))
            t_close = pd.Timestamp(int(last.time), unit="s")
            if t_close < since:
                continue
            ins = [d for d in ds if int(d.entry) == self.c("DEAL_ENTRY_IN")]
            if ins:
                side = Side.BUY if int(ins[0].type) == self.c("DEAL_TYPE_BUY") else Side.SELL
            else:  # closing deal is the opposite direction of the position
                side = Side.SELL if int(last.type) == self.c("DEAL_TYPE_BUY") else Side.BUY
            net = sum(float(d.profit) + float(d.commission) + float(d.swap) + float(getattr(d, "fee", 0.0)) for d in ds)
            trades.append(ClosedTrade(
                position_id=pid, symbol=str(last.symbol), side=side,
                volume=sum(float(d.volume) for d in outs), price_close=float(last.price),
                time_close=t_close, profit=net, reason=_DEAL_REASONS.get(int(getattr(last, "reason", -1)), "close"),
            ))
        return trades
