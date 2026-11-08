"""Bar-replay broker used by the backtester.

Timeline: the engine is called right after bar ``i`` closes. At that moment the current price is the
open of bar ``i+1`` (bid) plus the bar's spread (ask), so market orders fill there. ``advance()``
then plays bar ``i+1``: stops and targets are checked against its high/low (if both could have been
hit inside the same bar the stop is assumed to have been hit first - the conservative choice).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..types import AccountInfo, ClosedTrade, OrderResult, Position, Side, SymbolInfo, Tick
from .base import Broker, BrokerError


class SimBroker(Broker):
    name = "simulator"

    def __init__(self, data: Dict[str, pd.DataFrame], infos: Dict[str, SymbolInfo],
                 initial_balance: float = 10_000.0, commission_per_lot: float = 0.0,
                 default_spread_points: float = 10.0, currency: str = "USD"):
        self.data = {s: df.reset_index(drop=True) for s, df in data.items()}
        self.infos = infos
        self.balance = float(initial_balance)
        self.commission_per_lot = commission_per_lot
        self.currency = currency
        self.idx: Dict[str, int] = {s: 0 for s in data}
        self._arr = {}
        for s, df in self.data.items():
            spr = df["spread"].to_numpy(dtype=float) if "spread" in df else np.full(len(df), default_spread_points)
            spr = np.where(np.isfinite(spr) & (spr > 0), spr, default_spread_points)
            self._arr[s] = {
                "time": pd.to_datetime(df["time"]).reset_index(drop=True),
                "open": df["open"].to_numpy(dtype=float),
                "high": df["high"].to_numpy(dtype=float),
                "low": df["low"].to_numpy(dtype=float),
                "close": df["close"].to_numpy(dtype=float),
                "spread": spr * infos[s].point,
            }
        self._positions: Dict[int, Position] = {}
        self._closed: List[ClosedTrade] = []
        self._next_ticket = 1

    # ------------------------------------------------------------------ clock
    def set_index(self, symbol: str, i: int) -> None:
        self.idx[symbol] = i

    def _now_index(self, symbol: str) -> int:
        """Index of the bar whose open is the 'current' price (the bar after the last closed one)."""
        return min(self.idx[symbol] + 1, len(self._arr[symbol]["open"]) - 1)

    def advance(self, symbol: str) -> None:
        """Play the next bar: trigger stops/targets, then move the clock forward one bar."""
        k = self.idx[symbol] + 1
        a = self._arr[symbol]
        if k >= len(a["open"]):
            raise BrokerError("end of data")
        o, h, lo, spr = a["open"][k], a["high"][k], a["low"][k], a["spread"][k]
        for pos in [p for p in self._positions.values() if p.symbol == symbol]:
            if pos.side is Side.BUY:
                hit_sl = pos.sl > 0 and lo <= pos.sl
                hit_tp = pos.tp > 0 and h >= pos.tp
                if hit_sl:
                    self._close(pos, min(o, pos.sl), a["time"][k], "sl")
                elif hit_tp:
                    self._close(pos, max(o, pos.tp), a["time"][k], "tp")
            else:
                hit_sl = pos.sl > 0 and h + spr >= pos.sl
                hit_tp = pos.tp > 0 and lo + spr <= pos.tp
                if hit_sl:
                    self._close(pos, max(o + spr, pos.sl), a["time"][k], "sl")
                elif hit_tp:
                    self._close(pos, min(o + spr, pos.tp), a["time"][k], "tp")
        self.idx[symbol] = k

    # ------------------------------------------------------------------ market data
    def symbol_info(self, symbol: str) -> SymbolInfo:
        return self.infos[symbol]

    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        i = self.idx[symbol]
        return self.data[symbol].iloc[max(0, i - count + 1): i + 1].reset_index(drop=True)

    def get_tick(self, symbol: str) -> Tick:
        a = self._arr[symbol]
        k = self._now_index(symbol)
        bid = a["open"][k] if k > self.idx[symbol] else a["close"][k]
        return Tick(bid=float(bid), ask=float(bid + a["spread"][k]), time=a["time"][k])

    # ------------------------------------------------------------------ account
    def _mark(self, pos: Position) -> float:
        a = self._arr[pos.symbol]
        i = self.idx[pos.symbol]
        bid = a["close"][i]
        px = bid if pos.side is Side.BUY else bid + a["spread"][i]
        info = self.infos[pos.symbol]
        return pos.side.value * (px - pos.price_open) * info.value_per_price_unit * pos.volume

    def account(self) -> AccountInfo:
        floating = sum(self._mark(p) for p in self._positions.values())
        equity = self.balance + floating
        return AccountInfo(login=0, balance=self.balance, equity=equity, margin_free=equity,
                           currency=self.currency, is_demo=True)

    def positions(self, symbol: Optional[str] = None, magic: Optional[int] = None) -> List[Position]:
        out = []
        for p in self._positions.values():
            if symbol is not None and p.symbol != symbol:
                continue
            if magic is not None and p.magic != magic:
                continue
            out.append(replace(p, profit=self._mark(p)))
        return out

    # ------------------------------------------------------------------ orders
    def open_position(self, symbol: str, side: Side, volume: float, sl: float, tp: float,
                      magic: int, comment: str = "", deviation: int = 20) -> OrderResult:
        info = self.infos[symbol]
        if volume < info.volume_min - 1e-12:
            return OrderResult(False, message="volume below minimum")
        tick = self.get_tick(symbol)
        price = tick.ask if side is Side.BUY else tick.bid
        if side is Side.BUY and not (sl < price < tp):
            return OrderResult(False, message="invalid stops")
        if side is Side.SELL and not (tp < price < sl):
            return OrderResult(False, message="invalid stops")
        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions[ticket] = Position(ticket=ticket, symbol=symbol, side=side, volume=volume,
                                           price_open=price, sl=sl, tp=tp, time_open=tick.time,
                                           magic=magic, comment=comment)
        return OrderResult(True, ticket=ticket, price=price, volume=volume, retcode=0, message="filled")

    def modify_position(self, position: Position, sl: float, tp: float) -> OrderResult:
        pos = self._positions.get(position.ticket)
        if pos is None:
            return OrderResult(False, message="position not found")
        self._positions[pos.ticket] = replace(pos, sl=sl, tp=tp)
        return OrderResult(True, ticket=pos.ticket)

    def close_position(self, position: Position, comment: str = "", deviation: int = 20) -> OrderResult:
        pos = self._positions.get(position.ticket)
        if pos is None:
            return OrderResult(False, message="position not found")
        tick = self.get_tick(pos.symbol)
        price = tick.bid if pos.side is Side.BUY else tick.ask
        self._close(pos, price, tick.time, comment or "close")
        return OrderResult(True, ticket=pos.ticket, price=price, volume=pos.volume)

    def _close(self, pos: Position, price: float, when, reason: str) -> None:
        info = self.infos[pos.symbol]
        gross = pos.side.value * (price - pos.price_open) * info.value_per_price_unit * pos.volume
        net = gross - self.commission_per_lot * pos.volume
        self.balance += net
        del self._positions[pos.ticket]
        self._closed.append(ClosedTrade(position_id=pos.ticket, symbol=pos.symbol, side=pos.side,
                                        volume=pos.volume, price_close=float(price),
                                        time_close=pd.Timestamp(when), profit=float(net), reason=reason))

    def closed_trades(self, since: pd.Timestamp, magic: int, position_ids: Iterable[int] = ()) -> List[ClosedTrade]:
        return [t for t in self._closed if t.time_close >= since]

    @property
    def all_closed(self) -> List[ClosedTrade]:
        return list(self._closed)
