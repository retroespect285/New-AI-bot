"""Plain data types shared by brokers, the engine and the backtester."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import pandas as pd


class Side(IntEnum):
    BUY = 1
    SELL = -1

    @property
    def label(self) -> str:
        return "BUY" if self is Side.BUY else "SELL"

    @classmethod
    def from_label(cls, text: str) -> "Side":
        return cls.BUY if text.upper() == "BUY" else cls.SELL


@dataclass(frozen=True)
class SymbolInfo:
    """Contract specification needed for sizing and price rounding."""

    name: str
    point: float
    digits: int
    tick_size: float
    tick_value: float  # account-currency value of a one-tick move for 1.0 lot
    contract_size: float = 100_000.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    stops_level: int = 0  # minimum SL/TP distance from price, in points
    freeze_level: int = 0

    @property
    def value_per_price_unit(self) -> float:
        """Account-currency P/L of a 1.0 price-unit move for 1.0 lot."""
        return self.tick_value / self.tick_size

    def round_price(self, price: float) -> float:
        ticks = round(price / self.tick_size)
        return round(ticks * self.tick_size, self.digits)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "point": self.point,
            "digits": self.digits,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "contract_size": self.contract_size,
            "volume_min": self.volume_min,
            "volume_max": self.volume_max,
            "volume_step": self.volume_step,
            "stops_level": self.stops_level,
            "freeze_level": self.freeze_level,
        }


@dataclass(frozen=True)
class Tick:
    bid: float
    ask: float
    time: pd.Timestamp  # broker server time

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class Position:
    ticket: int
    symbol: str
    side: Side
    volume: float
    price_open: float
    sl: float
    tp: float
    time_open: pd.Timestamp  # broker server time
    profit: float = 0.0
    magic: int = 0
    comment: str = ""


@dataclass(frozen=True)
class AccountInfo:
    login: int
    balance: float
    equity: float
    margin_free: float
    currency: str
    is_demo: bool
    trade_allowed: bool = True


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    ticket: Optional[int] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    retcode: Optional[int] = None
    message: str = ""


@dataclass(frozen=True)
class ClosedTrade:
    position_id: int
    symbol: str
    side: Optional[Side]
    volume: float
    price_close: float
    time_close: pd.Timestamp
    profit: float  # net of commission, swap and fees
    reason: str = ""
