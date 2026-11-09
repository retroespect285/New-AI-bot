"""Broker interface. The trading logic only talks to this, so live trading (MT5) and the
backtest simulator run exactly the same decision code."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import pandas as pd

from ..types import AccountInfo, ClosedTrade, OrderResult, Position, Side, SymbolInfo, Tick

BAR_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread"]


class BrokerError(RuntimeError):
    pass


class Broker(ABC):
    name = "broker"

    def connect(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def ensure_connected(self) -> bool:
        return True

    @abstractmethod
    def symbol_info(self, symbol: str) -> SymbolInfo: ...

    @abstractmethod
    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Last ``count`` *closed* bars, oldest first, columns ``BAR_COLUMNS`` (time = server time)."""

    @abstractmethod
    def get_tick(self, symbol: str) -> Tick: ...

    @abstractmethod
    def account(self) -> AccountInfo: ...

    @abstractmethod
    def positions(self, symbol: Optional[str] = None, magic: Optional[int] = None) -> List[Position]: ...

    @abstractmethod
    def open_position(self, symbol: str, side: Side, volume: float, sl: float, tp: float,
                      magic: int, comment: str = "", deviation: int = 20) -> OrderResult: ...

    @abstractmethod
    def modify_position(self, position: Position, sl: float, tp: float) -> OrderResult: ...

    @abstractmethod
    def close_position(self, position: Position, comment: str = "", deviation: int = 20) -> OrderResult: ...

    def margin_required(self, symbol: str, side: Side, volume: float, price: float) -> Optional[float]:
        """Margin (account currency) needed to open ``volume`` lots, or None if unknown."""
        return None

    @abstractmethod
    def closed_trades(self, since: pd.Timestamp, magic: int,
                      position_ids: Iterable[int] = ()) -> List[ClosedTrade]:
        """Positions closed at or after ``since`` (server time) that belong to this bot."""
