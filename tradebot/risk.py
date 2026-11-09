"""Account-level risk management: position sizing, loss limits, kill switch, adaptive throttle."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import pandas as pd

from .config import RiskConfig
from .types import SymbolInfo


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(math.floor(value / step + 1e-9) * step, decimals + 2)


@dataclass
class RiskState:
    day: Optional[str] = None
    day_start_equity: Optional[float] = None
    peak_equity: Optional[float] = None
    killed: bool = False
    kill_reason: str = ""
    consecutive_losses: int = 0
    cooldown_until: Optional[str] = None  # ISO timestamp, broker server time
    recent_r: Deque[float] = field(default_factory=deque)

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "day_start_equity": self.day_start_equity,
            "peak_equity": self.peak_equity,
            "killed": self.killed,
            "kill_reason": self.kill_reason,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until,
            "recent_r": list(self.recent_r),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RiskState":
        s = cls(**{k: v for k, v in d.items() if k != "recent_r"})
        s.recent_r = deque(d.get("recent_r", []))
        return s


class RiskManager:
    def __init__(self, cfg: RiskConfig, state: Optional[RiskState] = None):
        self.cfg = cfg
        self.state = state or RiskState()
        self.state.recent_r = deque(self.state.recent_r, maxlen=max(1, cfg.adaptive_window))

    # ------------------------------------------------------------------ equity tracking
    def update_equity(self, equity: float, now: pd.Timestamp) -> None:
        s = self.state
        day = now.strftime("%Y-%m-%d")
        if s.day != day:
            s.day = day
            s.day_start_equity = equity
        if s.peak_equity is None or equity > s.peak_equity:
            s.peak_equity = equity
        if s.peak_equity and equity <= s.peak_equity * (1.0 - self.cfg.max_drawdown_pct / 100.0) and not s.killed:
            s.killed = True
            s.kill_reason = (
                f"max drawdown {self.cfg.max_drawdown_pct}% reached "
                f"(peak {s.peak_equity:.2f}, equity {equity:.2f}); delete the risk state to resume"
            )

    def daily_loss_hit(self, equity: float) -> bool:
        s = self.state
        if s.day_start_equity is None:
            return False
        return equity <= s.day_start_equity * (1.0 - self.cfg.max_daily_loss_pct / 100.0)

    # ------------------------------------------------------------------ trade outcomes
    def record_trade(self, r_multiple: float, now: pd.Timestamp) -> None:
        s = self.state
        s.recent_r.append(float(r_multiple))
        if r_multiple < 0:
            s.consecutive_losses += 1
            if s.consecutive_losses >= self.cfg.max_consecutive_losses:
                until = now + pd.Timedelta(minutes=self.cfg.cooldown_minutes)
                s.cooldown_until = until.isoformat()
                s.consecutive_losses = 0
        else:
            s.consecutive_losses = 0

    def risk_multiplier(self) -> float:
        """Scale risk down after a losing streak; recovers automatically as results improve."""
        recent = list(self.state.recent_r)
        if not recent:
            return 1.0
        total = sum(recent)
        if total >= 0:
            return 1.0
        mult = 1.0 + total / (2.0 * max(1, self.cfg.adaptive_window))
        return max(self.cfg.min_risk_multiplier, min(1.0, mult))

    # ------------------------------------------------------------------ gates
    def can_trade(self, equity: float, now: pd.Timestamp) -> Tuple[bool, str]:
        s = self.state
        if s.killed:
            return False, f"kill switch: {s.kill_reason}"
        if self.daily_loss_hit(equity):
            return False, f"daily loss limit {self.cfg.max_daily_loss_pct}% reached"
        if s.cooldown_until is not None and now < pd.Timestamp(s.cooldown_until):
            return False, f"cooling down after {self.cfg.max_consecutive_losses} losses until {s.cooldown_until}"
        return True, ""

    def check_trade(
        self,
        info: SymbolInfo,
        spread: float,
        sl_dist: float,
        tp_dist: float,
        open_symbol: int,
        open_total: int,
    ) -> Tuple[bool, str]:
        c = self.cfg
        if open_symbol >= c.max_positions_per_symbol:
            return False, "position already open on symbol"
        if open_total >= c.max_total_positions:
            return False, "max total positions reached"
        if spread > c.max_spread_sl_fraction * sl_dist:
            return False, f"spread {spread / info.point:.0f}pts too wide vs SL"
        if c.max_spread_points is not None and spread / info.point > c.max_spread_points:
            return False, f"spread {spread / info.point:.0f}pts > max {c.max_spread_points}"
        min_dist = (info.stops_level + 1) * info.point + spread
        if sl_dist < min_dist or tp_dist < min_dist:
            return False, "SL/TP closer than broker stops level"
        return True, ""

    # ------------------------------------------------------------------ sizing
    def position_size(self, equity: float, info: SymbolInfo, sl_dist: float) -> Tuple[float, float]:
        """Return (lots, money at risk). Lots is 0 if even the minimum volume would risk too much."""
        if sl_dist <= 0 or equity <= 0:
            return 0.0, 0.0
        risk_money = equity * self.cfg.risk_per_trade_pct / 100.0 * self.risk_multiplier()
        loss_per_lot = sl_dist * info.value_per_price_unit
        if loss_per_lot <= 0:
            return 0.0, 0.0
        lots = floor_to_step(risk_money / loss_per_lot, info.volume_step)
        lots = min(lots, info.volume_max, self.cfg.max_lots)
        lots = floor_to_step(lots, info.volume_step)
        if lots < info.volume_min - 1e-12:
            # Small-account mode: the minimum lot is allowed up to the user's own cap.
            cap = self.cfg.min_lot_max_risk_pct
            min_risk = info.volume_min * loss_per_lot
            if cap is not None and min_risk <= equity * cap / 100.0 + 1e-9:
                return info.volume_min, min_risk
            return 0.0, 0.0
        return lots, lots * loss_per_lot
