import pandas as pd

from tradebot.config import RiskConfig
from tradebot.risk import RiskManager, RiskState, floor_to_step

NOW = pd.Timestamp("2024-03-05 10:00")


def test_position_size_risks_the_configured_fraction(eurusd):
    rm = RiskManager(RiskConfig(risk_per_trade_pct=1.0))
    # 10,000 equity, 1% = 100 at risk; SL 20 pips = 0.0020 -> $200 per lot -> 0.5 lots
    lots, risk = rm.position_size(10_000, eurusd, 0.0020)
    assert lots == 0.5
    assert abs(risk - 100.0) < 1e-6


def test_position_size_rounds_down_and_respects_minimum(eurusd):
    rm = RiskManager(RiskConfig(risk_per_trade_pct=0.5))
    lots, _ = rm.position_size(1_000, eurusd, 0.0033)  # 5 / 330 = 0.01515 -> 0.01
    assert lots == 0.01
    lots, _ = rm.position_size(100, eurusd, 0.0050)  # 0.5 / 500 -> below 0.01 -> skip
    assert lots == 0.0


def test_floor_to_step():
    assert floor_to_step(0.0199999, 0.01) == 0.01
    assert floor_to_step(0.03, 0.01) == 0.03
    assert floor_to_step(1.29, 0.1) == 1.2
    assert floor_to_step(7.9, 1.0) == 7.0


def test_daily_loss_limit_and_reset_next_day():
    rm = RiskManager(RiskConfig(max_daily_loss_pct=3.0))
    rm.update_equity(10_000, NOW)
    rm.update_equity(9_690, NOW + pd.Timedelta(hours=2))
    ok, why = rm.can_trade(9_690, NOW + pd.Timedelta(hours=2))
    assert not ok and "daily loss" in why
    rm.update_equity(9_690, NOW + pd.Timedelta(days=1))
    assert rm.can_trade(9_690, NOW + pd.Timedelta(days=1))[0]


def test_drawdown_kill_switch_is_sticky():
    rm = RiskManager(RiskConfig(max_drawdown_pct=10.0, max_daily_loss_pct=50))
    rm.update_equity(10_000, NOW)
    rm.update_equity(12_000, NOW + pd.Timedelta(days=1))
    rm.update_equity(10_700, NOW + pd.Timedelta(days=2))
    ok, why = rm.can_trade(10_700, NOW + pd.Timedelta(days=2))
    assert not ok and "kill switch" in why
    rm.update_equity(13_000, NOW + pd.Timedelta(days=3))
    assert not rm.can_trade(13_000, NOW + pd.Timedelta(days=3))[0]


def test_cooldown_after_consecutive_losses():
    rm = RiskManager(RiskConfig(max_consecutive_losses=3, cooldown_minutes=60))
    for k in range(3):
        rm.record_trade(-1.0, NOW + pd.Timedelta(minutes=k))
    assert not rm.can_trade(10_000, NOW + pd.Timedelta(minutes=30))[0]
    assert rm.can_trade(10_000, NOW + pd.Timedelta(minutes=70))[0]


def test_risk_shrinks_after_losses_and_recovers():
    rm = RiskManager(RiskConfig(adaptive_window=10, min_risk_multiplier=0.25))
    assert rm.risk_multiplier() == 1.0
    for _ in range(4):
        rm.record_trade(-1.0, NOW)
    assert rm.risk_multiplier() == 0.8
    for _ in range(10):
        rm.record_trade(1.0, NOW)
    assert rm.risk_multiplier() == 1.0


def test_trade_checks(eurusd):
    rm = RiskManager(RiskConfig(max_spread_sl_fraction=0.2))
    assert rm.check_trade(eurusd, 0.0001, 0.0010, 0.0015, 0, 0)[0]
    assert not rm.check_trade(eurusd, 0.0003, 0.0010, 0.0015, 0, 0)[0]  # spread too wide
    assert not rm.check_trade(eurusd, 0.0001, 0.0010, 0.0015, 1, 1)[0]  # already in a trade


def test_state_roundtrip():
    rm = RiskManager(RiskConfig())
    rm.update_equity(10_000, NOW)
    rm.record_trade(-1.0, NOW)
    restored = RiskManager(RiskConfig(), RiskState.from_dict(rm.state.to_dict()))
    assert restored.state.day_start_equity == 10_000
    assert list(restored.state.recent_r) == [-1.0]
