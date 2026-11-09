"""Dynamic stop management: trailing stops, reversal exits, broker rules."""
import numpy as np
import pandas as pd
import pytest

from tests.conftest import EURUSD, make_bars, small_config
from tradebot.brain import ArmEval, Decision
from tradebot.brokers.sim import SimBroker
from tradebot.config import TradeStyle
from tradebot.core import BotCore
from tradebot.data import synthetic_bars
from tradebot.journal import Journal
from tradebot.labels import Arm, label_arm, make_arms
from tradebot.types import Side, SymbolInfo

PT = 1e-5


def _trail_arm(side):
    return Arm(side, "M5", 1.0, 3.0, 6, trail_atr=1.0, trail_start_r=1.0)


def test_trailing_label_locks_in_profit_and_never_loosens():
    rows = [(1.0,) * 4] * 3 + [
        (1.0, 1.0012, 0.9995, 1.0010),     # +1.2R seen -> stop trails to 1.0002
        (1.0010, 1.0020, 1.0008, 1.0018),  # new high 1.0020 -> stop 1.0010
        (1.0018, 1.0019, 1.0011, 1.0012),  # dips but above 1.0010; stop stays (never loosens)
        (1.0012, 1.0013, 1.0005, 1.0006),  # hits 1.0010 -> +1.0R
    ] + [(1.0006,) * 4] * 4
    bars = make_bars(rows)
    atr = np.full(len(bars), 0.001)
    lab = label_arm(bars, atr, _trail_arm(Side.BUY), 6, PT, default_spread_points=0)
    assert lab.win[2] == 1.0 and lab.loss[2] == 0.0 and lab.r[2] == pytest.approx(1.0)
    # the same path with a fixed stop just times out
    fixed = label_arm(bars, atr, Arm(Side.BUY, "M5", 1.0, 3.0, 6), 6, PT, default_spread_points=0)
    assert fixed.win[2] == 0.0 and fixed.loss[2] == 0.0


def test_trailing_label_before_activation_is_a_normal_loss():
    rows = [(1.0,) * 4] * 3 + [(1.0, 1.0005, 0.9985, 0.999)] + [(0.999,) * 4] * 6
    bars = make_bars(rows)
    lab = label_arm(bars, np.full(len(bars), 0.001), _trail_arm(Side.BUY), 6, PT, default_spread_points=0)
    assert lab.loss[2] == 1.0 and lab.r[2] == pytest.approx(-1.0)


def _core(tmp_path, styles, n=1500, seed=11, reversal=False):
    cfg = small_config()
    cfg.model.styles = styles
    cfg.model.slippage_points = 0.0
    cfg.risk.max_spread_sl_fraction = 1.0
    cfg.risk.exit_on_reversal = reversal
    cfg.validate()
    bars = synthetic_bars(n, seed=seed)
    sim = SimBroker({"EURUSD": bars}, {"EURUSD": EURUSD}, initial_balance=100_000)
    core = BotCore(cfg, sim, journal=Journal(str(tmp_path / f"logs{seed}")))
    core.add_symbol("EURUSD")
    return cfg, sim, core, core.prepare("EURUSD", bars)


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_live_trailing_matches_what_the_brain_learned(tmp_path, monkeypatch, side):
    """Trade a trailing style through the real cycle (stop moved every candle on the simulated
    broker) and compare the result with the outcome the learning simulation predicted."""
    style = TradeStyle("M5", 1.0, 3.0, 24, trail_atr=1.0, trail_start_r=0.5)
    cfg, sim, core, prep = _core(tmp_path, [style])
    arm = [a for a in make_arms(cfg.model.styles) if a.side is side][0]
    brain = core.brain("EURUSD", prep.features.names)
    trailed = 0
    for t in range(400, 1300, 45):
        if not prep.features.valid[t] or sim.positions():
            continue
        state = {"fired": False}

        def decide(*a, _t=t, **k):
            if not state["fired"]:
                state["fired"] = True
                return Decision(arm, prob=0.6, ev_r=0.3, reason="signal")
            return Decision(None, reason="test")

        monkeypatch.setattr(brain, "decide", decide)
        sim.set_index("EURUSD", t)
        core.on_bar("EURUSD", prep, t)
        (pos,) = sim.positions()
        risk = core.open_risk[pos.ticket]["risk_money"]
        sl0 = pos.sl
        k = t
        while sim.positions() and k < t + arm.horizon + 3:
            sim.advance("EURUSD")
            k += 1
            core.on_bar("EURUSD", prep, k)
            if sim.positions() and sim.positions()[0].sl != sl0:
                trailed += 1
        closed = sim.all_closed[-1]
        realised_r = closed.profit / risk
        expected_r = prep.labels[arm.key].r[t]
        assert realised_r == pytest.approx(expected_r, abs=0.06), (t, closed.reason)
        sim.set_index("EURUSD", k)
    assert trailed > 0  # the stop really was moved during some trades


def test_reversal_forecast_closes_the_trade(tmp_path, monkeypatch):
    styles = [TradeStyle("M5", 1.5, 3.0, 24)]
    cfg, sim, core, prep = _core(tmp_path, styles, reversal=True)
    buy, sell = make_arms(cfg.model.styles)
    brain = core.brain("EURUSD", prep.features.names)
    monkeypatch.setattr(brain, "decide", lambda *a, **k: Decision(buy, prob=0.6, ev_r=0.3, reason="signal"))
    sim.set_index("EURUSD", 600)
    core.on_bar("EURUSD", prep, 600)
    assert sim.positions()[0].side is Side.BUY
    opposite = ArmEval(sell, 0.6, 0.2, 0.3, 0.02, True, True, 30, 0.2, True)
    monkeypatch.setattr(brain, "decide", lambda *a, **k: Decision(sell, prob=0.6, ev_r=0.3, reason="signal",
                                                                  evals=[opposite]))
    sim.advance("EURUSD")
    core.on_bar("EURUSD", prep, 601)
    assert sim.all_closed[-1].reason == "ai:reversal"
    assert sim.positions()[0].side is Side.SELL  # and it took the opposite trade


def test_stop_moves_respect_broker_rules(tmp_path):
    info = SymbolInfo(name="XAUUSD", point=0.01, digits=2, tick_size=0.01, tick_value=1.0, contract_size=100,
                      stops_level=20)
    bars = make_bars([(2000.0, 2001.0, 1999.0, 2000.0)] * 5)
    cfg = small_config()
    sim = SimBroker({"XAUUSD": bars}, {"XAUUSD": info})
    core = BotCore(cfg, sim, journal=Journal(str(tmp_path / "l")))
    core.add_symbol("XAUUSD")
    sim.set_index("XAUUSD", 1)
    sim.open_position("XAUUSD", Side.BUY, 0.1, sl=1995.0, tp=2010.0, magic=cfg.trading.magic)
    (pos,) = sim.positions()
    tick = sim.get_tick("XAUUSD")  # bid 2000.00
    now = pd.Timestamp("2024-01-01 00:10")
    core._move_stop("XAUUSD", pos, 1994.0, "trail", tick, info, now)  # looser: ignored
    assert sim.positions()[0].sl == 1995.0
    core._move_stop("XAUUSD", pos, 1998.004, "trail", tick, info, now)  # tighter: moved, rounded
    assert sim.positions()[0].sl == 1998.0
    pos = sim.positions()[0]
    core._move_stop("XAUUSD", pos, 1999.95, "trail", tick, info, now)  # inside the 20-point stops level
    assert not sim.positions() and sim.all_closed[-1].reason == "ai:trail-stop"
