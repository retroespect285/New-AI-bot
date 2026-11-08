import json

import pandas as pd
import pytest

from tests import fake_mt5
from tests.conftest import EURUSD, small_config
from tradebot.brain import Decision
from tradebot.brokers.base import BrokerError
from tradebot.brokers.mt5 import MT5Broker
from tradebot.brokers.sim import SimBroker
from tradebot.core import BotCore
from tradebot.data import synthetic_bars
from tradebot.engine import LiveEngine
from tradebot.journal import Journal
from tradebot.labels import make_arms


def _core(tmp_path, dry_run=False, n=1200, state_dir=None):
    cfg = small_config()
    cfg.risk.max_spread_sl_fraction = 1.0  # the synthetic spreads are wide; don't let that block the tests
    bars = synthetic_bars(n, seed=11)
    sim = SimBroker({"EURUSD": bars}, {"EURUSD": EURUSD}, initial_balance=10_000)
    core = BotCore(cfg, sim, journal=Journal(str(tmp_path / "logs")), state_dir=state_dir, dry_run=dry_run)
    core.add_symbol("EURUSD")
    prep = core.prepare("EURUSD", bars)
    return cfg, sim, core, prep


def _patch_decision(monkeypatch, core, prep, arm):
    brain = core.brain("EURUSD", prep.features.names)
    monkeypatch.setattr(brain, "decide", lambda *a, **k: Decision(arm, prob=0.6, ev_r=0.3, reason="signal"))


def test_executes_signal_with_risk_based_size_and_atr_stops(tmp_path, monkeypatch):
    cfg, sim, core, prep = _core(tmp_path)
    arm = make_arms(cfg.model.exit_profiles)[0]  # BUY sl1 tp1.5
    _patch_decision(monkeypatch, core, prep, arm)
    i = 700
    sim.set_index("EURUSD", i)
    core.on_bar("EURUSD", prep, i)
    (pos,) = sim.positions("EURUSD")
    atr = prep.features.atr[i]
    assert pos.price_open - pos.sl == pytest.approx(arm.sl_atr * atr, abs=2e-5)
    assert pos.tp - pos.price_open == pytest.approx(arm.tp_atr * atr, abs=2e-5)
    risk = (pos.price_open - pos.sl) * EURUSD.value_per_price_unit * pos.volume
    assert risk <= 10_000 * cfg.risk.risk_per_trade_pct / 100 + 1e-6
    assert risk > 10_000 * cfg.risk.risk_per_trade_pct / 100 * 0.8
    # while that position is open, further signals are blocked: one position per symbol
    for k in range(1, 4):
        sim.advance("EURUSD")
        still_open = bool(sim.positions("EURUSD"))
        d = core.on_bar("EURUSD", prep, i + k)
        if still_open:
            assert d.reason.startswith("blocked")
        assert len(sim.positions("EURUSD")) <= 1


def test_position_is_closed_after_max_holding_time(tmp_path, monkeypatch):
    cfg, sim, core, prep = _core(tmp_path)
    arm = make_arms(cfg.model.exit_profiles)[2]  # BUY sl1.5 tp3 - unlikely to be hit quickly
    _patch_decision(monkeypatch, core, prep, arm)
    i = 700
    sim.set_index("EURUSD", i)
    core.on_bar("EURUSD", prep, i)
    brain = core.brain("EURUSD", prep.features.names)
    monkeypatch.setattr(brain, "decide", lambda *a, **k: Decision(None, reason="test"))
    for k in range(1, cfg.model.horizon_bars + 2):
        sim.advance("EURUSD")
        core.on_bar("EURUSD", prep, i + k)
    assert not sim.positions("EURUSD")
    reasons = [t.reason for t in sim.all_closed]
    assert len(reasons) == 1 and reasons[0] in ("sl", "tp", "ai:time-exit")
    # the real result (in R) is fed back into that setup's track record
    core.sync_closed(pd.Timestamp(prep.times[i + cfg.model.horizon_bars + 2]))
    track = brain.track[arm.key]
    assert track.n == 1  # no paper trades yet: the model is still learning
    assert track.recent[-1] == pytest.approx(sim.all_closed[0].profit / 50.0, rel=0.05)


def test_dry_run_never_sends_orders(tmp_path, monkeypatch):
    cfg, sim, core, prep = _core(tmp_path, dry_run=True)
    _patch_decision(monkeypatch, core, prep, make_arms(cfg.model.exit_profiles)[1])
    sim.set_index("EURUSD", 700)
    d = core.on_bar("EURUSD", prep, 700)
    core.journal.close()
    assert not sim.positions() and d.reason == "dry run"
    orders = pd.read_csv(tmp_path / "logs" / "orders.csv")
    assert list(orders["status"]) == ["dry_run"]


def test_state_is_saved_and_resumed(tmp_path):
    state = tmp_path / "state"
    cfg, sim, core, prep = _core(tmp_path, state_dir=str(state))
    core.learn("EURUSD", prep, 1100)
    core.risk.record_trade(-1.0, pd.Timestamp("2024-01-05"))
    core.save_state()
    n = core.brains["EURUSD"].n_learned
    assert n > 0
    core2 = BotCore(cfg, sim, state_dir=str(state))
    core2.add_symbol("EURUSD")
    brain2 = core2.brain("EURUSD", prep.features.names)
    assert brain2.n_learned == n
    assert list(core2.risk.state.recent_r) == [-1.0]
    # catching up only learns what is new
    assert core2.learn("EURUSD", prep, 1100) == 0
    assert json.loads((state / "bot_state.json").read_text())["risk"]["recent_r"] == [-1.0]


def _live_cfg(tmp_path):
    cfg = small_config()
    cfg.trading.warmup_bars = 900
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.paths.log_dir = str(tmp_path / "logs")
    return cfg


def test_live_engine_refuses_real_money_account(tmp_path):
    fake_mt5.state.reset()
    fake_mt5.state.trade_mode = fake_mt5.ACCOUNT_TRADE_MODE_REAL
    engine = LiveEngine(_live_cfg(tmp_path), MT5Broker(_live_cfg(tmp_path).mt5, module=fake_mt5))
    with pytest.raises(BrokerError, match="REAL-MONEY"):
        engine.start()


def test_live_engine_warms_up_and_processes_new_bars(tmp_path):
    fake_mt5.state.reset()
    cfg = _live_cfg(tmp_path)
    engine = LiveEngine(cfg, MT5Broker(cfg.mt5, module=fake_mt5))
    core = engine.start()
    assert core.brains["EURUSD"].n_learned > 0
    assert not engine._process_symbol(core, "EURUSD")  # no new bar yet
    fake_mt5.state.now += 300  # a bar closes
    assert engine._process_symbol(core, "EURUSD")
    core.save_state()
    engine.journal.close()
    assert (tmp_path / "state" / "bot_state.json").exists()
    decisions = pd.read_csv(tmp_path / "logs" / "decisions.csv")
    assert len(decisions) == 1 and decisions["symbol"].iat[0] == "EURUSD"
