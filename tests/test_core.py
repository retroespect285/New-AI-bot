import json

import pandas as pd
import pytest

from tests import fake_mt5
from tests.conftest import EURUSD, mtf_config, small_config
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
    arm = make_arms(cfg.model.styles)[0]  # BUY sl1 tp1.5
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
    arm = make_arms(cfg.model.styles)[2]  # BUY sl1.5 tp3 - unlikely to be hit quickly
    _patch_decision(monkeypatch, core, prep, arm)
    i = 700
    sim.set_index("EURUSD", i)
    core.on_bar("EURUSD", prep, i)
    brain = core.brain("EURUSD", prep.features.names)
    monkeypatch.setattr(brain, "decide", lambda *a, **k: Decision(None, reason="test"))
    for k in range(1, arm.horizon + 2):
        sim.advance("EURUSD")
        core.on_bar("EURUSD", prep, i + k)
    assert not sim.positions("EURUSD")
    reasons = [t.reason for t in sim.all_closed]
    assert len(reasons) == 1 and reasons[0] in ("sl", "tp", "ai:time-exit")
    # the real result (in R) is fed back into that setup's track record
    core.sync_closed(pd.Timestamp(prep.times[i + arm.horizon + 2]))
    track = brain.track[arm.key]
    assert track.n == 1  # no paper trades yet: the model is still learning
    assert track.recent[-1] == pytest.approx(sim.all_closed[0].profit / 50.0, rel=0.05)


def test_dry_run_never_sends_orders(tmp_path, monkeypatch):
    cfg, sim, core, prep = _core(tmp_path, dry_run=True)
    _patch_decision(monkeypatch, core, prep, make_arms(cfg.model.styles)[1])
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


def test_switching_from_demo_to_live_account_resets_risk_but_keeps_the_brain(tmp_path):
    state = tmp_path / "state"
    cfg, sim, core, prep = _core(tmp_path, state_dir=str(state))
    core.learn("EURUSD", prep, 1100)
    core.bind_account(sim.account())  # demo account (login 0 in the simulator)
    core.risk.update_equity(10_000, pd.Timestamp("2024-01-05"))
    core.risk.update_equity(12_000, pd.Timestamp("2024-01-06"))
    core.save_state()
    learned = core.brains["EURUSD"].n_learned

    from tradebot.types import AccountInfo

    live = AccountInfo(login=777, balance=1_000, equity=1_000, margin_free=1_000, currency="USD", is_demo=False)
    core2 = BotCore(cfg, sim, state_dir=str(state))
    assert core2.risk.state.peak_equity == 12_000
    core2.bind_account(live)
    assert core2.risk.state.peak_equity is None  # a 1,000 live account is not a 92% drawdown of 12,000
    core2.risk.update_equity(1_000, pd.Timestamp("2024-01-07"))
    assert core2.risk.can_trade(1_000, pd.Timestamp("2024-01-07"))[0]
    core2.add_symbol("EURUSD")
    assert core2.brain("EURUSD", prep.features.names).n_learned == learned


def test_news_blackout_blocks_new_trades(tmp_path, monkeypatch):
    cfg, sim, core, prep = _core(tmp_path)
    i = 700
    t = pd.Timestamp(int(prep.times[i])) + pd.Timedelta(minutes=5)
    news = tmp_path / "news.csv"
    news.write_text(f"time,title\n{t + pd.Timedelta(minutes=10):%Y-%m-%d %H:%M},US CPI\n")
    from tradebot.news import NewsCalendar

    core.news = NewsCalendar.load(str(news), 15, 15)
    _patch_decision(monkeypatch, core, prep, make_arms(cfg.model.styles)[0])
    sim.set_index("EURUSD", i)
    d = core.on_bar("EURUSD", prep, i)
    assert d.reason == "blocked: news blackout (US CPI)" and not sim.positions()


def test_daily_brain_backups(tmp_path):
    state = tmp_path / "state"
    cfg, sim, core, prep = _core(tmp_path, state_dir=str(state), n=2000)
    cfg.model.backup_days = 2
    for stop in (600, 900, 1200, 1500, 1900):  # several different days
        core.learn("EURUSD", prep, stop)
        core.save_state()
    backups = sorted((state / "backups").glob("brain_EURUSD_M5_*.joblib"))
    assert len(backups) == 2  # only the newest `backup_days` are kept


def test_multi_timeframe_live_engine_and_style_stops(tmp_path, monkeypatch):
    fake_mt5.state.reset()
    cfg = mtf_config()
    cfg.trading.warmup_bars = 900
    cfg.trading.lookback_bars = 600
    cfg.risk.max_spread_sl_fraction = 1.0
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.paths.log_dir = str(tmp_path / "logs")
    engine = LiveEngine(cfg, MT5Broker(cfg.mt5, module=fake_mt5))
    ctx = engine.fetch_context("EURUSD", 600)
    assert set(ctx) == {"M1", "M30", "H1"}
    assert len(ctx["M1"]) == cfg.context_fetch_bars("M1", 600)  # closed candles only
    assert ctx["H1"]["time"].iat[-1] + pd.Timedelta(hours=1) <= pd.Timestamp(fake_mt5.state.now, unit="s")
    core = engine.start()
    brain = core.brains["EURUSD"]
    assert any(n.startswith("h1_") for n in brain.feature_names)
    # an H1-style signal places its stop at 1 x ATR(H1), much wider than an M5 stop
    h1_arm = [a for a in core.arms if a.tf == "H1" and a.side.label == "BUY"][0]
    monkeypatch.setattr(brain, "decide", lambda *a, **k: Decision(h1_arm, prob=0.6, ev_r=0.3, reason="signal"))
    fake_mt5.state.now += 300
    assert engine._process_symbol(core, "EURUSD")
    req = [r for r in fake_mt5.state.requests if r["action"] == fake_mt5.TRADE_ACTION_DEAL][-1]
    bars = engine.broker.get_rates("EURUSD", "M5", 600)
    prep = core.prepare("EURUSD", bars, engine.fetch_context("EURUSD", 600))
    atr_h1, atr_m5 = prep.features.atr_tf["H1"][-1], prep.features.atr_tf["M5"][-1]
    assert atr_h1 > atr_m5
    assert abs((req["price"] - req["sl"]) - atr_h1) < 2e-5
    assert core.open_risk[555]["arm"] == h1_arm.key
    engine.journal.close()


def test_live_engine_in_one_minute_mode(tmp_path):
    from tradebot.config import config_from_dict

    fake_mt5.state.reset()
    cfg = config_from_dict({
        "trading": {"timeframe": "M1", "symbols": ["EURUSD"], "warmup_bars": 2500, "lookback_bars": 900,
                    "friday_cutoff_hour": None, "session": {"start_hour": 0, "end_hour": 24}},
        "model": {"activity": "active", "min_train_samples": 300, "gbm_window": 600, "gbm_refit_every": 500,
                  "gbm_max_iter": 20},
        "paths": {"state_dir": str(tmp_path / "state"), "log_dir": str(tmp_path / "logs")},
    }).validate()
    engine = LiveEngine(cfg, MT5Broker(cfg.mt5, module=fake_mt5))
    core = engine.start()
    brain = core.brains["EURUSD"]
    assert brain.bar_ns == 60 * 1_000_000_000
    assert any(n.startswith("m5_") for n in brain.feature_names) and not any(n.startswith("m1_") for n in brain.feature_names)
    assert set(brain.outlook_horizons) == {"M1", "M5", "M30", "H1", "H4"}
    fake_mt5.state.now += 60  # one minute later a new M1 candle has closed
    assert engine._process_symbol(core, "EURUSD")
    assert not engine._process_symbol(core, "EURUSD")  # nothing new until the next minute
    engine.journal.close()
    assert (tmp_path / "state" / "brain_EURUSD_M1.joblib").exists()
    d = pd.read_csv(tmp_path / "logs" / "decisions.csv")
    assert len(d) == 1 and "up_M1" in d.columns
