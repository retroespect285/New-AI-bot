import numpy as np

from tests.conftest import EURUSD, small_config
from tradebot.brain import Brain
from tradebot.brokers.sim import SimBroker
from tradebot.config import TradeStyle
from tradebot.core import BotCore
from tradebot.data import synthetic_bars


def _prep(cfg, n=2500, seed=31):
    bars = synthetic_bars(n, seed=seed)
    core = BotCore(cfg, SimBroker({"EURUSD": bars}, {"EURUSD": EURUSD}))
    core.add_symbol("EURUSD")
    return core, core.prepare("EURUSD", bars)


def test_each_style_learns_only_what_is_already_known():
    cfg = small_config()
    cfg.model.styles = [TradeStyle("M5", 1.0, 1.5, 6), TradeStyle("M5", 1.5, 3.0, 30)]
    core, prep = _prep(cfg)
    fs = prep.features
    brain = core.brain("EURUSD", fs.names)
    last = 1500
    brain.learn(fs.X, prep.times, fs.valid, prep.labels, prep.outlook_labels, last)
    fast, slow = brain.arms[0], brain.arms[2]
    # a style that holds up to H bars can only have learned samples up to bar last - H
    assert brain.cursor[fast.key] == prep.times[last - 6]
    assert brain.cursor[slow.key] == prep.times[last - 30]
    assert brain.learners[f"{fast.key}|tp"].n_updates > brain.learners[f"{slow.key}|tp"].n_updates
    assert brain.cursor["outlook:M30"] == prep.times[last - 6]  # 30 min = 6 M5 bars


def test_outlooks_are_scored_and_history_is_kept(tmp_path):
    cfg = small_config()
    core, prep = _prep(cfg, n=3000)
    core.learn_range("EURUSD", prep, len(prep.bars) - 1, progress=False)
    brain = core.brains["EURUSD"]
    o = brain.outlook_summary()
    assert set(o) == {"M5", "M30"}
    assert o["M5"]["forecasts_scored"] > 2000
    assert 0.3 < o["M5"]["all_time_accuracy"] < 0.7
    # one snapshot per day of learning, with per-timeframe accuracy and per-setup skill
    days = prep.bars["time"].dt.strftime("%Y-%m-%d").nunique()
    assert len(brain.history) >= days - 3
    snap = brain.history[-1]
    assert set(snap["outlooks"]) == {"M5", "M30"} and len(snap["setups"]) == 4
    # the decision carries the outlook for every timeframe
    i = len(prep.bars) - 1
    d = brain.decide(prep.features.X, prep.times, prep.features.valid, i)
    assert set(d.outlook) == {"M5", "M30"} and all(0 < p < 1 for p in d.outlook.values())
    assert "M5:" in d.outlook_text()
    # saving / loading keeps everything
    brain.save(tmp_path / "b.joblib")
    loaded = Brain.load(tmp_path / "b.joblib")
    assert loaded.n_learned == brain.n_learned and len(loaded.history) == len(brain.history)
    assert loaded.compatible_with(prep.features.names, cfg.model, cfg.outlook_horizons())


def test_changed_styles_make_a_saved_brain_incompatible():
    cfg = small_config()
    core, prep = _prep(cfg, n=600)
    brain = core.brain("EURUSD", prep.features.names)
    other = small_config()
    other.model.styles = [TradeStyle("M5", 1.0, 1.5, 24)]
    assert not brain.compatible_with(prep.features.names, other.model, other.outlook_horizons())
    assert np.isfinite(prep.outlook_labels["M30"][:-6]).all()
