import math

import numpy as np

from tests.conftest import small_config
from tradebot.learner import AdaptiveLearner


def _stream(n, rng, signal=True):
    X = rng.normal(size=(n, 6))
    logits = 2.0 * X[:, 0] - 1.0 if signal else np.full(n, -1.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(float)
    return X, y


def _run(learner, X, y):
    keys = np.arange(len(y))
    valid = np.ones(len(y), dtype=bool)
    for i in range(len(y)):
        comps = learner.components(X, keys, i, valid)
        learner.update(X[i], y[i], comps)


def test_learns_a_real_pattern_and_reports_skill():
    cfg = small_config().model
    rng = np.random.default_rng(0)
    X, y = _stream(3000, rng)
    learner = AdaptiveLearner(6, cfg, seed=1)
    _run(learner, X, y)
    assert learner.ready()
    assert learner.skill() > 0.1
    assert learner.weights()["base"] < 0.2  # the chart-reading experts took over
    p_hi = learner.blend(learner.components(np.array([[2.0, 0, 0, 0, 0, 0]]), np.array([-1]), 0, np.array([True])))
    p_lo = learner.blend(learner.components(np.array([[-2.0, 0, 0, 0, 0, 0]]), np.array([-2]), 0, np.array([True])))
    assert p_hi > 0.8 and p_lo < 0.2


def test_no_skill_on_pure_noise():
    cfg = small_config().model
    rng = np.random.default_rng(1)
    X, y = _stream(3000, rng, signal=False)
    learner = AdaptiveLearner(6, cfg, seed=1)
    _run(learner, X, y)
    assert learner.skill() < cfg.min_skill + 0.01
    assert abs(learner.base_rate - 1 / (1 + math.e)) < 0.08


def test_adapts_when_the_market_flips():
    cfg = small_config().model
    rng = np.random.default_rng(2)
    X1, y1 = _stream(2000, rng)
    X2, _ = _stream(2000, rng)
    y2 = (rng.random(2000) < 1 / (1 + np.exp(-(-2.0 * X2[:, 0] - 1.0)))).astype(float)  # relationship reversed
    learner = AdaptiveLearner(6, cfg, seed=1)
    _run(learner, X1, y1)
    _run(learner, X2, y2)
    probe = np.array([[2.0, 0, 0, 0, 0, 0]])
    p = learner.blend(learner.components(probe, np.array([-9]), 0, np.array([True])))
    assert p < 0.3  # it now believes the reversed relationship


def test_unscored_updates_still_learn():
    cfg = small_config().model
    learner = AdaptiveLearner(3, cfg)
    for _ in range(100):
        learner.update(np.zeros(3), 1.0, None)
    assert learner.n_updates == 100 and learner.n_eval == 0
    assert learner.base_rate > 0.9
