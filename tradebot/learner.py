"""Self-adapting probability model for one arm (direction x exit profile).

Three "experts" forecast the probability that a trade hits its take-profit first:

* ``base``  - the recent base rate (what you'd guess knowing nothing about the chart);
* ``logit`` - an online logistic regression updated after *every* resolved sample (fast adaptation);
* ``gbm``   - gradient-boosted trees re-fit regularly on a rolling window of the most recent
  samples (non-linear patterns; old market regimes drop out of the window).

They are blended with exponentially-weighted Hedge weights driven by each expert's *recent*
out-of-sample log-loss, so whichever model is currently reading the market best gets the most say.
Every forecast is scored before the model learns from that sample (prequential evaluation), which
gives an honest, continuously updated "skill" score versus the base rate. Arms without skill are
not traded.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .config import ModelConfig

EPS = 1e-4
EXPERTS = ("base", "logit", "gbm")


def _logloss(p: float, y: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def _threads(n: int):
    """Limit OpenMP threads used by scikit-learn. The data sets here are small, so one thread is
    about as fast, and it avoids heavy slow-downs when several bots/backtests share the CPU."""
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=max(1, int(n)), user_api="openmp")


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


class EWScaler:
    """Exponentially-weighted online standardisation (tracks drifting feature distributions)."""

    def __init__(self, n_features: int, span: int):
        self.mean = np.zeros(n_features)
        self.var = np.ones(n_features)
        self.n = 0
        self.span = span

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / np.sqrt(self.var + 1e-12)
        return np.clip(z, -5.0, 5.0)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        alpha = max(1.0 / self.n, 2.0 / (self.span + 1.0))
        delta = x - self.mean
        self.mean = self.mean + alpha * delta
        self.var = (1.0 - alpha) * (self.var + alpha * delta * delta)
        if self.n == 1:
            self.var = np.ones_like(self.var)


class OnlineLogit:
    def __init__(self, n_features: int, lr: float, l2: float, span: int):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.scaler = EWScaler(n_features, span)

    def predict(self, x: np.ndarray) -> float:
        return _sigmoid(float(self.scaler.transform(x) @ self.w) + self.b)

    def update(self, x: np.ndarray, y: float) -> None:
        if self.scaler.n >= 2:
            z = self.scaler.transform(x)
            g = _sigmoid(float(z @ self.w) + self.b) - y
            self.w -= self.lr * (g * z + self.l2 * self.w)
            self.b -= self.lr * g
        self.scaler.update(x)


class RingBuffer:
    def __init__(self, capacity: int, n_features: int):
        self.X = np.empty((capacity, n_features))
        self.y = np.empty(capacity)
        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add(self, x: np.ndarray, y: float) -> None:
        self.X[self.pos] = x
        self.y[self.pos] = y
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def ordered(self):
        """Samples oldest -> newest."""
        if self.size < self.capacity:
            return self.X[: self.size], self.y[: self.size]
        idx = np.r_[self.pos : self.capacity, 0 : self.pos]
        return self.X[idx], self.y[idx]


class AdaptiveLearner:
    def __init__(self, n_features: int, cfg: ModelConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.n_features = n_features
        self.logit = OnlineLogit(n_features, cfg.logit_lr, cfg.logit_l2, cfg.scaler_span)
        self.gbm = None
        self.gbm_version = 0
        self.buffer = RingBuffer(cfg.gbm_window, n_features)
        self.base_rate = 0.5
        self.n_updates = 0
        self.n_eval = 0
        self.since_refit = 0
        # decayed cumulative out-of-sample log-loss of each expert and of the blend
        self.loss = {name: 0.0 for name in EXPERTS}
        # slower-decaying scores of the blend and the base rate, for the skill gate
        self.skill_blend = 0.0
        self.skill_base = 0.0
        self._gbm_cache: Dict[int, float] = {}

    # ------------------------------------------------------------------ prediction
    def weights(self) -> Dict[str, float]:
        active = [e for e in EXPERTS if e != "gbm" or self.gbm is not None]
        lmin = min(self.loss[e] for e in active)
        raw = {e: math.exp(-self.cfg.hedge_eta * (self.loss[e] - lmin)) for e in active}
        total = sum(raw.values())
        return {e: raw.get(e, 0.0) / total for e in EXPERTS}

    def gbm_proba(self, X: np.ndarray, keys: np.ndarray, i: int, valid: np.ndarray) -> Optional[float]:
        """GBM forecast for row ``i``. Predictions are made in batches (row i and the rows after it
        that the caller already has) and cached per model version; this is only a speed-up, the
        model itself is never trained on those rows."""
        if self.gbm is None:
            return None
        key = int(keys[i])
        p = self._gbm_cache.get(key)
        if p is not None:
            return p
        stop = min(len(X), i + max(1, self.cfg.gbm_refit_every))
        rows = np.arange(i, stop)
        rows = rows[valid[rows]]
        if rows.size == 0 or rows[0] != i:
            rows = np.r_[i, rows[rows != i]]
        with _threads(self.cfg.n_threads):
            probs = self.gbm.predict_proba(X[rows])[:, 1]
        for r_idx, prob in zip(rows, probs):
            self._gbm_cache[int(keys[r_idx])] = float(prob)
        return self._gbm_cache[key]

    def components(self, X: np.ndarray, keys: np.ndarray, i: int, valid: np.ndarray) -> Dict[str, float]:
        comps = {"base": self.base_rate, "logit": self.logit.predict(X[i])}
        g = self.gbm_proba(X, keys, i, valid)
        if g is not None:
            comps["gbm"] = g
        return comps

    def blend(self, comps: Dict[str, float]) -> float:
        w = self.weights()
        p = sum(w[e] * comps[e] for e in comps if e in w)
        norm = sum(w[e] for e in comps if e in w)
        return p / norm if norm > 0 else self.base_rate

    # ------------------------------------------------------------------ learning
    def update(self, x: np.ndarray, y: float, comps: Optional[Dict[str, float]]) -> None:
        """Score the forecast ``comps`` (made before the outcome was known), then learn ``(x, y)``."""
        d = self.cfg.loss_decay
        if comps is not None and self.n_updates >= 50:  # let the experts settle before keeping score
            p_blend = self.blend(comps)
            for e in EXPERTS:
                # an expert that had no forecast is charged the base-rate loss (neutral)
                self.loss[e] = d * self.loss[e] + _logloss(comps.get(e, comps["base"]), y)
            ds = self.cfg.skill_decay
            self.skill_blend = ds * self.skill_blend + _logloss(p_blend, y)
            self.skill_base = ds * self.skill_base + _logloss(comps["base"], y)
            self.n_eval += 1
        alpha = max(1.0 / (self.n_updates + 1), 1.0 - d)
        self.base_rate += alpha * (y - self.base_rate)
        self.logit.update(x, y)
        self.buffer.add(x, y)
        self.n_updates += 1
        self.since_refit += 1
        if self.cfg.gbm_enabled and self.since_refit >= self.cfg.gbm_refit_every:
            self.refit()

    def refit(self) -> bool:
        self.since_refit = 0
        X, y = self.buffer.ordered()
        if len(y) < max(200, self.cfg.min_train_samples // 2):
            return False
        n_pos = int(y.sum())
        if n_pos < 20 or len(y) - n_pos < 20:
            return False
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=self.cfg.gbm_max_iter,
            learning_rate=self.cfg.gbm_learning_rate,
            max_depth=self.cfg.gbm_max_depth,
            max_bins=63,
            min_samples_leaf=self.cfg.gbm_min_samples_leaf,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=self.seed,
        )
        # Recency comes from the rolling window: only the latest ``gbm_window`` samples are used, so
        # patterns from old market regimes drop out. (Per-sample weights are avoided on purpose:
        # they make scikit-learn's histogram binning ~30x slower.)
        with _threads(self.cfg.n_threads):
            model.fit(X, y.astype(int))
        if self.gbm is None:
            # a newly introduced expert starts with a neutral track record
            self.loss["gbm"] = self.loss["base"]
        self.gbm = model
        self.gbm_version += 1
        self._gbm_cache.clear()
        return True

    # ------------------------------------------------------------------ diagnostics
    def skill(self) -> float:
        """Log-loss skill score of the blend vs. the base rate (>0 means the chart helps)."""
        if self.n_eval < self.cfg.min_eval_samples or self.skill_base <= 0:
            return float("-inf")
        return 1.0 - self.skill_blend / self.skill_base

    def ready(self) -> bool:
        return self.n_updates >= self.cfg.min_train_samples

    def summary(self) -> dict:
        w = self.weights()
        s = self.skill()
        return {
            "samples": self.n_updates,
            "base_rate": round(self.base_rate, 4),
            "skill": None if not math.isfinite(s) else round(s, 4),
            "w_base": round(w["base"], 3),
            "w_logit": round(w["logit"], 3),
            "w_gbm": round(w["gbm"], 3),
            "gbm_version": self.gbm_version,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_gbm_cache"] = {}
        return state
