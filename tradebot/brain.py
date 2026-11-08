"""The decision maker for one symbol.

For every *arm* (BUY or SELL x a stop-loss/take-profit profile) two adaptive learners forecast:

* ``tp`` - the probability that the take-profit is hit first;
* ``sl`` - the probability that the stop-loss is hit first (otherwise the trade times out).

Expected value in R (multiples of the money risked) is then
``EV = P(tp) * avg_tp_R + P(sl) * avg_sl_R + P(timeout) * avg_timeout_R``, with the average results
measured from the market itself (spread and costs included).

An arm *signals* when its model has proven skill and EV clears the threshold. Every signal is
paper-traded in the background ("shadow track record"); real orders are only allowed while that
out-of-sample track record is positive. So when the market changes and the patterns stop working,
the bot notices from its own recent results and stops trading that setup until it works again.
"""
from __future__ import annotations

import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np

from .config import ModelConfig
from .features import FEATURE_VERSION
from .labels import Arm, ArmLabels, make_arms
from .learner import AdaptiveLearner

BRAIN_VERSION = 3
KINDS = ("tp", "sl")


def _ewma(avg: float, n: int, value: float, alpha: float) -> float:
    return avg + max(alpha, 1.0 / n) * (value - avg)


@dataclass
class OutcomeStats:
    """Average realised R of each outcome type for one arm (spread / costs / gaps included)."""

    tp_r: float
    sl_r: float = -1.0
    timeout_r: float = 0.0
    alpha: float = 0.01
    n_tp: int = 0
    n_sl: int = 0
    n_timeout: int = 0

    def update(self, won: bool, lost: bool, r: float) -> None:
        if won:
            self.n_tp += 1
            self.tp_r = _ewma(self.tp_r, self.n_tp, r, self.alpha)
        elif lost:
            self.n_sl += 1
            self.sl_r = _ewma(self.sl_r, self.n_sl, r, self.alpha)
        else:
            self.n_timeout += 1
            self.timeout_r = _ewma(self.timeout_r, self.n_timeout, r, self.alpha)


class TrackRecord:
    """Paper trades of one arm: every signal is "traded" on paper, one at a time (like a real
    position), and the results of the most recent ones decide whether real orders are allowed."""

    def __init__(self, window: int):
        self.recent: deque = deque(maxlen=max(2, window))
        self.n = 0
        self.total_r = 0.0
        self.last_open: Optional[int] = None  # bar time (ns) of the last paper trade opened

    def update(self, r: float) -> None:
        self.n += 1
        self.total_r += r
        self.recent.append(r)

    @property
    def avg_r(self) -> float:
        return float(np.mean(self.recent)) if self.recent else 0.0

    @property
    def tstat(self) -> float:
        k = len(self.recent)
        if k < 3:
            return 0.0
        sd = float(np.std(self.recent, ddof=1))
        return self.avg_r / (sd / math.sqrt(k)) if sd > 0 else 0.0


@dataclass
class ArmEval:
    arm: Arm
    p_tp: float
    p_sl: float
    ev_r: float
    skill: float
    ready: bool
    signal: bool  # the model says trade
    track_n: int
    track_r: float
    eligible: bool  # signal AND a positive recent track record

    @property
    def prob(self) -> float:
        return self.p_tp


@dataclass
class Decision:
    arm: Optional[Arm]
    prob: float = 0.0
    ev_r: float = 0.0
    reason: str = ""
    evals: List[ArmEval] = field(default_factory=list)

    @property
    def action(self) -> str:
        return "HOLD" if self.arm is None else self.arm.side.label


class Brain:
    def __init__(self, symbol: str, feature_names: List[str], cfg: ModelConfig, bar_minutes: int = 5):
        self.version = BRAIN_VERSION
        self.bar_ns = int(bar_minutes) * 60 * 1_000_000_000
        self.feature_version = FEATURE_VERSION
        self.symbol = symbol
        self.feature_names = list(feature_names)
        self.cfg = cfg
        self.arms = make_arms(cfg.exit_profiles)
        n = len(feature_names)
        self.learners: Dict[str, AdaptiveLearner] = {}
        for k, arm in enumerate(self.arms):
            for kind in KINDS:
                self.learners[self._lkey(arm, kind)] = AdaptiveLearner(n, cfg, seed=cfg.seed + 2 * k + KINDS.index(kind))
        self.outcomes = {a.key: OutcomeStats(tp_r=a.tp_atr / a.sl_atr, alpha=cfg.outcome_alpha) for a in self.arms}
        self.track = {a.key: TrackRecord(cfg.paper_window) for a in self.arms}
        self.last_learned_time: Optional[int] = None  # bar open time (ns) of the last learned sample
        self.n_learned = 0
        self._forecasts: Dict[Tuple[str, int], Dict[str, float]] = {}
        self._pending: Set[Tuple[str, int]] = set()  # shadow signals waiting for their outcome

    @staticmethod
    def _lkey(arm: Arm, kind: str) -> str:
        return f"{arm.key}|{kind}"

    # ------------------------------------------------------------------ forecasting
    def _forecast(self, X, times, valid, i: int) -> Dict[str, Dict[str, float]]:
        """Expert forecasts for row ``i`` for every learner (cached so each is made once)."""
        t = int(times[i])
        out = {}
        for key, learner in self.learners.items():
            comps = self._forecasts.get((key, t))
            if comps is None:
                comps = learner.components(X, times, i, valid)
                self._forecasts[(key, t)] = comps
            out[key] = comps
        return out

    def _assess(self, arm: Arm, comps: Dict[str, Dict[str, float]]) -> ArmEval:
        cfg = self.cfg
        l_tp = self.learners[self._lkey(arm, "tp")]
        l_sl = self.learners[self._lkey(arm, "sl")]
        p_tp = l_tp.blend(comps[self._lkey(arm, "tp")])
        p_sl = l_sl.blend(comps[self._lkey(arm, "sl")])
        total = p_tp + p_sl
        if total > 1.0:
            p_tp, p_sl = p_tp / total, p_sl / total
        p_to = 1.0 - p_tp - p_sl
        st = self.outcomes[arm.key]
        ev = p_tp * st.tp_r + p_sl * st.sl_r + p_to * st.timeout_r
        skill = l_tp.skill()
        ready = l_tp.ready() and l_sl.ready()
        signal = (ready and math.isfinite(skill) and skill >= cfg.min_skill
                  and p_tp >= cfg.min_prob and ev >= cfg.min_ev_r)
        tr = self.track[arm.key]
        track_ok = (len(tr.recent) >= cfg.min_paper_trades and tr.avg_r >= cfg.min_paper_avg_r
                    and tr.tstat >= cfg.min_paper_tstat)
        return ArmEval(arm, p_tp, p_sl, ev, skill, ready, signal, len(tr.recent), tr.avg_r, signal and track_ok)

    def _evaluate_row(self, X, times, valid, i: int) -> List[ArmEval]:
        comps = self._forecast(X, times, valid, i)
        evals = [self._assess(arm, comps) for arm in self.arms]
        t = int(times[i])
        spacing = self.cfg.horizon_bars * self.bar_ns
        for e in evals:
            tr = self.track[e.arm.key]
            if e.signal and (tr.last_open is None or t - tr.last_open >= spacing):
                self._pending.add((e.arm.key, t))  # open a paper trade; scored once its outcome is known
                tr.last_open = t
        return evals

    # ------------------------------------------------------------------ learning
    def learn(self, X: np.ndarray, times: np.ndarray, valid: np.ndarray,
              labels: Dict[str, ArmLabels], upto: int) -> int:
        """Learn every sample with index <= ``upto`` that is newer than the last learned one.

        ``upto`` must be ``last_closed_index - horizon`` so that each sample's full outcome window
        is already in the past.
        """
        if upto < 0:
            return 0
        start = 0
        if self.last_learned_time is not None:
            start = int(np.searchsorted(times, self.last_learned_time, side="right"))
        horizon = self.cfg.horizon_bars
        n = len(X)
        learned = 0
        for j in range(start, upto + 1):
            t_j = int(times[j])
            if valid[j]:
                x = X[j]
                for arm in self.arms:
                    lab = labels[arm.key]
                    y_tp, y_sl, r = lab.win[j], lab.loss[j], lab.r[j]
                    if not (np.isfinite(y_tp) and np.isfinite(r)):
                        continue
                    for kind, y in (("tp", y_tp), ("sl", y_sl)):
                        key = self._lkey(arm, kind)
                        # Score the forecast made back at bar j (None right after start-up: then
                        # the sample is learned but not scored).
                        comps = self._forecasts.pop((key, t_j), None)
                        self.learners[key].update(x, float(y), comps)
                    self.outcomes[arm.key].update(bool(y_tp), bool(y_sl), float(r))
                    if (arm.key, t_j) in self._pending:
                        self._pending.discard((arm.key, t_j))
                        self.track[arm.key].update(float(r))
                learned += 1
            self.last_learned_time = t_j
            # Live, sample j is learned when bar j+horizon closes - exactly when the forecast for
            # bar j+horizon is made. Making that forecast here, with the model as it is at that
            # moment, means replaying history (warm-up, backtest) is scored as honestly as live.
            k = j + horizon
            if k < n and valid[k]:
                self._evaluate_row(X, times, valid, k)
        self.n_learned += learned
        cutoff = self.last_learned_time or 0
        if self._forecasts:
            self._forecasts = {k: v for k, v in self._forecasts.items() if k[1] > cutoff}
        if self._pending:
            self._pending = {k for k in self._pending if k[1] > cutoff}
        return learned

    def record_real_trade(self, arm_key: str, r: float) -> None:
        """A closed real trade is first-hand evidence about its setup: it counts in the same track
        record, so a losing streak switches the setup off quickly."""
        if arm_key in self.track:
            self.track[arm_key].update(float(r))

    # ------------------------------------------------------------------ deciding
    def evaluate(self, X: np.ndarray, times: np.ndarray, valid: np.ndarray, i: int) -> List[ArmEval]:
        return self._evaluate_row(X, times, valid, i)

    def decide(self, X: np.ndarray, times: np.ndarray, valid: np.ndarray, i: int) -> Decision:
        cfg = self.cfg
        if not valid[i]:
            return Decision(None, reason="indicators not ready")
        evals = self.evaluate(X, times, valid, i)
        eligible = [e for e in evals if e.eligible]
        if eligible:
            best = max(eligible, key=lambda e: e.ev_r)
            return Decision(best.arm, prob=best.p_tp, ev_r=best.ev_r, reason="signal", evals=evals)
        signals = [e for e in evals if e.signal]
        if not any(e.ready for e in evals):
            n = min(lr.n_updates for lr in self.learners.values())
            reason = f"learning ({n}/{cfg.min_train_samples} samples)"
        elif signals:
            s = max(signals, key=lambda e: e.ev_r)
            reason = (f"{s.arm.key} signals EV {s.ev_r:+.2f}R, but its recent paper trades average "
                      f"{s.track_r:+.2f}R over {s.track_n} (needs {cfg.min_paper_trades}+ trades, "
                      f"avg >= {cfg.min_paper_avg_r:+.2f}R, t-stat >= {cfg.min_paper_tstat})")
        elif not any(math.isfinite(e.skill) and e.skill >= cfg.min_skill for e in evals):
            reason = "no setup currently beats the base rate"
        else:
            best = max(evals, key=lambda e: e.ev_r)
            reason = f"best EV {best.ev_r:+.2f}R < {cfg.min_ev_r}R"
        return Decision(None, reason=reason, evals=evals)

    # ------------------------------------------------------------------ persistence
    def compatible_with(self, feature_names: List[str], cfg: ModelConfig) -> bool:
        return (
            getattr(self, "version", None) == BRAIN_VERSION
            and self.feature_version == FEATURE_VERSION
            and self.feature_names == list(feature_names)
            and [a.key for a in self.arms] == [a.key for a in make_arms(cfg.exit_profiles)]
        )

    def apply_config(self, cfg: ModelConfig) -> None:
        """Use new thresholds / cadences from the current config after loading a saved brain."""
        self.cfg = cfg
        for learner in self.learners.values():
            learner.cfg = cfg

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        try:
            joblib.dump(self, tmp, compress=3)
            os.replace(tmp, path)  # atomic: a crash never leaves a half-written model
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @staticmethod
    def load(path: Path) -> "Brain":
        return joblib.load(path)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_forecasts"] = {}
        state["_pending"] = set()
        return state

    def summary(self) -> Dict[str, dict]:
        out = {}
        for arm in self.arms:
            tp = self.learners[self._lkey(arm, "tp")].summary()
            sl = self.learners[self._lkey(arm, "sl")].summary()
            st, tr = self.outcomes[arm.key], self.track[arm.key]
            out[arm.key] = {
                "samples": tp["samples"],
                "tp_rate": tp["base_rate"],
                "sl_rate": sl["base_rate"],
                "skill_tp": tp["skill"],
                "skill_sl": sl["skill"],
                "weights_tp(b/l/g)": f"{tp['w_base']:.2f}/{tp['w_logit']:.2f}/{tp['w_gbm']:.2f}",
                "avg_r(tp/sl/timeout)": f"{st.tp_r:+.2f}/{st.sl_r:+.2f}/{st.timeout_r:+.2f}",
                "paper_trades": tr.n,
                "paper_recent_avg_r": round(tr.avg_r, 3) + 0.0,
                "paper_recent_tstat": round(tr.tstat, 2) + 0.0,
                "paper_total_r": round(tr.total_r, 1),
            }
        return out
