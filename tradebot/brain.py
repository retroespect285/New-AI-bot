"""The bot's "brain" for one symbol - everything it has learned, saved to ``state/brain_*.joblib``.

**Trade setups.** For every *arm* (BUY or SELL x a trade style such as "H1: stop 1 ATR, target
2 ATR, hold up to 8 hours") two adaptive learners forecast:

* ``tp`` - the probability that the take-profit is hit first;
* ``sl`` - the probability that the stop-loss is hit first (otherwise the trade times out).

Expected value in R (multiples of the money risked) is
``EV = P(tp) * avg_tp_R + P(sl) * avg_sl_R + P(timeout) * avg_timeout_R``, with the average results
measured from the market itself (spread and costs included).

An arm *signals* when its model has proven skill and EV clears the threshold. Every signal is
paper-traded in the background; real orders are only allowed while that out-of-sample track record
(plus the setup's real results) is positive. When the market changes and a pattern stops working,
the bot sees it in its own recent results and stops trading that setup until it works again.

**Outlooks.** For each outlook timeframe (5 min, 30 min, 1 h, 4 h by default) a learner forecasts
whether price will be higher one candle from now. Each forecast is scored when the answer is known,
giving a running accuracy per timeframe - the bot's report card.

**Learning history.** A daily snapshot of skill, accuracy and paper results is kept so you can see
whether the brain is getting smarter (``python -m tradebot status``).
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
import pandas as pd

from .config import ModelConfig
from .features import FEATURE_VERSION
from .labels import Arm, ArmLabels, make_arms
from .learner import AdaptiveLearner

BRAIN_VERSION = 5
KINDS = ("tp", "sl")
NS_PER_MIN = 60_000_000_000


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
    position), and the results of the most recent ones decide whether real orders are allowed.
    Real trades of the arm are added too."""

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


class OutlookStats:
    """Directional hit rate of one timeframe's forecasts (recent window and all-time). Longer
    timeframes get a longer window: their forecasts overlap, so each one carries less news."""

    def __init__(self, window: int = 500):
        self.recent: deque = deque(maxlen=window)
        self.n = 0
        self.hits = 0

    def update(self, p_up: float, went_up: float) -> None:
        hit = (p_up >= 0.5) == (went_up >= 0.5)
        self.recent.append(1.0 if hit else 0.0)
        self.n += 1
        self.hits += int(hit)

    @property
    def accuracy(self) -> Optional[float]:
        return float(np.mean(self.recent)) if len(self.recent) >= 20 else None


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
    outlook: Dict[str, float] = field(default_factory=dict)  # timeframe -> P(price higher)

    @property
    def action(self) -> str:
        return "HOLD" if self.arm is None else self.arm.side.label

    def outlook_text(self) -> str:
        return " ".join(f"{tf}:{'UP' if p >= 0.5 else 'DN'}{max(p, 1 - p):.0%}" for tf, p in self.outlook.items())


class Brain:
    def __init__(self, symbol: str, feature_names: List[str], cfg: ModelConfig, bar_minutes: int = 5,
                 outlook_horizons: Optional[Dict[str, int]] = None):
        self.version = BRAIN_VERSION
        self.feature_version = FEATURE_VERSION
        self.symbol = symbol
        self.feature_names = list(feature_names)
        self.cfg = cfg
        self.bar_ns = int(bar_minutes) * NS_PER_MIN
        self.arms = make_arms(cfg.styles)
        self.outlook_horizons = dict(outlook_horizons or {})
        n = len(feature_names)
        self.learners: Dict[str, AdaptiveLearner] = {}
        seed = cfg.seed
        for arm in self.arms:
            for kind in KINDS:
                self.learners[self._lkey(arm, kind)] = AdaptiveLearner(n, cfg, seed=seed, horizon=arm.horizon)
                seed += 1
        for tf, h in self.outlook_horizons.items():
            self.learners[self._okey(tf)] = AdaptiveLearner(n, cfg, seed=seed, horizon=h)
            seed += 1
        self.outcomes = {a.key: OutcomeStats(tp_r=a.tp_atr / a.sl_atr, alpha=cfg.outcome_alpha) for a in self.arms}
        self.track = {a.key: TrackRecord(cfg.paper_window) for a in self.arms}
        self.outlook_stats = {tf: OutlookStats(500 * max(1, h // 6)) for tf, h in self.outlook_horizons.items()}
        self.cursor: Dict[str, Optional[int]] = {}  # learning group -> bar time (ns) of last learned sample
        self.history: List[dict] = []  # one snapshot per day of learning
        self._day: Optional[str] = None
        self._forecasts: Dict[Tuple[str, int], Dict[str, float]] = {}
        self._pending: Set[Tuple[str, int]] = set()  # paper trades waiting for their outcome

    @staticmethod
    def _lkey(arm: Arm, kind: str) -> str:
        return f"{arm.key}|{kind}"

    @staticmethod
    def _okey(tf: str) -> str:
        return f"outlook:{tf}"

    @property
    def n_learned(self) -> int:
        return max((lr.n_updates for lr in self.learners.values()), default=0)

    # ------------------------------------------------------------------ forecasting
    def _comps(self, key: str, X, times, valid, i: int) -> Dict[str, float]:
        """Expert forecasts of learner ``key`` for row ``i`` (cached: each is made once, and later
        scored against the real outcome)."""
        t = int(times[i])
        comps = self._forecasts.get((key, t))
        if comps is None:
            comps = self.learners[key].components(X, times, i, valid)
            self._forecasts[(key, t)] = comps
        return comps

    def _assess(self, arm: Arm, X, times, valid, i: int) -> ArmEval:
        cfg = self.cfg
        l_tp = self.learners[self._lkey(arm, "tp")]
        l_sl = self.learners[self._lkey(arm, "sl")]
        p_tp = l_tp.blend(self._comps(self._lkey(arm, "tp"), X, times, valid, i))
        p_sl = l_sl.blend(self._comps(self._lkey(arm, "sl"), X, times, valid, i))
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
        if signal:  # open a paper trade (one at a time per arm); scored once its outcome is known
            t = int(times[i])
            if tr.last_open is None or t - tr.last_open >= arm.horizon * self.bar_ns:
                self._pending.add((arm.key, t))
                tr.last_open = t
        track_ok = (len(tr.recent) >= cfg.min_paper_trades and tr.avg_r >= cfg.min_paper_avg_r
                    and tr.tstat >= cfg.min_paper_tstat)
        return ArmEval(arm, p_tp, p_sl, ev, skill, ready, signal, len(tr.recent), tr.avg_r, signal and track_ok)

    def outlook(self, X, times, valid, i: int) -> Dict[str, float]:
        return {tf: self.learners[self._okey(tf)].blend(self._comps(self._okey(tf), X, times, valid, i))
                for tf in self.outlook_horizons}

    # ------------------------------------------------------------------ learning
    def _start(self, group: str, times: np.ndarray) -> int:
        c = self.cursor.get(group)
        return 0 if c is None else int(np.searchsorted(times, c, side="right"))

    def learn(self, X: np.ndarray, times: np.ndarray, valid: np.ndarray, labels: Dict[str, ArmLabels],
              outlook_labels: Optional[Dict[str, np.ndarray]], last: int) -> int:
        """Learn every sample whose outcome is fully known once bar ``last`` has closed.

        A setup with a holding time of H bars learns sample j when bar j+H closes - exactly when, in
        live trading, the forecast for bar j+H is made. That forecast is made here, with the model as
        it is at that moment, so replaying history (warm-up, backtests) is scored as honestly as live.
        Returns the number of new bars learned by the fastest-learning setup.
        """
        n = len(X)
        learned_max = 0
        for arm in self.arms:
            lab = labels[arm.key]
            group, h = arm.key, arm.horizon
            learned = 0
            for j in range(self._start(group, times), last - h + 1):
                t_j = int(times[j])
                y_tp, y_sl, r = lab.win[j], lab.loss[j], lab.r[j]
                if valid[j] and np.isfinite(y_tp) and np.isfinite(r):
                    for kind, y in (("tp", y_tp), ("sl", y_sl)):
                        key = self._lkey(arm, kind)
                        # score the forecast made back at bar j (None right after start-up)
                        self.learners[key].update(X[j], float(y), self._forecasts.pop((key, t_j), None))
                    self.outcomes[arm.key].update(bool(y_tp), bool(y_sl), float(r))
                    if (arm.key, t_j) in self._pending:
                        self._pending.discard((arm.key, t_j))
                        self.track[arm.key].update(float(r))
                    learned += 1
                self.cursor[group] = t_j
                k = j + h
                if k < n and valid[k]:
                    self._assess(arm, X, times, valid, k)
            learned_max = max(learned_max, learned)
        for tf, h in self.outlook_horizons.items():
            key = self._okey(tf)
            y_arr = outlook_labels[tf] if outlook_labels else None
            if y_arr is None:
                continue
            for j in range(self._start(key, times), last - h + 1):
                t_j = int(times[j])
                y = y_arr[j]
                if valid[j] and np.isfinite(y):
                    comps = self._forecasts.pop((key, t_j), None)
                    if comps is not None:
                        self.outlook_stats[tf].update(self.learners[key].blend(comps), float(y))
                    self.learners[key].update(X[j], float(y), comps)
                self.cursor[key] = t_j
                k = j + h
                if k < n and valid[k]:
                    self._comps(key, X, times, valid, k)
        self._prune()
        if 0 <= last < n:
            self._maybe_snapshot(int(times[last]))
        return learned_max

    def _prune(self) -> None:
        if not self.cursor:
            return
        oldest = min(self.cursor.values())
        if self._forecasts:
            self._forecasts = {k: v for k, v in self._forecasts.items() if k[1] > oldest}
        if self._pending:
            self._pending = {k for k in self._pending if k[1] > self.cursor.get(k[0], oldest)}

    def _maybe_snapshot(self, t_ns: int) -> None:
        day = pd.Timestamp(t_ns).strftime("%Y-%m-%d")
        if self._day is not None and day != self._day:
            self.history.append(self.snapshot(self._day))
            del self.history[:-400]
        self._day = day

    def snapshot(self, day: str) -> dict:
        arms = {}
        for arm in self.arms:
            s = self.learners[self._lkey(arm, "tp")].skill()
            tr = self.track[arm.key]
            arms[arm.key] = {"skill": round(s, 4) if math.isfinite(s) else None,
                             "paper_avg_r": round(tr.avg_r, 3), "paper_n": tr.n}
        outlooks = {}
        for tf, st in self.outlook_stats.items():
            s = self.learners[self._okey(tf)].skill()
            outlooks[tf] = {"accuracy": None if st.accuracy is None else round(st.accuracy, 3),
                            "skill": round(s, 4) if math.isfinite(s) else None}
        return {"date": day, "samples": self.n_learned, "setups": arms, "outlooks": outlooks}

    def record_real_trade(self, arm_key: str, r: float) -> None:
        """A closed real trade is first-hand evidence about its setup: it counts in the same track
        record, so a losing streak switches the setup off quickly."""
        if arm_key in self.track:
            self.track[arm_key].update(float(r))

    # ------------------------------------------------------------------ deciding
    def evaluate(self, X: np.ndarray, times: np.ndarray, valid: np.ndarray, i: int) -> List[ArmEval]:
        return [self._assess(arm, X, times, valid, i) for arm in self.arms]

    def decide(self, X: np.ndarray, times: np.ndarray, valid: np.ndarray, i: int) -> Decision:
        cfg = self.cfg
        if not valid[i]:
            return Decision(None, reason="indicators not ready (needs more chart history)")
        evals = self.evaluate(X, times, valid, i)
        outlook = self.outlook(X, times, valid, i)
        eligible = [e for e in evals if e.eligible]
        if eligible:
            best = max(eligible, key=lambda e: e.ev_r)
            return Decision(best.arm, prob=best.p_tp, ev_r=best.ev_r, reason="signal", evals=evals, outlook=outlook)
        signals = [e for e in evals if e.signal]
        if not any(e.ready for e in evals):
            n = min(self.learners[self._lkey(a, "tp")].n_updates for a in self.arms)
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
        return Decision(None, reason=reason, evals=evals, outlook=outlook)

    # ------------------------------------------------------------------ persistence
    def compatible_with(self, feature_names: List[str], cfg: ModelConfig,
                        outlook_horizons: Optional[Dict[str, int]] = None) -> bool:
        return (
            getattr(self, "version", None) == BRAIN_VERSION
            and self.feature_version == FEATURE_VERSION
            and self.feature_names == list(feature_names)
            and [(a.key, a.horizon) for a in self.arms] == [(a.key, a.horizon) for a in make_arms(cfg.styles)]
            and self.outlook_horizons == dict(outlook_horizons or {})
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
            os.replace(tmp, path)  # atomic: a crash never leaves a half-written brain
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

    def outlook_summary(self) -> Dict[str, dict]:
        out = {}
        for tf, st in self.outlook_stats.items():
            s = self.learners[self._okey(tf)].summary()
            out[tf] = {"forecasts_scored": st.n,
                       "recent_accuracy": None if st.accuracy is None else round(st.accuracy, 3),
                       "all_time_accuracy": round(st.hits / st.n, 3) if st.n else None,
                       "skill": s["skill"]}
        return out
