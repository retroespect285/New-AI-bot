"""The bot's per-bar cycle, shared by live trading and the backtester.

Each time a bar closes (every 5 minutes on M5) ``BotCore.on_bar`` does:

1. **learn**    - every trade setup / direction forecast whose outcome is now known becomes a
                  training sample (the brain keeps learning from the market, not only its own trades);
2. **account**  - update equity, daily-loss / drawdown limits and learn from the bot's closed trades;
3. **decide**   - read all charts (M1/M5/M30/H1/H4), score every setup (BUY/SELL x trade style),
                  pick the best positive-EV one, and update the per-timeframe outlook;
4. **manage**   - for open trades: move trailing stops behind the price, close when the brain is
                  confident the market turned against the trade, at the style's max holding
                  time or the Friday cut-off;
5. **execute**  - risk checks, position sizing from the stop distance, send the order with SL/TP.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .brain import Brain, Decision
from .brokers.base import Broker
from .config import TIMEFRAME_MINUTES, BotConfig
from .features import FeatureSet, build_mtf_features
from .journal import Journal
from .labels import ArmLabels, direction_labels, label_all, make_arms, spread_prices
from .news import NewsCalendar
from .risk import RiskManager, RiskState
from .types import AccountInfo, Position, Side, SymbolInfo, Tick

log = logging.getLogger(__name__)


def to_ns(times: pd.Series) -> np.ndarray:
    return pd.to_datetime(times).to_numpy(dtype="datetime64[ns]").astype(np.int64)


@dataclass
class Prepared:
    """Bars plus everything derived from them (features, outcome labels, int64 times)."""

    bars: pd.DataFrame
    features: FeatureSet
    labels: Dict[str, ArmLabels]
    outlook_labels: Dict[str, np.ndarray]
    times: np.ndarray


class BotCore:
    def __init__(self, cfg: BotConfig, broker: Broker, journal: Optional[Journal] = None,
                 state_dir: Optional[str] = None, dry_run: bool = False):
        self.cfg = cfg
        self.broker = broker
        self.journal = journal or Journal(None)
        self.state_dir = Path(state_dir) if state_dir else None
        self.dry_run = dry_run or cfg.trading.dry_run
        self.arms = make_arms(cfg.model.styles)
        self.arm_by_key = {a.key: a for a in self.arms}
        self.outlook_horizons = cfg.outlook_horizons()
        self.tf = pd.Timedelta(minutes=cfg.timeframe_minutes)
        self.tf_ns = int(self.tf.value)
        self.infos: Dict[str, SymbolInfo] = {}
        self.brains: Dict[str, Brain] = {}
        self.open_risk: Dict[int, dict] = {}  # ticket -> sizing info, to express results in R
        self.seen_closed: Deque[int] = deque(maxlen=5000)
        self._seen_set: set = set()
        self.last_sync: Optional[pd.Timestamp] = None
        self.account_login: Optional[int] = None
        self._last_backup_day: Optional[str] = None
        self.news = NewsCalendar.load(cfg.trading.news_file, cfg.trading.news_minutes_before,
                                      cfg.trading.news_minutes_after)
        self.risk = RiskManager(cfg.risk, self._load_state())

    # ------------------------------------------------------------------ setup
    def add_symbol(self, symbol: str) -> SymbolInfo:
        info = self.broker.symbol_info(symbol)
        self.infos[symbol] = info
        return info

    def prepare(self, symbol: str, bars: pd.DataFrame,
                context: Optional[Mapping[str, pd.DataFrame]] = None) -> Prepared:
        """Read every chart: ``bars`` is the trading timeframe, ``context`` maps the other
        timeframes (``model.context_timeframes``) to their bars."""
        info = self.infos[symbol]
        cfg, m, t = self.cfg, self.cfg.model, self.cfg.trading
        context = dict(context or {})
        missing = [tf for tf in m.context_timeframes if tf not in context]
        if missing:
            raise ValueError(f"{symbol}: bars for context timeframe(s) {missing} are missing")
        context = {tf: context[tf] for tf in m.context_timeframes}
        style_tfs = sorted({s.timeframe for s in m.styles})
        fs = build_mtf_features(bars, context, t.timeframe, TIMEFRAME_MINUTES, info.point, m.atr_period,
                                required_atr=style_tfs)
        extra_cost = m.slippage_points * info.point
        if t.commission_per_lot > 0:
            extra_cost += t.commission_per_lot / info.value_per_price_unit
        labels = label_all(bars, fs.atr_tf, self.arms, info.point, cfg.backtest.default_spread_points, extra_cost)
        outlook = {tf: direction_labels(bars, h) for tf, h in self.outlook_horizons.items()}
        return Prepared(bars, fs, labels, outlook, to_ns(bars["time"]))

    def brain(self, symbol: str, feature_names) -> Brain:
        if symbol in self.brains:
            return self.brains[symbol]
        brain = None
        path = self._brain_path(symbol)
        if path is not None and path.exists():
            try:
                loaded = Brain.load(path)
                if loaded.compatible_with(feature_names, self.cfg.model, self.outlook_horizons):
                    loaded.apply_config(self.cfg.model)
                    brain = loaded
                    log.info("%s: loaded brain from %s (%d samples learned, %d days of history)",
                             symbol, path, loaded.n_learned, len(loaded.history))
                else:
                    old = path.with_suffix(".incompatible")
                    shutil.copy2(path, old)
                    log.warning("%s: saved brain does not match the current settings (charts / styles / "
                                "outlooks changed) - kept a copy at %s and starting a new brain", symbol, old)
            except Exception as exc:  # corrupted file etc.
                log.warning("%s: could not load saved brain (%s) - starting a new one", symbol, exc)
        if brain is None:
            brain = Brain(symbol, feature_names, self.cfg.model, bar_minutes=self.cfg.timeframe_minutes,
                          outlook_horizons=self.outlook_horizons)
        self.brains[symbol] = brain
        return brain

    def learn(self, symbol: str, prep: Prepared, i: int) -> int:
        """Learn everything whose outcome is known once bar ``i`` has closed."""
        fs = prep.features
        brain = self.brain(symbol, fs.names)
        return brain.learn(fs.X, prep.times, fs.valid, prep.labels, prep.outlook_labels, i)

    def warmup(self, symbol: str, bars: pd.DataFrame, context: Optional[Mapping[str, pd.DataFrame]] = None,
               progress: bool = True) -> int:
        """Study history before trading: replays it day by day exactly as if it were live."""
        prep = self.prepare(symbol, bars, context)
        return self.learn_range(symbol, prep, len(bars) - 1, progress)

    def learn_range(self, symbol: str, prep: Prepared, last: int, progress: bool = True) -> int:
        n = 0
        step = max(1, int(1440 / self.cfg.timeframe_minutes))  # one day of bars per chunk
        stops = list(range(min(step, last), last, step)) + [last]
        report_every = max(1, len(stops) // 10)
        for k, stop in enumerate(stops, 1):
            n += self.learn(symbol, prep, stop)
            if progress and (k % report_every == 0 or k == len(stops)):
                log.info("%s: studying history... %3.0f%% (%s)", symbol, 100.0 * k / len(stops),
                         pd.Timestamp(int(prep.times[stop])).strftime("%Y-%m-%d"))
        return n

    # ------------------------------------------------------------------ main cycle
    def on_bar(self, symbol: str, prep: Prepared, i: Optional[int] = None) -> Decision:
        cfg = self.cfg
        i = len(prep.bars) - 1 if i is None else i
        info = self.infos[symbol]
        fs = prep.features
        bar_close = pd.Timestamp(int(prep.times[i])) + self.tf

        # 1) learn
        brain = self.brain(symbol, fs.names)
        brain.learn(fs.X, prep.times, fs.valid, prep.labels, prep.outlook_labels, i)

        # 2) account
        acct = self.broker.account()
        self.bind_account(acct)
        self.risk.update_equity(acct.equity, bar_close)
        self.sync_closed(bar_close)

        # 3) decide
        tick = self.broker.get_tick(symbol)
        decision = brain.decide(fs.X, prep.times, fs.valid, i)

        # 4) manage open trades (trailing stop, reversal, time limits)
        positions = self.broker.positions(symbol, cfg.trading.magic)
        if positions:
            self._manage(symbol, positions, prep, i, tick, bar_close, decision)

        # 5) execute
        if decision.arm is not None:
            blocked = self._gate(tick, bar_close, acct.equity)
            if blocked:
                decision.reason = f"blocked: {blocked}"
            else:
                self._execute(symbol, info, decision, tick, acct.equity, fs.atr_tf[decision.arm.tf][i], bar_close)
        self._journal_decision(symbol, bar_close, decision)
        return decision

    # ------------------------------------------------------------------ account binding
    def bind_account(self, acct: AccountInfo) -> None:
        """The risk state (equity peak, daily start, loss streak) belongs to one account. When you
        move from demo to live, it starts fresh - the learned brain is kept."""
        if self.account_login == acct.login:
            return
        if self.account_login is not None:
            log.warning("Account changed from %s to %s: starting a fresh risk state (the brain is kept)",
                        self.account_login, acct.login)
            self.risk = RiskManager(self.cfg.risk, RiskState())
            self.open_risk.clear()
            self.seen_closed.clear()
            self._seen_set.clear()
            self.last_sync = None
        self.account_login = acct.login

    # ------------------------------------------------------------------ steps
    def _bars_held(self, pos: Position, prep: Prepared, i: int) -> int:
        open_ns = int(pd.Timestamp(pos.time_open).value)
        open_bar = open_ns - (open_ns % self.tf_ns)
        first = int(np.searchsorted(prep.times[: i + 1], open_bar, side="left"))
        return i + 1 - first

    def _max_hold(self, pos: Position) -> int:
        meta = self.open_risk.get(pos.ticket) or {}
        arm = self.arm_by_key.get(meta.get("arm", ""))
        return arm.horizon if arm is not None else self.cfg.model.max_hold

    def _friday_cutoff(self, when: pd.Timestamp) -> bool:
        cut = self.cfg.trading.friday_cutoff_hour
        return cut is not None and when.dayofweek == 4 and when.hour >= cut

    def _manage(self, symbol, positions, prep: Prepared, i: int, tick: Tick, now: pd.Timestamp,
                decision: Optional[Decision] = None) -> None:
        cfg = self.cfg
        info = self.infos[symbol]
        for pos in positions:
            meta = self.open_risk.get(pos.ticket) or {}
            reason = None
            if self._bars_held(pos, prep, i) >= self._max_hold(pos):
                reason = "time-exit"
            elif self._friday_cutoff(now):
                reason = "friday-close"
            elif cfg.risk.exit_on_reversal and decision is not None and any(
                    e.eligible and e.arm.side is not pos.side for e in decision.evals):
                reason = "reversal"  # the brain is now confident the market goes the other way
            if reason:
                self._close(symbol, pos, reason, now)
                continue
            # dynamic stop: the most protective of the current stop, the trailing level and break-even
            levels = []
            trail = self._trail_level(pos, meta, prep, i, info)
            if trail is not None:
                levels.append((trail, "trail"))
            be = cfg.risk.breakeven_at_r
            if be is not None and meta.get("sl_dist"):
                cur = tick.bid if pos.side is Side.BUY else tick.ask
                if pos.side.value * (cur - pos.price_open) / meta["sl_dist"] >= be:
                    levels.append((pos.price_open, "breakeven"))
            if levels:
                pick = max if pos.side is Side.BUY else min
                level, why = pick(levels, key=lambda lv: lv[0])
                self._move_stop(symbol, pos, level, why, tick, info, now)

    def _close(self, symbol: str, pos: Position, reason: str, now: pd.Timestamp) -> None:
        if self.dry_run:
            log.info("%s DRY-RUN would close ticket %s (%s)", symbol, pos.ticket, reason)
            return
        res = self.broker.close_position(pos, comment=f"ai:{reason}", deviation=self.cfg.trading.deviation_points)
        self._event(now, symbol, reason, f"ticket {pos.ticket}: {'closed' if res.ok else res.message}")

    def _trail_level(self, pos: Position, meta: dict, prep: Prepared, i: int, info: SymbolInfo) -> Optional[float]:
        """Where a trailing stop should be after the last closed candle (None if not trailing yet).
        Mirrors the learning simulation in ``labels._label_trailing``."""
        arm = self.arm_by_key.get(meta.get("arm", ""))
        if arm is None or not arm.trail_atr or "trail_dist" not in meta:
            return None
        first = i + 1 - self._bars_held(pos, prep, i)
        if first > i:
            return None
        bars = prep.bars.iloc[first: i + 1]
        if pos.side is Side.BUY:
            best = float(bars["high"].max())
            if best - pos.price_open >= meta["start_dist"]:
                return best - meta["trail_dist"]
        else:
            spr = spread_prices(bars, info.point, self.cfg.backtest.default_spread_points)
            best = float((bars["low"].to_numpy(dtype=float) + spr).min())  # shorts are closed at the ask
            if pos.price_open - best >= meta["start_dist"]:
                return best + meta["trail_dist"]
        return None

    def _move_stop(self, symbol: str, pos: Position, level: float, why: str, tick: Tick,
                   info: SymbolInfo, now: pd.Timestamp) -> None:
        """Move the stop-loss on the broker's server - only ever in the trade's favour, and within
        the broker's minimum distance (stops level) and freeze rules."""
        new_sl = info.round_price(level)
        buy = pos.side is Side.BUY
        half_tick = info.tick_size / 2
        if pos.sl > 0 and ((buy and new_sl <= pos.sl + half_tick) or (not buy and new_sl >= pos.sl - half_tick)):
            return  # not an improvement
        cur = tick.bid if buy else tick.ask
        freeze = info.freeze_level * info.point
        if freeze > 0 and ((pos.sl > 0 and abs(cur - pos.sl) <= freeze) or (pos.tp > 0 and abs(pos.tp - cur) <= freeze)):
            return  # the broker does not allow changes this close to a level; retry next candle
        gap = (info.stops_level + 1) * info.point
        if (buy and new_sl > cur - gap) or (not buy and new_sl < cur + gap):
            # price has already come back through the new stop level: do what the stop would do
            self._close(symbol, pos, f"{why}-stop", now)
            return
        if self.dry_run:
            log.info("%s DRY-RUN would move SL of %s to %s (%s)", symbol, pos.ticket, new_sl, why)
            return
        res = self.broker.modify_position(pos, new_sl, pos.tp)
        self._event(now, symbol, why, f"ticket {pos.ticket}: SL {pos.sl} -> {new_sl}"
                    + ("" if res.ok else f" failed: {res.message}"))

    def _gate(self, tick: Tick, now: pd.Timestamp, equity: float) -> str:
        cfg = self.cfg
        ok, why = self.risk.can_trade(equity, now)
        if not ok:
            return why
        s = cfg.trading.session
        if not (s.start_hour <= now.hour < s.end_hour):
            return "outside trading session"
        if self._friday_cutoff(now):
            return "friday cut-off"
        if tick.time - now > self.tf * cfg.trading.stale_bar_tolerance:
            return "market data is stale (market closed?)"
        news = self.news.blocked(now)
        if news:
            return f"news blackout ({news})"
        return ""

    def _execute(self, symbol: str, info: SymbolInfo, decision: Decision, tick: Tick,
                 equity: float, atr_i: float, now: pd.Timestamp) -> None:
        cfg = self.cfg
        arm = decision.arm
        if arm.side is Side.BUY:
            price = tick.ask
            sl, tp = price - arm.sl_atr * atr_i, price + arm.tp_atr * atr_i
        else:
            price = tick.bid
            sl, tp = price + arm.sl_atr * atr_i, price - arm.tp_atr * atr_i
        sl, tp = info.round_price(sl), info.round_price(tp)
        # size from the stop exactly as it will be placed (after rounding to the tick size)
        sl_dist, tp_dist = abs(price - sl), abs(tp - price)
        n_sym = len(self.broker.positions(symbol, cfg.trading.magic))
        n_tot = len(self.broker.positions(None, cfg.trading.magic))
        ok, why = self.risk.check_trade(info, tick.spread, sl_dist, tp_dist, n_sym, n_tot)
        if ok:
            lots, risk_money = self.risk.position_size(equity, info, sl_dist)
            if lots <= 0:
                ok, why = False, "minimum lot size would exceed the risk budget"
        if not ok:
            decision.reason = f"blocked: {why}"
            return
        row = {"time": now, "account": self.account_login, "symbol": symbol, "side": arm.side.label,
               "volume": lots, "price": price, "sl": sl, "tp": tp, "risk_money": round(risk_money, 2),
               "prob": round(decision.prob, 4), "ev_r": round(decision.ev_r, 4), "arm": arm.key,
               "outlook": decision.outlook_text()}
        if self.dry_run:
            log.info("%s DRY-RUN %s %.2f lots @ %s SL %s TP %s [%s] (p=%.2f, EV=%+.2fR)", symbol, arm.side.label,
                     lots, price, sl, tp, arm.key, decision.prob, decision.ev_r)
            self.journal.write("orders", {**row, "status": "dry_run"})
            decision.reason = "dry run"
            return
        comment = f"ai {arm.side.label[0]} {arm.tf} p{decision.prob:.2f}"
        res = self.broker.open_position(symbol, arm.side, lots, sl, tp, cfg.trading.magic, comment,
                                        cfg.trading.deviation_points)
        if res.ok:
            if res.ticket is not None:
                meta = {"symbol": symbol, "risk_money": risk_money, "sl_dist": sl_dist, "arm": arm.key}
                if arm.trail_atr:
                    meta["trail_dist"] = arm.trail_atr * atr_i
                    meta["start_dist"] = arm.trail_start_r * sl_dist
                self.open_risk[int(res.ticket)] = meta
            log.info("%s OPEN %s %.2f lots @ %s SL %s TP %s [%s, hold <= %d bars] (p=%.2f, EV=%+.2fR, risk %.2f)",
                     symbol, arm.side.label, lots, res.price, sl, tp, arm.key, arm.horizon, decision.prob,
                     decision.ev_r, risk_money)
            decision.reason = "opened"
        else:
            log.warning("%s order rejected: %s", symbol, res.message)
            decision.reason = f"order rejected: {res.message}"
        self.journal.write("orders", {**row, "price": res.price or price, "ticket": res.ticket,
                                      "status": "filled" if res.ok else "rejected", "message": res.message})

    def sync_closed(self, now: pd.Timestamp) -> None:
        since = (self.last_sync or now - pd.Timedelta(days=1)) - pd.Timedelta(hours=1)
        trades = self.broker.closed_trades(since, self.cfg.trading.magic, list(self.open_risk))
        for t in sorted(trades, key=lambda x: x.time_close):
            if t.position_id in self._seen_set:
                continue
            if len(self.seen_closed) == self.seen_closed.maxlen:
                self._seen_set.discard(self.seen_closed[0])
            self.seen_closed.append(t.position_id)
            self._seen_set.add(t.position_id)
            meta = self.open_risk.pop(t.position_id, None)
            if meta and meta.get("risk_money", 0) > 0:
                r = t.profit / meta["risk_money"]
            else:
                approx = max(1e-9, self.risk.state.peak_equity or 1.0) * self.cfg.risk.risk_per_trade_pct / 100
                r = t.profit / approx
            self.risk.record_trade(r, t.time_close)
            if meta and meta.get("arm") and meta.get("symbol") in self.brains:
                self.brains[meta["symbol"]].record_real_trade(meta["arm"], r)
            self.journal.write("closed", {"time": t.time_close, "account": self.account_login, "symbol": t.symbol,
                                          "ticket": t.position_id, "side": t.side.label if t.side else "",
                                          "volume": t.volume, "price_close": t.price_close,
                                          "profit": round(t.profit, 2), "r_multiple": round(r, 3),
                                          "arm": (meta or {}).get("arm", ""), "reason": t.reason})
            log.info("%s CLOSED ticket %s: %+.2f (%+.2fR, %s)", t.symbol, t.position_id, t.profit, r, t.reason)
        self.last_sync = now

    # ------------------------------------------------------------------ journal / state
    def _journal_decision(self, symbol: str, now: pd.Timestamp, d: Decision) -> None:
        if not self.journal.log_decisions or self.journal.dir is None:
            return
        buy = [e.prob for e in d.evals if e.arm.side is Side.BUY]
        sell = [e.prob for e in d.evals if e.arm.side is Side.SELL]
        row = {
            "time": now, "symbol": symbol, "action": d.action, "arm": d.arm.key if d.arm else "",
            "prob": round(d.prob, 4), "ev_r": round(d.ev_r, 4), "reason": d.reason,
            "best_buy_p": round(max(buy), 4) if buy else "", "best_sell_p": round(max(sell), 4) if sell else "",
        }
        for tf, p in d.outlook.items():
            row[f"up_{tf}"] = round(p, 4)
        self.journal.write("decisions", row)

    def _event(self, now, symbol, event, detail) -> None:
        log.info("%s %s: %s", symbol, event, detail)
        self.journal.write("events", {"time": now, "symbol": symbol, "event": event, "detail": detail})

    def _brain_path(self, symbol: str) -> Optional[Path]:
        if self.state_dir is None:
            return None
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in symbol)
        return self.state_dir / f"brain_{safe}_{self.cfg.trading.timeframe}.joblib"

    def _state_path(self) -> Optional[Path]:
        return None if self.state_dir is None else self.state_dir / "bot_state.json"

    def _load_state(self) -> Optional[RiskState]:
        path = self._state_path()
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Could not read %s (%s); starting with a clean risk state", path, exc)
            return None
        self.account_login = data.get("account")
        self.open_risk = {int(k): v for k, v in data.get("open_risk", {}).items()}
        for pid in data.get("seen_closed", []):
            self.seen_closed.append(int(pid))
            self._seen_set.add(int(pid))
        if data.get("last_sync"):
            self.last_sync = pd.Timestamp(data["last_sync"])
        return RiskState.from_dict(data.get("risk", {}))

    def save_state(self, models: bool = True) -> None:
        """Persist the risk state (small) and, if ``models``, the brains (larger). Once a day a
        dated copy of each brain is kept in ``state/backups`` so a good brain can be restored."""
        if self.state_dir is None:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if models:
            for symbol, brain in self.brains.items():
                path = self._brain_path(symbol)
                brain.save(path)
                self._backup(path, brain)
        data = {
            "account": self.account_login,
            "risk": self.risk.state.to_dict(),
            "open_risk": {str(k): v for k, v in self.open_risk.items()},
            "seen_closed": list(self.seen_closed)[-1000:],
            "last_sync": self.last_sync.isoformat() if self.last_sync is not None else None,
        }
        path = self._state_path()
        fd, tmp = tempfile.mkstemp(dir=str(self.state_dir), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp, path)

    def _backup(self, path: Path, brain: Brain) -> None:
        keep = self.cfg.model.backup_days
        day = brain._day
        if keep <= 0 or not day:
            return
        backups = self.state_dir / "backups"
        target = backups / f"{path.stem}_{day}{path.suffix}"
        if target.exists():
            return
        backups.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        olds = sorted(backups.glob(f"{path.stem}_*{path.suffix}"))
        for old in olds[:-keep]:
            old.unlink()
