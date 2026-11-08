"""Live trading loop: waits for each new closed bar on MT5 and runs the bot cycle on it."""
from __future__ import annotations

import logging
import signal
import time
from typing import Dict, Optional

import pandas as pd

from .brokers.base import Broker, BrokerError
from .config import BotConfig
from .core import BotCore
from .journal import Journal

log = logging.getLogger(__name__)


class LiveEngine:
    def __init__(self, cfg: BotConfig, broker: Broker, dry_run: bool = False):
        self.cfg = cfg
        self.broker = broker
        self.dry_run = dry_run or cfg.trading.dry_run
        self.journal = Journal(cfg.paths.log_dir, log_decisions=True)
        self.core: Optional[BotCore] = None
        self._running = False
        self._last_bar: Dict[str, pd.Timestamp] = {}
        self._bars_seen: Dict[str, int] = {}

    def stop(self, *_):
        log.info("Stop requested - finishing the current cycle...")
        self._running = False

    def start(self) -> BotCore:
        cfg = self.cfg
        self.broker.connect()
        acct = self.broker.account()
        log.info("Account %s | balance %.2f %s | equity %.2f | %s", acct.login, acct.balance, acct.currency,
                 acct.equity, "DEMO" if acct.is_demo else "REAL MONEY")
        if not acct.is_demo and not cfg.mt5.allow_live_account and not self.dry_run:
            raise BrokerError(
                "This is a REAL-MONEY account. Trading it requires `mt5.allow_live_account: true` in the "
                "config. Run on a demo account first (or use --dry-run)."
            )
        if self.dry_run:
            log.info("DRY-RUN mode: decisions are logged, no orders will be sent")
        core = BotCore(cfg, self.broker, journal=self.journal, state_dir=cfg.paths.state_dir, dry_run=self.dry_run)
        for symbol in cfg.trading.symbols:
            info = core.add_symbol(symbol)
            log.info("%s: point %g, digits %d, tick value %g, lots %g..%g step %g, stops level %d",
                     symbol, info.point, info.digits, info.tick_value, info.volume_min, info.volume_max,
                     info.volume_step, info.stops_level)
            bars = self.broker.get_rates(symbol, cfg.trading.timeframe, cfg.trading.warmup_bars)
            if len(bars) < cfg.model.min_train_samples + cfg.model.horizon_bars + 300:
                log.warning("%s: only %d bars of history available - the model needs time to learn before it "
                            "trades. Increase 'Max bars in chart' in MT5 (Tools > Options > Charts).", symbol, len(bars))
            core.warmup(symbol, bars)
            self._last_bar[symbol] = bars["time"].iat[-1]
            self._log_model(core, symbol)
        core.save_state()
        self.core = core
        return core

    def run(self, max_cycles: Optional[int] = None) -> None:
        core = self.core or self.start()
        cfg = self.cfg
        self._running = True
        try:
            signal.signal(signal.SIGINT, self.stop)
            signal.signal(signal.SIGTERM, self.stop)
        except (ValueError, AttributeError):  # not in main thread / platform without SIGTERM
            pass
        log.info("Bot running on %s %s - waiting for each new bar (Ctrl+C to stop)",
                 ",".join(cfg.trading.symbols), cfg.trading.timeframe)
        cycles = 0
        try:
            while self._running:
                if not self.broker.ensure_connected():
                    time.sleep(max(cfg.trading.poll_seconds, 10))
                    continue
                processed = False
                for symbol in cfg.trading.symbols:
                    try:
                        processed |= self._process_symbol(core, symbol)
                    except BrokerError as exc:
                        log.error("%s: broker error: %s", symbol, exc)
                    except Exception:
                        log.exception("%s: unexpected error (continuing)", symbol)
                if processed:
                    cycles += 1
                    try:  # risk state every bar, the (larger) models every `save_models_every` bars
                        core.save_state(models=cycles % max(1, cfg.trading.save_models_every) == 0)
                    except Exception:
                        log.exception("Could not save state")
                    if max_cycles is not None and cycles >= max_cycles:
                        break
                time.sleep(cfg.trading.poll_seconds)
        finally:
            try:
                core.save_state()
            finally:
                self.journal.close()
                self.broker.shutdown()
                log.info("Bot stopped")

    def _process_symbol(self, core: BotCore, symbol: str) -> bool:
        cfg = self.cfg
        latest = self.broker.get_rates(symbol, cfg.trading.timeframe, 1)
        if latest.empty:
            return False
        bar_time = latest["time"].iat[-1]
        if self._last_bar.get(symbol) is not None and bar_time <= self._last_bar[symbol]:
            return False
        bars = self.broker.get_rates(symbol, cfg.trading.timeframe, cfg.trading.lookback_bars)
        core.add_symbol(symbol)  # refresh the contract spec (tick value moves with FX rates)
        prep = core.prepare(symbol, bars)
        decision = core.on_bar(symbol, prep)
        self._last_bar[symbol] = bars["time"].iat[-1]
        best = max(decision.evals, key=lambda e: e.ev_r) if decision.evals else None
        log.info("%s bar %s | %s | %s%s", symbol, bar_time, decision.action, decision.reason,
                 f" | best {best.arm.key} p={best.prob:.2f} EV={best.ev_r:+.2f}R" if best else "")
        self._bars_seen[symbol] = self._bars_seen.get(symbol, 0) + 1
        if self._bars_seen[symbol] % 144 == 0:  # periodic model report
            self._log_model(core, symbol)
        return True

    @staticmethod
    def _log_model(core: BotCore, symbol: str) -> None:
        for key, s in core.brains[symbol].summary().items():
            log.info("%s model %-17s samples %6d | TP rate %.2f skill %s | weights %s | paper trades %d, "
                     "recent avg %+.2fR (t %.1f)", symbol, key, s["samples"], s["tp_rate"], s["skill_tp"],
                     s["weights_tp(b/l/g)"], s["paper_trades"], s["paper_recent_avg_r"], s["paper_recent_tstat"])
