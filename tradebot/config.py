"""Configuration: typed dataclasses loaded from YAML, with env-var overrides for credentials."""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

TIMEFRAME_MINUTES = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6, "M10": 10, "M12": 12,
    "M15": 15, "M20": 20, "M30": 30, "H1": 60, "H2": 120, "H3": 180, "H4": 240,
    "H6": 360, "H8": 480, "H12": 720, "D1": 1440,
}


class ConfigError(ValueError):
    pass


@dataclass
class MT5Config:
    path: Optional[str] = None  # terminal64.exe; None = auto-detect the installed terminal
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    timeout_ms: int = 60_000
    # Safety switch: the bot refuses to send orders on a real-money account unless this is true.
    allow_live_account: bool = False


@dataclass
class SessionConfig:
    # Broker *server* time. Gold spreads jump around the daily roll-over at server midnight, so
    # the default skips the first and last hour of the server day.
    start_hour: int = 1  # inclusive
    end_hour: int = 23  # exclusive


@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: ["XAUUSD"])
    timeframe: str = "M5"  # the bot decides once per closed bar of this chart
    magic: int = 285001
    dry_run: bool = False  # log decisions but never send orders
    poll_seconds: float = 5.0
    lookback_bars: int = 1500  # trading-timeframe bars fetched each cycle
    warmup_bars: int = 20000  # history (trading-timeframe bars) studied at start-up
    deviation_points: int = 30
    comment: str = "ai-bot"
    session: SessionConfig = field(default_factory=SessionConfig)
    friday_cutoff_hour: Optional[int] = 20  # no new trades + flatten after this hour on Fridays
    stale_bar_tolerance: int = 3  # skip trading if the last closed bar is older than N bars
    save_models_every: int = 12  # save the learned models every N bars (and always on exit)
    # Round-turn commission per 1.0 lot charged by your broker (account currency). Used to make the
    # learning labels cost-aware and by the backtest simulator. 0 for spread-only accounts.
    commission_per_lot: float = 0.0
    # Optional CSV of high-impact news (columns: time [server time], optional title,
    # minutes_before, minutes_after). No new trades are opened inside those windows.
    news_file: Optional[str] = None
    news_minutes_before: int = 15
    news_minutes_after: int = 15


@dataclass
class TradeStyle:
    """One way of trading the chart: stop-loss / take-profit measured in the ATR of ``timeframe``
    and a maximum holding time in trading-timeframe bars.

    With ``trail_atr`` set, the stop-loss *trails* the price: once the trade is ``trail_start_r``
    R in profit, after every candle the stop is moved to ``trail_atr`` ATR behind the best price
    reached (never backwards). The learning simulates exactly this, so the brain knows what a
    trailing exit is worth and chooses between fixed and trailing styles itself."""

    timeframe: str = "M5"
    sl_atr: float = 1.0
    tp_atr: float = 1.5
    max_hold_bars: int = 24
    trail_atr: Optional[float] = None
    trail_start_r: float = 1.0

    @property
    def name(self) -> str:
        trail = f"_tr{self.trail_atr:g}" if self.trail_atr else ""
        return f"{self.timeframe}:sl{self.sl_atr:g}_tp{self.tp_atr:g}{trail}"


def _gold_styles() -> List[TradeStyle]:
    # An H4-sized style (e.g. TradeStyle("H4", 1.0, 1.5, 288): up to 24 hours) can be added in the
    # config, but its outcomes arrive a day late and overlap heavily, so it learns poorly; the H4
    # chart is still read for every decision and forecast every 5 minutes.
    return [
        TradeStyle("M5", 1.5, 2.0, 12),    # scalp: up to 1 hour
        TradeStyle("M30", 1.0, 1.5, 48),   # intraday: up to 4 hours
        TradeStyle("H1", 1.0, 2.0, 96),    # intraday swing: up to 8 hours
        # Trailing variants (stop follows the price once +1R) are supported, e.g.
        # TradeStyle("M30", 1.0, 3.0, 48, trail_atr=1.0, trail_start_r=1.0); they are not on by
        # default because they did not improve the validation backtests (see README).
    ]


@dataclass
class ModelConfig:
    atr_period: int = 14
    # Charts read in addition to the trading timeframe. M1 shows every small movement inside each
    # 5-minute candle; M30/H1/H4 show the bigger picture.
    context_timeframes: List[str] = field(default_factory=lambda: ["M1", "M30", "H1", "H4"])
    context_bars: int = 600  # extra bars fetched per context chart for its indicators
    # Direction forecasts ("will price be higher in 5 min / 30 min / 1 h / 4 h?") made every bar
    # and scored when the answer is known - the bot's report card per timeframe.
    outlooks: List[str] = field(default_factory=lambda: ["M5", "M30", "H1", "H4"])
    styles: List[TradeStyle] = field(default_factory=_gold_styles)
    min_train_samples: int = 1500  # learned samples before a setup may trade
    min_eval_samples: int = 1000  # out-of-sample evaluations before skill is trusted
    min_skill: float = 0.005  # required improvement over the base rate (log-loss skill score)
    min_ev_r: float = 0.10  # minimum expected value, in R (multiples of the risk)
    min_prob: float = 0.0  # optional extra filter on win probability
    # Paper trading: every signal is also traded on paper (one at a time per setup). A setup may
    # only trade for real while its last ``paper_window`` trades made money on average with some
    # statistical confidence (t-statistic of the mean R).
    paper_window: int = 40
    min_paper_trades: int = 20
    min_paper_avg_r: float = 0.0
    min_paper_tstat: float = 1.5
    loss_decay: float = 0.995  # memory of the expert weights (~200 samples: adapts quickly)
    skill_decay: float = 0.999  # memory of the skill score that gates trading (~1000 samples)
    hedge_eta: float = 0.3  # how sharply the experts' weights follow recent accuracy
    logit_lr: float = 0.002
    logit_l2: float = 1e-4
    scaler_span: int = 3000
    gbm_enabled: bool = True
    gbm_window: int = 3000  # most recent samples used for each re-fit
    gbm_refit_every: int = 288  # re-fit cadence in learned samples (288 x M5 = 1 day)
    gbm_max_iter: int = 100
    gbm_learning_rate: float = 0.05
    gbm_max_depth: int = 3  # shallow trees: market data is noisy, deep trees memorise noise
    gbm_min_samples_leaf: int = 40
    gbm_max_features: float = 0.5  # fraction of features considered per split
    n_threads: int = 1  # CPU threads for the gradient-boosting model
    outcome_alpha: float = 0.01  # EWMA rate for the average win / loss R of each setup
    slippage_points: float = 3.0  # extra cost assumed per trade when labelling
    backup_days: int = 7  # daily copies of the brain kept in state/backups
    seed: int = 7

    @property
    def max_hold(self) -> int:
        return max(s.max_hold_bars for s in self.styles)


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.5  # % of equity lost if the stop loss is hit
    max_positions_per_symbol: int = 1
    max_total_positions: int = 3
    max_daily_loss_pct: float = 3.0  # halt new trades for the rest of the day
    max_drawdown_pct: float = 15.0  # kill switch from the equity peak (manual reset)
    max_spread_sl_fraction: float = 0.25  # skip if spread > this fraction of the SL distance
    max_spread_points: Optional[float] = None
    max_consecutive_losses: int = 4
    cooldown_minutes: int = 60
    adaptive_window: int = 10  # recent trades used to scale risk down after losses
    min_risk_multiplier: float = 0.25
    max_lots: float = 5.0
    breakeven_at_r: Optional[float] = None  # e.g. 1.0 => move SL to entry after +1R
    # Close an open trade when the brain becomes confident the market will go the other way
    # (an opposite setup passes every gate). The opposite trade may then be opened.
    exit_on_reversal: bool = True
    # SMALL-ACCOUNT MODE (off when null). On a small account even the minimum lot (0.01) can risk
    # far more than risk_per_trade_pct. With this set, the bot may still trade the minimum lot
    # as long as its loss at the stop is at most this % of equity. You accept that one losing
    # trade costs up to this much of the account. Normal % sizing resumes once the account is
    # big enough.
    min_lot_max_risk_pct: Optional[float] = None


@dataclass
class ReadinessConfig:
    """What a demo record must show before ``tradebot readiness`` says "ready for live"."""

    min_days: int = 14
    min_trades: int = 30
    min_profit_factor: float = 1.2
    min_avg_r: float = 0.0
    max_drawdown_pct: float = 10.0


@dataclass
class BacktestConfig:
    initial_balance: float = 10_000.0
    default_spread_points: float = 25.0  # used if the data has no spread column
    warmup_bars: int = 6000  # bars used only for learning before the simulation trades
    symbol_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PathsConfig:
    state_dir: str = "state"
    log_dir: str = "logs"


@dataclass
class BotConfig:
    mt5: MT5Config = field(default_factory=MT5Config)
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    readiness: ReadinessConfig = field(default_factory=ReadinessConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @property
    def timeframe_minutes(self) -> int:
        return TIMEFRAME_MINUTES[self.trading.timeframe]

    def context_fetch_bars(self, tf: str, base_bars: int) -> int:
        """Bars of context timeframe ``tf`` that cover ``base_bars`` trading bars plus indicator history."""
        span_minutes = base_bars * self.timeframe_minutes
        return int(span_minutes // TIMEFRAME_MINUTES[tf]) + self.model.context_bars

    def outlook_horizons(self) -> Dict[str, int]:
        return {tf: TIMEFRAME_MINUTES[tf] // self.timeframe_minutes for tf in self.model.outlooks}

    def validate(self) -> "BotConfig":
        t, m, r = self.trading, self.model, self.risk
        if t.timeframe not in TIMEFRAME_MINUTES:
            raise ConfigError(f"Unknown timeframe {t.timeframe!r}; use one of {list(TIMEFRAME_MINUTES)}")
        if not t.symbols:
            raise ConfigError("trading.symbols must list at least one symbol")
        base = t.timeframe
        for tf in m.context_timeframes:
            if tf not in TIMEFRAME_MINUTES:
                raise ConfigError(f"model.context_timeframes: unknown timeframe {tf!r}")
            if tf == base:
                raise ConfigError(f"model.context_timeframes must not repeat the trading timeframe {base}")
        if len(set(m.context_timeframes)) != len(m.context_timeframes):
            raise ConfigError("model.context_timeframes has duplicates")
        if not m.styles:
            raise ConfigError("model.styles must not be empty")
        for s in m.styles:
            if s.timeframe != base and s.timeframe not in m.context_timeframes:
                raise ConfigError(f"style timeframe {s.timeframe} must be {base} or one of model.context_timeframes")
            if s.sl_atr <= 0 or s.tp_atr <= 0 or s.max_hold_bars < 1:
                raise ConfigError("style sl_atr / tp_atr must be > 0 and max_hold_bars >= 1")
            if s.trail_atr is not None and (s.trail_atr <= 0 or s.trail_start_r < 0):
                raise ConfigError("style trail_atr must be > 0 and trail_start_r >= 0")
        if len({s.name for s in m.styles}) != len(m.styles):
            raise ConfigError("model.styles has duplicates")
        for tf in m.outlooks:
            minutes = TIMEFRAME_MINUTES.get(tf)
            if minutes is None or minutes < self.timeframe_minutes or minutes % self.timeframe_minutes:
                raise ConfigError(f"model.outlooks: {tf!r} must be a timeframe that is a multiple of {base}")
        if t.lookback_bars < 300 + m.max_hold:
            raise ConfigError(f"trading.lookback_bars must be >= 300 + the longest max_hold_bars ({300 + m.max_hold})")
        if not 0 < r.risk_per_trade_pct <= 5:
            raise ConfigError("risk.risk_per_trade_pct must be in (0, 5]")
        if r.min_lot_max_risk_pct is not None and not 0 < r.min_lot_max_risk_pct <= 100:
            raise ConfigError("risk.min_lot_max_risk_pct must be in (0, 100] or null")
        if not 0 < m.loss_decay < 1 or not 0 < m.skill_decay < 1:
            raise ConfigError("model.loss_decay and model.skill_decay must be in (0, 1)")
        if not 0 <= t.session.start_hour <= 24 or not 0 <= t.session.end_hour <= 24:
            raise ConfigError("trading.session hours must be within 0..24")
        return self


def _build(cls, data: Optional[dict], where: str):
    """Recursively build a dataclass from a dict, rejecting unknown keys (catches typos)."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{where} must be a mapping")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        hint = " (exit_profiles/horizon_bars were replaced by model.styles)" if {"exit_profiles", "horizon_bars"} & unknown else ""
        raise ConfigError(f"Unknown key(s) in {where}: {sorted(unknown)}{hint}")
    kwargs = {}
    for name, value in data.items():
        sub = _SUBSECTIONS.get((cls, name))
        if sub is not None:
            kwargs[name] = _build(sub, value, f"{where}.{name}")
        elif (cls, name) == (ModelConfig, "styles"):
            kwargs[name] = [_build(TradeStyle, p, f"{where}.styles[{i}]") for i, p in enumerate(value)]
        else:
            kwargs[name] = value
    return cls(**kwargs)


_SUBSECTIONS = {
    (BotConfig, "mt5"): MT5Config,
    (BotConfig, "trading"): TradingConfig,
    (BotConfig, "model"): ModelConfig,
    (BotConfig, "risk"): RiskConfig,
    (BotConfig, "readiness"): ReadinessConfig,
    (BotConfig, "backtest"): BacktestConfig,
    (BotConfig, "paths"): PathsConfig,
    (TradingConfig, "session"): SessionConfig,
}


def apply_env_overrides(cfg: BotConfig, env: Optional[Dict[str, str]] = None) -> BotConfig:
    env = os.environ if env is None else env
    if env.get("MT5_LOGIN"):
        cfg.mt5.login = int(env["MT5_LOGIN"])
    if env.get("MT5_PASSWORD"):
        cfg.mt5.password = env["MT5_PASSWORD"]
    if env.get("MT5_SERVER"):
        cfg.mt5.server = env["MT5_SERVER"]
    if env.get("MT5_PATH"):
        cfg.mt5.path = env["MT5_PATH"]
    return cfg


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: Optional[str] = None) -> BotConfig:
    data: dict = {}
    if path:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        load_dotenv(p.parent / ".env")
    load_dotenv(Path.cwd() / ".env")
    cfg = _build(BotConfig, data, "config")
    if cfg.mt5.login is not None:
        cfg.mt5.login = int(cfg.mt5.login)
    return apply_env_overrides(cfg).validate()
