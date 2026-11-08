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
    start_hour: int = 0  # broker server time, inclusive
    end_hour: int = 24  # broker server time, exclusive


@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: ["EURUSD"])
    timeframe: str = "M5"
    magic: int = 285001
    dry_run: bool = False  # log decisions but never send orders
    poll_seconds: float = 5.0
    lookback_bars: int = 1500  # bars fetched each cycle to compute indicators
    warmup_bars: int = 20000  # history used to pre-train the model at start-up
    deviation_points: int = 20
    comment: str = "ai-bot"
    session: SessionConfig = field(default_factory=SessionConfig)
    friday_cutoff_hour: Optional[int] = 20  # no new trades + flatten after this hour on Fridays
    stale_bar_tolerance: int = 3  # skip trading if the last closed bar is older than N bars
    save_models_every: int = 12  # save the learned models every N bars (and always on exit)
    # Round-turn commission per 1.0 lot charged by your broker (account currency). Used to make the
    # learning labels cost-aware and by the backtest simulator. 0 for spread-only accounts.
    commission_per_lot: float = 0.0


@dataclass
class ExitProfile:
    sl_atr: float
    tp_atr: float

    @property
    def name(self) -> str:
        return f"sl{self.sl_atr:g}_tp{self.tp_atr:g}"


@dataclass
class ModelConfig:
    horizon_bars: int = 24  # max holding time; also the label horizon
    atr_period: int = 14
    exit_profiles: List[ExitProfile] = field(
        default_factory=lambda: [
            ExitProfile(1.0, 1.5),
            ExitProfile(1.5, 2.5),
            ExitProfile(2.0, 4.0),
        ]
    )
    min_train_samples: int = 1500  # learned samples before an arm may trade
    min_eval_samples: int = 1000  # out-of-sample evaluations before skill is trusted
    min_skill: float = 0.005  # required improvement over the base rate (log-loss skill score)
    min_ev_r: float = 0.10  # minimum expected value, in R (multiples of the risk)
    min_prob: float = 0.0  # optional extra filter on win probability
    # Paper trading: every signal is also traded on paper (one at a time per setup). A setup may
    # only trade for real while its last ``paper_window`` paper trades made money on average with
    # some statistical confidence (t-statistic of the mean R).
    paper_window: int = 40
    min_paper_trades: int = 20
    min_paper_avg_r: float = 0.0
    min_paper_tstat: float = 1.5
    loss_decay: float = 0.995  # memory of the expert weights (~200 samples: adapts quickly)
    skill_decay: float = 0.999  # memory of the skill score that gates trading (~1000 samples)
    hedge_eta: float = 1.0
    logit_lr: float = 0.01
    logit_l2: float = 1e-4
    scaler_span: int = 3000
    gbm_enabled: bool = True
    gbm_window: int = 3000  # most recent samples used for each re-fit
    gbm_refit_every: int = 288  # re-fit cadence in learned samples (288 x M5 = 1 day)
    gbm_max_iter: int = 100
    gbm_learning_rate: float = 0.05
    gbm_max_depth: int = 3  # shallow trees: market data is noisy, deep trees memorise noise
    gbm_min_samples_leaf: int = 40
    n_threads: int = 1  # CPU threads for the gradient-boosting model
    outcome_alpha: float = 0.01  # EWMA rate for the average win / loss R of each arm
    slippage_points: float = 1.0  # extra cost assumed per trade when labelling
    seed: int = 7


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


@dataclass
class BacktestConfig:
    initial_balance: float = 10_000.0
    default_spread_points: float = 10.0  # used if the data has no spread column
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
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @property
    def timeframe_minutes(self) -> int:
        return TIMEFRAME_MINUTES[self.trading.timeframe]

    def validate(self) -> "BotConfig":
        t, m, r = self.trading, self.model, self.risk
        if t.timeframe not in TIMEFRAME_MINUTES:
            raise ConfigError(f"Unknown timeframe {t.timeframe!r}; use one of {list(TIMEFRAME_MINUTES)}")
        if not t.symbols:
            raise ConfigError("trading.symbols must list at least one symbol")
        if not m.exit_profiles:
            raise ConfigError("model.exit_profiles must not be empty")
        for p in m.exit_profiles:
            if p.sl_atr <= 0 or p.tp_atr <= 0:
                raise ConfigError("exit profile multiples must be positive")
        if m.horizon_bars < 1:
            raise ConfigError("model.horizon_bars must be >= 1")
        if t.lookback_bars < 400 + m.horizon_bars:
            raise ConfigError("trading.lookback_bars is too small for the indicators (need >= 400 + horizon)")
        if not 0 < r.risk_per_trade_pct <= 5:
            raise ConfigError("risk.risk_per_trade_pct must be in (0, 5]")
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
        raise ConfigError(f"Unknown key(s) in {where}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        sub = _SUBSECTIONS.get((cls, name))
        if sub is not None:
            kwargs[name] = _build(sub, value, f"{where}.{name}")
        elif (cls, name) == (ModelConfig, "exit_profiles"):
            kwargs[name] = [_build(ExitProfile, p, f"{where}.exit_profiles[{i}]") for i, p in enumerate(value)]
        else:
            kwargs[name] = value
    return cls(**kwargs)


_SUBSECTIONS = {
    (BotConfig, "mt5"): MT5Config,
    (BotConfig, "trading"): TradingConfig,
    (BotConfig, "model"): ModelConfig,
    (BotConfig, "risk"): RiskConfig,
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
