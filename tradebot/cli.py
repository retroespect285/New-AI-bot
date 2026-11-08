"""Command line: ``python -m tradebot <command>``.

Commands
  check     connect to MT5 and print account + symbol details (no trading)
  live      run the bot on MT5 (use a DEMO account first; --dry-run sends no orders)
  backtest  walk-forward backtest on a CSV export or synthetic data
  export    download bar history from MT5 to CSV (for backtesting)
  status    print what the saved models have learned so far
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import BotConfig, load_config
from .types import SymbolInfo

log = logging.getLogger("tradebot")

DEFAULT_SYMBOL_INFO = {
    # Typical values for a USD account; `export` saves your broker's exact spec next to the CSV.
    "EURUSD": dict(point=1e-5, digits=5, tick_size=1e-5, tick_value=1.0, contract_size=100000),
    "GBPUSD": dict(point=1e-5, digits=5, tick_size=1e-5, tick_value=1.0, contract_size=100000),
    "AUDUSD": dict(point=1e-5, digits=5, tick_size=1e-5, tick_value=1.0, contract_size=100000),
    "USDJPY": dict(point=1e-3, digits=3, tick_size=1e-3, tick_value=0.67, contract_size=100000),
    "XAUUSD": dict(point=0.01, digits=2, tick_size=0.01, tick_value=1.0, contract_size=100),
}


def setup_logging(log_dir: str, verbose: bool = False, to_file: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    if to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(Path(log_dir) / "bot.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _symbol_info(cfg: BotConfig, symbol: str, csv_path: str = None) -> SymbolInfo:
    from .data import load_symbol_info

    if csv_path:
        info = load_symbol_info(csv_path)
        if info is not None:
            return info
    spec = dict(cfg.backtest.symbol_info.get(symbol) or DEFAULT_SYMBOL_INFO.get(symbol.upper()) or {})
    if not spec:
        log.warning("No contract spec for %s - assuming a 5-digit FX pair. Add it under backtest.symbol_info "
                    "or use `export` to save the broker's spec.", symbol)
        spec = dict(DEFAULT_SYMBOL_INFO["EURUSD"])
    spec.setdefault("tick_size", spec.get("point"))
    return SymbolInfo(name=symbol, **spec)


def cmd_check(cfg: BotConfig, args) -> int:
    from .brokers.mt5 import MT5Broker

    broker = MT5Broker(cfg.mt5)
    broker.connect()
    try:
        acct = broker.account()
        print(f"Account {acct.login}: balance {acct.balance:.2f} {acct.currency}, equity {acct.equity:.2f}, "
              f"{'DEMO' if acct.is_demo else 'REAL MONEY'}, trading allowed: {acct.trade_allowed}")
        for symbol in cfg.trading.symbols:
            info = broker.symbol_info(symbol)
            tick = broker.get_tick(symbol)
            bars = broker.get_rates(symbol, cfg.trading.timeframe, cfg.trading.warmup_bars)
            print(f"{symbol}: bid {tick.bid} ask {tick.ask} spread {tick.spread / info.point:.0f}pts | "
                  f"{len(bars)} closed {cfg.trading.timeframe} bars available "
                  f"({bars['time'].iat[0]} -> {bars['time'].iat[-1]}) | spec {info.to_dict()}")
        positions = broker.positions(magic=cfg.trading.magic)
        print(f"Open positions of this bot (magic {cfg.trading.magic}): {len(positions)}")
    finally:
        broker.shutdown()
    return 0


def cmd_live(cfg: BotConfig, args) -> int:
    from .brokers.mt5 import MT5Broker
    from .engine import LiveEngine

    engine = LiveEngine(cfg, MT5Broker(cfg.mt5), dry_run=args.dry_run)
    engine.run(max_cycles=args.max_cycles)
    return 0


def cmd_export(cfg: BotConfig, args) -> int:
    from .brokers.mt5 import MT5Broker
    from .data import save_bars

    broker = MT5Broker(cfg.mt5)
    broker.connect()
    try:
        for symbol in args.symbol or cfg.trading.symbols:
            info = broker.symbol_info(symbol)
            bars = broker.get_rates(symbol, cfg.trading.timeframe, args.bars)
            path = Path(args.out) / f"{symbol}_{cfg.trading.timeframe}.csv"
            save_bars(bars, str(path), info)
            print(f"Saved {len(bars)} bars to {path} (+ contract spec {path.with_suffix('.json').name})")
    finally:
        broker.shutdown()
    return 0


def cmd_backtest(cfg: BotConfig, args) -> int:
    from .backtest import run_backtest
    from .data import load_bars, synthetic_bars

    if args.csv:
        bars = load_bars(args.csv)
        symbol = args.symbol[0] if args.symbol else Path(args.csv).stem.split("_")[0]
    else:
        symbol = args.symbol[0] if args.symbol else "SYNTH"
        bars = synthetic_bars(args.synthetic, seed=args.seed, minutes=cfg.timeframe_minutes)
        log.info("Using %d synthetic bars (pipeline demo only - not evidence of real-market performance)", len(bars))
    if args.bars:
        bars = bars.iloc[-args.bars:].reset_index(drop=True)
    info = _symbol_info(cfg, symbol, args.csv)
    out = Path(args.out) / f"{symbol}_{cfg.trading.timeframe}"
    out.mkdir(parents=True, exist_ok=True)
    log.info("Backtesting %s on %d bars (%s -> %s); output in %s", symbol, len(bars),
             bars["time"].iat[0], bars["time"].iat[-1], out)
    res = run_backtest(cfg, bars, symbol, info, out_dir=str(out), warmup_bars=args.warmup)
    stats = dict(res.stats)
    model = stats.pop("model")
    print("\n=== Backtest summary ===")
    for k, v in stats.items():
        print(f"{k:>22}: {v}")
    print("\n=== What the model learned (per arm) ===")
    for key, s in model.items():
        print(f"{key:>18}: {json.dumps(s)}")
    print(f"\nFiles in {out}: {', '.join(sorted(p.name for p in out.iterdir()))}")
    return 0


def cmd_status(cfg: BotConfig, args) -> int:
    from .brain import Brain

    state = Path(cfg.paths.state_dir)
    files = sorted(state.glob("brain_*.joblib"))
    if not files:
        print(f"No saved models in {state}/")
    for f in files:
        brain = Brain.load(f)
        print(f"\n{f.name}: {brain.n_learned} samples learned")
        for key, s in brain.summary().items():
            print(f"  {key:>18}: {json.dumps(s)}")
    risk = state / "bot_state.json"
    if risk.exists():
        print(f"\nRisk state: {json.dumps(json.loads(risk.read_text())['risk'], indent=2)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradebot", description="Self-learning MT5 trading bot")
    p.add_argument("-c", "--config", default="config.yaml", help="YAML config file (default: config.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="test the MT5 connection")
    live = sub.add_parser("live", help="run the bot on MT5")
    live.add_argument("--dry-run", action="store_true", help="log decisions only, never send orders")
    live.add_argument("--max-cycles", type=int, default=None, help=argparse.SUPPRESS)

    bt = sub.add_parser("backtest", help="walk-forward backtest")
    src = bt.add_mutually_exclusive_group()
    src.add_argument("--csv", help="bars CSV (from `export` or the MT5 terminal's Export Bars)")
    src.add_argument("--synthetic", type=int, default=30000, help="number of synthetic bars (default)")
    bt.add_argument("--symbol", nargs="*", help="symbol name (default: from the CSV file name)")
    bt.add_argument("--bars", type=int, help="only use the last N bars")
    bt.add_argument("--warmup", type=int, help="bars to learn from before trading (default: config)")
    bt.add_argument("--seed", type=int, default=1)
    bt.add_argument("--out", default="backtests")

    ex = sub.add_parser("export", help="download history from MT5 to CSV")
    ex.add_argument("--symbol", nargs="*")
    ex.add_argument("--bars", type=int, default=100_000)
    ex.add_argument("--out", default="data")

    sub.add_parser("status", help="show what the saved models learned")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config if Path(args.config).exists() else None
    if config_path is None and args.config != "config.yaml":
        print(f"Config file {args.config} not found", file=sys.stderr)
        return 2
    cfg = load_config(config_path)
    setup_logging(cfg.paths.log_dir, args.verbose, to_file=args.command in ("live", "check"))
    if config_path is None:
        log.info("No config.yaml found - using defaults (copy config.example.yaml to config.yaml)")
    handlers = {"check": cmd_check, "live": cmd_live, "backtest": cmd_backtest,
                "export": cmd_export, "status": cmd_status}
    try:
        return handlers[args.command](cfg, args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        from .brokers.base import BrokerError

        if isinstance(exc, BrokerError):
            log.error("%s", exc)
            return 1
        raise
