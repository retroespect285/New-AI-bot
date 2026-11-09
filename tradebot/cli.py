"""Command line: ``python -m tradebot <command>``.

Commands
  check      connect to MT5 and print account + symbol details (no trading)
  live       run the bot on MT5 (use a DEMO account first; --dry-run sends no orders)
  backtest   walk-forward backtest on exported CSVs (all timeframes) or synthetic data
  export     download history of every chart the bot reads (M1, M5, M30, H1, H4) to CSV
  status     what the brain has learned, and whether it is getting smarter day by day
  readiness  does the demo record justify trading real money?
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
    "GOLD": dict(point=0.01, digits=2, tick_size=0.01, tick_value=1.0, contract_size=100),
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
    base_name = symbol.upper().split(".")[0].replace("-SIM", "").rstrip("M") if symbol else ""
    spec = dict(cfg.backtest.symbol_info.get(symbol) or DEFAULT_SYMBOL_INFO.get(symbol.upper())
                or DEFAULT_SYMBOL_INFO.get(base_name) or {})
    if not spec:
        log.warning("No contract spec for %s - assuming gold (XAUUSD, 100 oz per lot). Add it under "
                    "backtest.symbol_info or use `export` to save the broker's spec.", symbol)
        spec = dict(DEFAULT_SYMBOL_INFO["XAUUSD"])
    spec.setdefault("tick_size", spec.get("point"))
    return SymbolInfo(name=symbol, **spec)


def _mt5_broker(cfg: BotConfig):
    from .brokers.mt5 import MT5Broker

    return MT5Broker(cfg.mt5)


def cmd_check(cfg: BotConfig, args) -> int:
    from .fit import assess_fit, fit_report, latest_atr
    from .types import Side

    broker = _mt5_broker(cfg)
    broker.connect()
    try:
        acct = broker.account()
        print(f"Account {acct.login}: balance {acct.balance:.2f} {acct.currency}, equity {acct.equity:.2f}, "
              f"{'DEMO' if acct.is_demo else 'REAL MONEY'}, trading allowed: {acct.trade_allowed}")
        for symbol in cfg.trading.symbols:
            info = broker.symbol_info(symbol)
            tick = broker.get_tick(symbol)
            print(f"{symbol}: bid {tick.bid} ask {tick.ask} spread {tick.spread / info.point:.0f} points | "
                  f"spec {info.to_dict()}")
            by_tf = {}
            for tf in [cfg.trading.timeframe] + cfg.model.context_timeframes:
                want = (cfg.trading.warmup_bars if tf == cfg.trading.timeframe
                        else cfg.context_fetch_bars(tf, cfg.trading.warmup_bars))
                bars = broker.get_rates(symbol, tf, want)
                by_tf[tf] = bars
                print(f"  {tf:>4}: {len(bars):6d} of {want} wanted closed bars "
                      f"({bars['time'].iat[0]} -> {bars['time'].iat[-1]})")
            rows = assess_fit(info, cfg.model.styles, latest_atr(by_tf, cfg.model.atr_period), acct.equity,
                              cfg.risk.risk_per_trade_pct)
            margin = broker.margin_required(symbol, Side.BUY, info.volume_min, tick.ask)
            print(fit_report(rows, acct.equity, acct.currency, cfg.risk.risk_per_trade_pct, margin, acct.margin_free))
        positions = broker.positions(magic=cfg.trading.magic)
        print(f"Open positions of this bot (magic {cfg.trading.magic}): {len(positions)}")
    finally:
        broker.shutdown()
    return 0


def cmd_live(cfg: BotConfig, args) -> int:
    from .engine import LiveEngine

    engine = LiveEngine(cfg, _mt5_broker(cfg), dry_run=args.dry_run)
    engine.run(max_cycles=args.max_cycles)
    return 0


def cmd_export(cfg: BotConfig, args) -> int:
    from .data import save_bars

    from .config import TIMEFRAME_MINUTES

    broker = _mt5_broker(cfg)
    broker.connect()
    try:
        for symbol in args.symbol or cfg.trading.symbols:
            info = broker.symbol_info(symbol)
            for tf in [cfg.trading.timeframe] + cfg.model.context_timeframes:
                count = int(args.days * 1440 / TIMEFRAME_MINUTES[tf]) + cfg.model.context_bars
                bars = broker.get_rates(symbol, tf, count)
                path = Path(args.out) / f"{symbol}_{tf}.csv"
                save_bars(bars, str(path), info)
                print(f"Saved {len(bars):7d} {tf:>4} bars to {path} ({bars['time'].iat[0]} -> {bars['time'].iat[-1]})")
    finally:
        broker.shutdown()
    return 0


def cmd_backtest(cfg: BotConfig, args) -> int:
    from .backtest import run_backtest
    from .data import load_bars, load_context, synthetic_market

    base_tf = cfg.trading.timeframe
    if args.csv:
        bars = load_bars(args.csv)
        symbol = args.symbol[0] if args.symbol else Path(args.csv).stem.rsplit("_", 1)[0]
        context = load_context(args.csv, base_tf, cfg.model.context_timeframes, bars)
    else:
        symbol = args.symbol[0] if args.symbol else "XAUUSD-SIM"
        bars, context = synthetic_market(args.synthetic, base_tf, cfg.model.context_timeframes, seed=args.seed)
        log.info("Using %d synthetic gold-like bars (pipeline demo only - not evidence of real-market "
                 "performance)", len(bars))
    if args.bars:
        bars = bars.iloc[-args.bars:].reset_index(drop=True)
    info = _symbol_info(cfg, symbol, args.csv)
    out = Path(args.out) / f"{symbol}_{base_tf}"
    out.mkdir(parents=True, exist_ok=True)
    log.info("Backtesting %s on %d %s bars (%s -> %s) reading %s; output in %s", symbol, len(bars), base_tf,
             bars["time"].iat[0], bars["time"].iat[-1], ", ".join([base_tf] + list(context)), out)
    res = run_backtest(cfg, bars, symbol, info, context=context, out_dir=str(out), warmup_bars=args.warmup)
    stats = dict(res.stats)
    model = stats.pop("model")
    outlooks = stats.pop("outlooks")
    stats.pop("learning_history", None)
    print("\n=== Backtest summary ===")
    for k, v in stats.items():
        print(f"{k:>22}: {v}")
    print("\n=== Trade setups (what the brain learned) ===")
    for key, s in model.items():
        print(f"{key:>20}: {json.dumps(s)}")
    print("\n=== Direction forecasts per timeframe ===")
    for tf, s in outlooks.items():
        print(f"{tf:>20}: {json.dumps(s)}")
    print(f"\nFiles in {out}: {', '.join(sorted(p.name for p in out.iterdir()))}")
    return 0


def _fmt(v, pct=False):
    if v is None:
        return "  -  "
    return f"{v:5.1%}" if pct else f"{v:+.3f}"


def cmd_status(cfg: BotConfig, args) -> int:
    from .brain import Brain

    state = Path(cfg.paths.state_dir)
    files = sorted(state.glob("brain_*.joblib"))
    if not files:
        print(f"No saved brain in {state}/ yet - it is created the first time you run `live`.")
    for f in files:
        brain = Brain.load(f)
        print(f"\n=== {f.name}: {brain.n_learned} bars learned, {len(brain.history)} days of history ===")
        print("\nTrade setups:")
        for key, s in brain.summary().items():
            print(f"  {key:>20}: {json.dumps(s)}")
        print("\nDirection forecasts (is price higher one candle later?):")
        for tf, s in brain.outlook_summary().items():
            print(f"  {tf:>4}: {json.dumps(s)}")
        if brain.history:
            tfs = list(brain.outlook_stats)
            print("\nIs it getting smarter? (one row per day; accuracy = recent direction hit rate, "
                  "best skill = best setup's edge over guessing)")
            print("  date        " + " ".join(f"{tf:>6}" for tf in tfs) + "   best skill   paper trades")
            for h in brain.history[-args.days:]:
                acc = " ".join(_fmt(h["outlooks"].get(tf, {}).get("accuracy"), pct=True).rjust(6) for tf in tfs)
                skills = [v["skill"] for v in h["setups"].values() if v["skill"] is not None]
                papers = sum(v["paper_n"] for v in h["setups"].values())
                print(f"  {h['date']}  {acc}   {_fmt(max(skills) if skills else None):>10}   {papers:>12}")
    risk = state / "bot_state.json"
    if risk.exists():
        data = json.loads(risk.read_text())
        print(f"\nRisk state (account {data.get('account')}): {json.dumps(data['risk'], indent=2)}")
    return 0


def cmd_readiness(cfg: BotConfig, args) -> int:
    from .readiness import assess

    res = assess(cfg, account=args.account, symbol=args.symbol)
    print(res.report())
    return 0 if res.ready else 1


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
    src.add_argument("--csv", help="trading-timeframe CSV, e.g. data/XAUUSD_M5.csv; the other charts "
                                   "(XAUUSD_M1.csv, ...) are read from the same folder")
    src.add_argument("--synthetic", type=int, default=20000, help="number of synthetic bars (default)")
    bt.add_argument("--symbol", nargs="*", help="symbol name (default: from the CSV file name)")
    bt.add_argument("--bars", type=int, help="only use the last N bars")
    bt.add_argument("--warmup", type=int, help="bars to learn from before trading (default: config)")
    bt.add_argument("--seed", type=int, default=1)
    bt.add_argument("--out", default="backtests")

    ex = sub.add_parser("export", help="download history of every chart from MT5 to CSV")
    ex.add_argument("--symbol", nargs="*")
    ex.add_argument("--days", type=int, default=180, help="calendar days of history (default 180)")
    ex.add_argument("--out", default="data")

    st = sub.add_parser("status", help="show what the brain learned and its progress")
    st.add_argument("--days", type=int, default=30, help="days of learning history to show")

    rd = sub.add_parser("readiness", help="check whether the demo record justifies going live")
    rd.add_argument("--account", type=int, help="only count trades of this (demo) account number")
    rd.add_argument("--symbol")
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
                "export": cmd_export, "status": cmd_status, "readiness": cmd_readiness}
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
