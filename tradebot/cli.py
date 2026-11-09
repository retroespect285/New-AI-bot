"""Command line: ``python -m tradebot <command>``.

Commands
  setup      guided set-up: connects to MT5, finds gold, asks two questions, writes config.yaml
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


def _symbol_report(cfg: BotConfig, broker, symbol: str, acct) -> str:
    """Price, history available on every chart, and whether this account can afford the trades."""
    from .fit import assess_fit, fit_report, latest_atr
    from .types import Side

    info = broker.symbol_info(symbol)
    tick = broker.get_tick(symbol)
    print(f"{symbol}: bid {tick.bid} ask {tick.ask} spread {tick.spread / info.point:.0f} points | "
          f"spec {info.to_dict()}")
    print(f"  value: {info.volume_min:g} lot gains/loses {info.value_per_price_unit * info.volume_min:,.2f} "
          f"{acct.currency} per 1.00 move in price (checked with MT5's profit calculator)")
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
    report = fit_report(rows, acct.equity, acct.currency, cfg.risk.risk_per_trade_pct, margin, acct.margin_free)
    print(report)
    return report


def cmd_check(cfg: BotConfig, args) -> int:
    broker = _mt5_broker(cfg)
    broker.connect()
    try:
        acct = broker.account()
        print(f"Connected to MetaTrader 5. Account {acct.login}: balance {acct.balance:.2f} {acct.currency}, "
              f"equity {acct.equity:.2f}, {'DEMO' if acct.is_demo else 'REAL MONEY'}")
        if not acct.is_demo and not cfg.mt5.allow_live_account:
            print("  Note: this is a real-money account and mt5.allow_live_account is false - the bot will "
                  "refuse to trade it (run setup again to change that).")
        for symbol in cfg.trading.symbols:
            _symbol_report(cfg, broker, symbol, acct)
        positions = broker.positions(magic=cfg.trading.magic)
        print(f"Open positions of this bot (magic {cfg.trading.magic}): {len(positions)}")
    finally:
        broker.shutdown()
    return 0


def _ask(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{question}{suffix}: ").strip()
    except EOFError:
        answer = ""
    return answer or default


def cmd_setup(cfg: BotConfig, args) -> int:
    """Guided set-up for people who don't want to edit files."""
    import shutil

    import yaml

    from .brokers.base import BrokerError

    path = Path(args.config)
    print("\n=== New-AI-bot set-up ===")
    print("Step 1 of 4: connecting to MetaTrader 5 on this computer ...")
    broker = _mt5_broker(cfg)
    try:
        broker.connect()
    except BrokerError as exc:
        print(f"\n{exc}\n")
        if "package is not installed" in str(exc):
            return 1
        answer = _ask("If MT5 is open and logged in but installed in an unusual folder, paste the full path of "
                      "terminal64.exe here (or just press Enter to stop)")
        if not answer:
            print("\nOpen MetaTrader 5, log in, then run set-up again.")
            return 1
        cfg.mt5.path = answer.strip().strip('"')
        broker = _mt5_broker(cfg)
        try:
            broker.connect()
        except BrokerError as exc2:
            print(f"\nStill no connection: {exc2}")
            return 1
    try:
        acct = broker.account()
        kind = "DEMO" if acct.is_demo else "REAL MONEY"
        print(f"  Connected: account {acct.login}, balance {acct.balance:,.2f} {acct.currency} ({kind})")
        cent = acct.currency.upper() in ("USC", "EUC", "GBC") or "cent" in acct.currency.lower()

        print("\nStep 2 of 4: which gold symbol should the bot trade?")
        symbols = broker.find_symbols(["XAU", "GOLD"])
        symbol = None
        if symbols:
            for k, (name, desc) in enumerate(symbols, 1):
                print(f"  {k}) {name:<14} {desc}")
            default = 1
            if cent:  # prefer a cent-looking symbol on a cent account
                for k, (name, _) in enumerate(symbols, 1):
                    if name.lower().endswith("c"):
                        default = k
                        break
            while symbol is None:
                choice = _ask("Type the number of the symbol", str(default))
                if choice.isdigit() and 1 <= int(choice) <= len(symbols):
                    symbol = symbols[int(choice) - 1][0]
                elif any(choice == n for n, _ in symbols):
                    symbol = choice
                else:
                    print("  Please type one of the numbers above.")
        else:
            print("  No gold symbol found. In MT5: View > Market Watch, right-click > Symbols, search 'XAU' or "
                  "'GOLD', select it and click Show Symbol.")
            symbol = _ask("Or type the exact symbol name here")
            if not symbol:
                return 1
        broker.symbol_info(symbol)  # makes sure MT5 shows it in Market Watch
        print(f"  Using {symbol}")

        print("\nStep 3 of 4: how much of the account may one losing trade cost?")
        default_risk = "1.0" if cent or acct.equity < 500 else "0.5"
        print(f"  (in percent; {default_risk} is suggested{' for a cent / small account' if default_risk == '1.0' else ''}"
              f"; the maximum allowed is 5)")
        risk = None
        while risk is None:
            text = _ask("Risk per trade in %", default_risk).replace(",", ".").rstrip("%")
            try:
                value = float(text)
            except ValueError:
                value = -1
            if 0 < value <= 5:
                risk = value
            else:
                print("  Please type a number between 0.1 and 5.")

        allow_live = cfg.mt5.allow_live_account
        print("\nStep 4 of 4: real money")
        if acct.is_demo:
            print("  This is a demo account - nothing to confirm.")
        else:
            print("  This is a REAL-MONEY account. The bot can lose money; only continue if you accept that.")
            answer = _ask("Allow the bot to trade this real account? Type yes or no", "no").lower()
            allow_live = answer in ("y", "yes")
            print("  Allowed." if allow_live else "  Not allowed - the bot will only watch (dry run) on this account.")

        data = {}
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        data.setdefault("mt5", {})
        data["mt5"]["allow_live_account"] = bool(allow_live)
        if cfg.mt5.path:
            data["mt5"]["path"] = cfg.mt5.path
        data.setdefault("trading", {})["symbols"] = [symbol]
        data.setdefault("risk", {})["risk_per_trade_pct"] = risk
        if cent or acct.equity < 500:
            data["risk"].setdefault("max_total_positions", 1)
        path.write_text("# Written by `python -m tradebot setup`. See config.example.yaml for every option.\n"
                        + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        new_cfg = load_config(str(path))
        print(f"\nSaved your settings to {path}.")

        print("\nChecking what this account can trade ...")
        report = _symbol_report(new_cfg, broker, symbol, acct)
        if "WILL NOT" in report:
            print("\nSettings saved, but this account is too small for the bot to open trades (see 'Account fit').")
            print("  - 3-TEST-WITHOUT-TRADING.bat still works: the bot learns and shows what it would do.")
            print("  - To trade for real: use a CENT account (then run this set-up again) or add funds.")
        else:
            print("\nAll set. Next:")
            print("  - 3-TEST-WITHOUT-TRADING.bat : watch the bot think for a few candles, no orders")
            print("  - 4-START-TRADING.bat        : start trading" +
                  (" (allowed on this real account)" if not acct.is_demo and allow_live else ""))
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

    sub.add_parser("setup", help="guided set-up: connect to MT5, pick gold, write config.yaml")
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
    if config_path is None and args.config != "config.yaml" and args.command != "setup":
        print(f"Config file {args.config} not found", file=sys.stderr)
        return 2
    cfg = load_config(config_path)
    setup_logging(cfg.paths.log_dir, args.verbose, to_file=args.command in ("live", "check"))
    if config_path is None and args.command != "setup":
        log.info("No config.yaml found - using defaults (copy config.example.yaml to config.yaml)")
    handlers = {"setup": cmd_setup, "check": cmd_check, "live": cmd_live, "backtest": cmd_backtest,
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
