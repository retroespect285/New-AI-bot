

from tests import fake_mt5
from tests.conftest import EURUSD, small_config
from tradebot.brain import Decision
from tradebot.brokers.mt5 import MT5Broker
from tradebot.brokers.sim import SimBroker
from tradebot.config import TradeStyle
from tradebot.core import BotCore
from tradebot.data import synthetic_bars
from tradebot.fit import assess_fit, fit_report
from tradebot.labels import make_arms
from tradebot.types import SymbolInfo

GOLD = SymbolInfo(name="XAUUSD", point=0.01, digits=2, tick_size=0.01, tick_value=1.0, contract_size=100)
# cent account: balance in US cents, 1 lot = 1 oz, so 0.01 lot moves 1 cent per $1
GOLD_CENT = SymbolInfo(name="XAUUSDc", point=0.01, digits=2, tick_size=0.01, tick_value=1.0, contract_size=1)
STYLES = [TradeStyle("M5", 1.5, 2.0, 12), TradeStyle("M30", 1.0, 1.5, 48), TradeStyle("H1", 1.0, 2.0, 96)]
ATR = {"M5": 3.0, "M30": 7.0, "H1": 10.0}


def test_twenty_dollar_standard_gold_account_cannot_trade():
    rows = assess_fit(GOLD, STYLES, ATR, equity=20.0, risk_pct=0.5)
    assert [r.fits for r in rows] == [False, False, False]
    assert rows[0].min_risk == 4.5  # 0.01 lot = 1 oz, stop 1.5 x $3
    report = fit_report(rows, 20.0, "USD", 0.5, min_lot_margin=40.0, free_margin=20.0)
    assert "WILL NOT OPEN TRADES" in report and "CENT account" in report and "margin" in report


def test_same_money_on_a_cent_account_fits():
    # $20 = 2,000 US cents; 0.01 lot with a $4.50 stop risks 4.5 cents
    rows = assess_fit(GOLD_CENT, STYLES, ATR, equity=2000.0, risk_pct=1.0)
    assert all(r.fits for r in rows)
    assert rows[2].min_risk == 10.0 and rows[2].min_risk_pct == 0.5
    assert "every trade style fits" in fit_report(rows, 2000.0, "USC", 1.0)


class TightMarginSim(SimBroker):
    def margin_required(self, symbol, side, volume, price):
        return 1e9


def test_order_is_skipped_when_margin_is_not_enough(tmp_path, monkeypatch):
    cfg = small_config()
    cfg.risk.max_spread_sl_fraction = 1.0
    bars = synthetic_bars(900, seed=11)
    sim = TightMarginSim({"EURUSD": bars}, {"EURUSD": EURUSD})
    core = BotCore(cfg, sim)
    core.add_symbol("EURUSD")
    prep = core.prepare("EURUSD", bars)
    brain = core.brain("EURUSD", prep.features.names)
    arm = make_arms(cfg.model.styles)[0]
    monkeypatch.setattr(brain, "decide", lambda *a, **k: Decision(arm, prob=0.6, ev_r=0.3, reason="signal"))
    sim.set_index("EURUSD", 700)
    d = core.on_bar("EURUSD", prep, 700)
    assert d.reason.startswith("blocked: not enough free margin") and not sim.positions()


def test_small_account_skip_reason_shows_the_numbers(tmp_path, monkeypatch):
    cfg = small_config()
    cfg.risk.max_spread_sl_fraction = 1.0
    bars = synthetic_bars(900, seed=11)
    sim = SimBroker({"EURUSD": bars}, {"EURUSD": EURUSD}, initial_balance=20.0)
    core = BotCore(cfg, sim)
    core.add_symbol("EURUSD")
    prep = core.prepare("EURUSD", bars)
    brain = core.brain("EURUSD", prep.features.names)
    monkeypatch.setattr(brain, "decide", lambda *a, **k: Decision(make_arms(cfg.model.styles)[0], prob=0.6,
                                                                  ev_r=0.3, reason="signal"))
    sim.set_index("EURUSD", 700)
    d = core.on_bar("EURUSD", prep, 700)
    assert "minimum lot size would exceed the risk budget" in d.reason and "% of equity" in d.reason
    assert not sim.positions()


def test_check_command_reports_account_fit(tmp_path, monkeypatch, capsys):
    from tradebot import cli

    fake_mt5.state.reset()
    fake_mt5.state.equity = 20.0
    monkeypatch.setattr(cli, "_mt5_broker", lambda cfg: MT5Broker(cfg.mt5, module=fake_mt5))
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("trading:\n  symbols: [EURUSD]\n  warmup_bars: 2000\n"
                        f"paths:\n  log_dir: {(tmp_path / 'logs').as_posix()}\n")
    assert cli.main(["-c", str(cfg_file), "check"]) == 0
    out = capsys.readouterr().out
    assert "Account fit: equity 20.00 USD" in out
    assert "WILL NOT OPEN TRADES" in out


# ---------------------------------------------------------------- small-account mode
from tradebot.config import RiskConfig  # noqa: E402
from tradebot.risk import RiskManager  # noqa: E402


def test_small_account_mode_allows_the_minimum_lot_up_to_the_cap():
    gold_1usd = GOLD  # 0.01 lot = 1 oz: $1 per $1 move
    assert RiskManager(RiskConfig(risk_per_trade_pct=1.0)).position_size(20.0, gold_1usd, 2.0) == (0.0, 0.0)
    lots, risk = RiskManager(RiskConfig(risk_per_trade_pct=1.0, min_lot_max_risk_pct=15)).position_size(
        20.0, gold_1usd, 2.0)
    assert lots == 0.01 and risk == 2.0  # 10% of $20, under the 15% cap
    assert RiskManager(RiskConfig(min_lot_max_risk_pct=5)).position_size(20.0, gold_1usd, 2.0) == (0.0, 0.0)
    # a bigger account goes back to normal % sizing automatically
    lots, risk = RiskManager(RiskConfig(risk_per_trade_pct=1.0, min_lot_max_risk_pct=15)).position_size(
        1000.0, gold_1usd, 2.0)
    assert lots == 0.05 and risk == 10.0


def test_fit_report_in_small_account_mode():
    styles = STYLES + [TradeStyle("M1", 1.0, 1.5, 6)]
    rows = assess_fit(GOLD, styles, {**ATR, "M1": 2.0}, equity=20.0, risk_pct=1.0, min_lot_cap_pct=15)
    assert [r.fits for r in rows] == [False, False, False, True] and rows[3].small_mode
    report = fit_report(rows, 20.0, "USD", 1.0, 8.25, 20.0)
    assert "OK (small-account mode)" in report and "SMALL-ACCOUNT MODE" in report and "WILL NOT" not in report


def test_twenty_dollars_trades_the_minimum_lot_in_small_account_mode(monkeypatch):
    cfg = small_config()
    cfg.risk.max_spread_sl_fraction = 1.0
    cfg.risk.min_lot_max_risk_pct = 50.0
    bars = synthetic_bars(900, seed=11)
    sim = SimBroker({"EURUSD": bars}, {"EURUSD": EURUSD}, initial_balance=20.0)
    core = BotCore(cfg, sim)
    core.add_symbol("EURUSD")
    prep = core.prepare("EURUSD", bars)
    brain = core.brain("EURUSD", prep.features.names)
    monkeypatch.setattr(brain, "decide", lambda *a, **k: Decision(make_arms(cfg.model.styles)[0], prob=0.6,
                                                                  ev_r=0.3, reason="signal"))
    sim.set_index("EURUSD", 700)
    core.on_bar("EURUSD", prep, 700)
    (pos,) = sim.positions()
    assert pos.volume == 0.01
    assert core.open_risk[pos.ticket]["risk_money"] <= 20.0 * 0.5


def test_takes_the_next_best_setup_that_fits_the_limit(monkeypatch):
    from tradebot.brain import ArmEval

    cfg = small_config()  # styles: M5 sl1/tp1.5 and M5 sl1.5/tp3
    cfg.risk.max_spread_sl_fraction = 1.0
    bars = synthetic_bars(900, seed=11)
    sim = SimBroker({"EURUSD": bars}, {"EURUSD": EURUSD}, initial_balance=20.0)
    core = BotCore(cfg, sim)
    core.add_symbol("EURUSD")
    prep = core.prepare("EURUSD", bars)
    i = 700
    atr = prep.features.atr_tf["M5"][i]
    small_pct = 100 * 1.0 * atr * EURUSD.value_per_price_unit * 0.01 / 20.0
    cfg.risk.min_lot_max_risk_pct = small_pct * 1.25  # the 1-ATR stop fits, the 1.5-ATR stop doesn't
    buy_small, _, buy_big, _ = make_arms(cfg.model.styles)
    evals = [ArmEval(buy_big, 0.4, 0.3, 0.30, 0.02, True, True, 30, 0.2, True),
             ArmEval(buy_small, 0.45, 0.4, 0.15, 0.02, True, True, 30, 0.2, True)]
    brain = core.brain("EURUSD", prep.features.names)
    monkeypatch.setattr(brain, "decide", lambda *a, **k: Decision(buy_big, prob=0.4, ev_r=0.30, reason="signal",
                                                                  evals=evals))
    sim.set_index("EURUSD", i)
    d = core.on_bar("EURUSD", prep, i)
    assert d.reason == "opened" and d.arm == buy_small and d.ev_r == 0.15
    (pos,) = sim.positions()
    assert pos.volume == 0.01 and core.open_risk[pos.ticket]["arm"] == buy_small.key
