import builtins

import yaml

from tests import fake_mt5
from tradebot import cli
from tradebot.brokers.mt5 import MT5Broker
from tradebot.config import load_config


def _run(tmp_path, monkeypatch, answers):
    replies = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(replies))
    monkeypatch.setattr(cli, "_mt5_broker", lambda cfg: MT5Broker(cfg.mt5, module=fake_mt5))
    monkeypatch.chdir(tmp_path)
    return cli.main(["-c", str(tmp_path / "config.yaml"), "setup"])


def test_setup_on_a_demo_account_with_defaults(tmp_path, monkeypatch, capsys):
    fake_mt5.state.reset()
    assert _run(tmp_path, monkeypatch, ["", ""]) == 0  # press Enter twice
    out = capsys.readouterr().out
    assert "1) XAUUSD" in out and "2) XAUUSDc" in out and "EURUSD" not in out.split("Step 2")[1].split("Using")[0]
    cfg = load_config(str(tmp_path / "config.yaml"))
    assert cfg.trading.symbols == ["XAUUSD"]
    assert cfg.risk.risk_per_trade_pct == 0.5
    assert cfg.mt5.allow_live_account is False
    assert "Account fit" in out and "3-TEST-WITHOUT-TRADING.bat" in out


def test_setup_on_a_small_real_cent_account(tmp_path, monkeypatch, capsys):
    fake_mt5.state.reset()
    fake_mt5.state.trade_mode = fake_mt5.ACCOUNT_TRADE_MODE_REAL
    fake_mt5.state.equity = 20.0
    (tmp_path / "config.yaml").write_text("risk:\n  max_daily_loss_pct: 2.0\n")
    assert _run(tmp_path, monkeypatch, ["2", "", "yes"]) == 0
    data = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert data["trading"]["symbols"] == ["XAUUSDc"]
    assert data["risk"]["risk_per_trade_pct"] == 1.0  # suggested for a small account
    assert data["risk"]["max_daily_loss_pct"] == 2.0  # existing settings are kept
    assert data["risk"]["max_total_positions"] == 1
    assert data["mt5"]["allow_live_account"] is True
    assert (tmp_path / "config.yaml.bak").exists()
    assert "REAL MONEY" in capsys.readouterr().out


def test_setup_explains_when_mt5_is_not_running(tmp_path, monkeypatch, capsys):
    fake_mt5.state.reset()
    fake_mt5.state.init_ok = False
    assert _run(tmp_path, monkeypatch, [""]) == 1
    out = capsys.readouterr().out
    assert "MetaTrader 5 is installed on THIS computer and is OPEN" in out
    assert not (tmp_path / "config.yaml").exists()


def test_order_rejection_is_explained(monkeypatch):
    fake_mt5.state.reset()
    b = MT5Broker(load_config(None).mt5, module=fake_mt5)
    b.connect()
    monkeypatch.setattr(fake_mt5, "order_send", lambda req: fake_mt5.Result(10027, 0, 0, 0.0, 0.0, "AutoTrading disabled"))
    from tradebot.types import Side

    res = b.open_position("XAUUSD", Side.BUY, 0.01, sl=1.0, tp=2.0, magic=1)
    assert not res.ok and "Algo Trading" in res.message
