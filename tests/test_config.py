from pathlib import Path

import pytest

from tradebot.config import BotConfig, ConfigError, apply_env_overrides, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads_and_matches_defaults():
    cfg = load_config(str(ROOT / "config.example.yaml"))
    default = BotConfig()
    assert cfg.model == default.model
    assert cfg.risk == default.risk
    assert cfg.trading == default.trading
    assert cfg.timeframe_minutes == 5


def test_unknown_keys_are_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("risk:\n  risk_per_trade: 1.0\n")
    with pytest.raises(ConfigError, match="risk_per_trade"):
        load_config(str(p))


def test_invalid_values_are_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("trading:\n  timeframe: M7\n")
    with pytest.raises(ConfigError, match="timeframe"):
        load_config(str(p))
    p.write_text("risk:\n  risk_per_trade_pct: 20\n")
    with pytest.raises(ConfigError):
        load_config(str(p))


def test_credentials_from_environment():
    cfg = apply_env_overrides(BotConfig(), {"MT5_LOGIN": "5551234", "MT5_PASSWORD": "secret", "MT5_SERVER": "Broker-Demo"})
    assert cfg.mt5.login == 5551234 and cfg.mt5.password == "secret" and cfg.mt5.server == "Broker-Demo"
