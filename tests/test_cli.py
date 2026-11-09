from tests.conftest import EURUSD, small_config
from tradebot.brokers.sim import SimBroker
from tradebot.cli import main
from tradebot.core import BotCore
from tradebot.data import synthetic_bars


def test_status_shows_learning_progress(tmp_path, capsys):
    cfg = small_config()
    state = tmp_path / "state"
    bars = synthetic_bars(2500, seed=41)
    core = BotCore(cfg, SimBroker({"EURUSD": bars}, {"EURUSD": EURUSD}), state_dir=str(state))
    core.add_symbol("EURUSD")
    core.learn_range("EURUSD", core.prepare("EURUSD", bars), len(bars) - 1, progress=False)
    core.save_state()
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(f"paths:\n  state_dir: {state.as_posix()}\n  log_dir: {(tmp_path / 'logs').as_posix()}\n"
                        "model:\n  outlooks: [M5, M30]\n")
    assert main(["-c", str(cfg_file), "status", "--days", "3"]) == 0
    out = capsys.readouterr().out
    assert "brain_EURUSD_M5.joblib" in out and "Is it getting smarter?" in out
    assert out.count("2024-01-") >= 3  # rows of the daily history table
    assert "Direction forecasts" in out


def test_readiness_exit_code(tmp_path, capsys):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(f"paths:\n  log_dir: {(tmp_path / 'logs').as_posix()}\n")
    assert main(["-c", str(cfg_file), "readiness"]) == 1
    assert "NOT READY" in capsys.readouterr().out
