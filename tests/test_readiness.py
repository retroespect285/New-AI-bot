import pandas as pd

from tests.conftest import small_config
from tradebot.readiness import assess


def _write(tmp_path, rs, days=20, account=111):
    logs = tmp_path / "logs"
    logs.mkdir()
    times = pd.date_range("2024-01-01", periods=len(rs), freq=f"{max(1, days * 24 // max(1, len(rs)))}h")
    pd.DataFrame({"time": times, "account": account, "symbol": "XAUUSD", "profit": [r * 50 for r in rs],
                  "r_multiple": rs}).to_csv(logs / "closed.csv", index=False)
    pd.DataFrame({"time": pd.date_range("2024-01-01", periods=2, freq=f"{days}D"), "symbol": "XAUUSD"}
                 ).to_csv(logs / "decisions.csv", index=False)
    return str(logs)


def test_one_day_of_demo_is_not_enough(tmp_path):
    logs = _write(tmp_path, [1.5, -1, 1.5], days=1)
    res = assess(small_config(), log_dir=logs)
    assert not res.ready
    assert "NOT READY" in res.report()


def test_solid_demo_record_is_ready(tmp_path):
    rs = [1.5, -1.0, 1.5, -1.0, 2.0, -1.0] * 6  # 36 trades, profit factor ~1.7
    res = assess(small_config(), log_dir=_write(tmp_path, rs, days=20))
    assert res.ready, res.report()
    assert res.trades == 36


def test_only_counts_the_chosen_account(tmp_path):
    rs = [1.5, -1.0, 1.5, -1.0, 2.0, -1.0] * 6
    logs = _write(tmp_path, rs, days=20, account=111)
    assert assess(small_config(), log_dir=logs, account=999).trades == 0
