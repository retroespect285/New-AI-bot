import pandas as pd

from tradebot.news import NewsCalendar


def test_news_blackout_windows(tmp_path):
    p = tmp_path / "news.csv"
    p.write_text("time,title,minutes_before,minutes_after\n"
                 "2024-03-08 15:30,US Non-Farm Payrolls,30,30\n"
                 "2024-03-12 15:30,US CPI,,\n")
    cal = NewsCalendar.load(str(p), before_min=10, after_min=20)
    assert cal.blocked(pd.Timestamp("2024-03-08 15:05")) == "US Non-Farm Payrolls"
    assert cal.blocked(pd.Timestamp("2024-03-08 16:00")) == "US Non-Farm Payrolls"
    assert cal.blocked(pd.Timestamp("2024-03-08 16:01")) is None
    assert cal.blocked(pd.Timestamp("2024-03-12 15:19")) is None  # default 10 min before
    assert cal.blocked(pd.Timestamp("2024-03-12 15:45")) == "US CPI"  # default 20 min after
    assert NewsCalendar.load(None, 10, 10).blocked(pd.Timestamp("2024-03-08 15:30")) is None
