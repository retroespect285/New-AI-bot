"""High-impact news blackout. Gold reacts violently to US data (NFP, CPI, FOMC...), and candle
patterns mean little in those minutes, so the bot can be told not to open trades around them.

The file is a CSV in **broker server time**, for example::

    time,title,minutes_before,minutes_after
    2026-11-06 15:30,US Non-Farm Payrolls,30,30
    2026-11-12 15:30,US CPI,,
    2026-12-10 21:00,FOMC rate decision,30,60

``minutes_before`` / ``minutes_after`` are optional (defaults come from the config). Copy the
times from any economic calendar, converted to your broker's server time zone.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)


class NewsCalendar:
    def __init__(self, windows: Optional[List[Tuple[pd.Timestamp, pd.Timestamp, str]]] = None):
        self.windows = sorted(windows or [], key=lambda w: w[0])

    @classmethod
    def load(cls, path: Optional[str], before_min: int, after_min: int) -> "NewsCalendar":
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            log.warning("News file %s not found - no news blackout", p)
            return cls()
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        if "time" not in df.columns:
            raise ValueError(f"{p}: needs a 'time' column (broker server time)")
        windows = []
        for _, row in df.iterrows():
            t = pd.Timestamp(row["time"])
            before = row.get("minutes_before")
            after = row.get("minutes_after")
            before = before_min if pd.isna(before) else int(before)
            after = after_min if pd.isna(after) else int(after)
            title = str(row.get("title", "news")) if not pd.isna(row.get("title", "news")) else "news"
            windows.append((t - pd.Timedelta(minutes=before), t + pd.Timedelta(minutes=after), title))
        log.info("Loaded %d news events from %s", len(windows), p)
        return cls(windows)

    def blocked(self, now: pd.Timestamp) -> Optional[str]:
        """Title of the news event whose blackout window contains ``now``, else None."""
        for start, end, title in self.windows:
            if start > now:
                break
            if now <= end:
                return title
        return None
