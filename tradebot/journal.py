"""CSV journals so every decision and trade can be audited later (open them in Excel)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional


class Journal:
    FILES = {
        "decisions": ["time", "symbol", "action", "arm", "prob", "ev_r", "reason", "best_buy_p", "best_sell_p"],
        "orders": ["time", "symbol", "side", "volume", "price", "sl", "tp", "risk_money", "prob", "ev_r",
                   "arm", "ticket", "status", "message"],
        "closed": ["time", "symbol", "ticket", "side", "volume", "price_close", "profit", "r_multiple", "reason"],
        "events": ["time", "symbol", "event", "detail"],
    }

    def __init__(self, directory: Optional[str], log_decisions: bool = True, auto_flush: bool = True):
        self.dir = Path(directory) if directory else None
        self.log_decisions = log_decisions
        self.auto_flush = auto_flush
        self._handles: Dict[str, tuple] = {}
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _writer(self, name: str):
        if name not in self._handles:
            path = self.dir / f"{name}.csv"
            new = not path.exists() or path.stat().st_size == 0
            fh = open(path, "a", newline="", encoding="utf-8")
            w = csv.DictWriter(fh, fieldnames=self.FILES[name], extrasaction="ignore")
            if new:
                w.writeheader()
            self._handles[name] = (fh, w)
        return self._handles[name][1]

    def write(self, name: str, row: dict) -> None:
        if self.dir is None or (name == "decisions" and not self.log_decisions):
            return
        self._writer(name).writerow(row)
        if self.auto_flush:
            self._handles[name][0].flush()

    def close(self) -> None:
        for fh, _ in self._handles.values():
            fh.close()
        self._handles.clear()
