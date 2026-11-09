"""CSV journals so every decision and trade can be audited later (open them in Excel)."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional


class Journal:
    FILES = {
        "decisions": ["time", "symbol", "action", "arm", "prob", "ev_r", "reason", "best_buy_p", "best_sell_p"],
        "orders": ["time", "account", "symbol", "side", "volume", "price", "sl", "tp", "risk_money", "prob",
                   "ev_r", "arm", "outlook", "ticket", "status", "message"],
        "closed": ["time", "account", "symbol", "ticket", "side", "volume", "price_close", "profit",
                   "r_multiple", "arm", "reason"],
        "events": ["time", "symbol", "event", "detail"],
    }

    def __init__(self, directory: Optional[str], log_decisions: bool = True, auto_flush: bool = True):
        self.dir = Path(directory) if directory else None
        self.log_decisions = log_decisions
        self.auto_flush = auto_flush
        self._handles: Dict[str, tuple] = {}
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _writer(self, name: str, row: dict):
        if name not in self._handles:
            fields: List[str] = list(self.FILES[name])
            fields += [k for k in row if k not in fields]  # e.g. one up_<TF> column per outlook
            path = self.dir / f"{name}.csv"
            if path.exists() and path.stat().st_size > 0:
                with open(path, newline="", encoding="utf-8") as fh:
                    header = next(csv.reader(fh), [])
                if header != fields:  # layout changed (new version / settings): archive the old file
                    path.rename(path.with_name(f"{name}.{time.strftime('%Y%m%d-%H%M%S')}.csv"))
            new = not path.exists() or path.stat().st_size == 0
            fh = open(path, "a", newline="", encoding="utf-8")
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            self._handles[name] = (fh, w)
        return self._handles[name][1]

    def write(self, name: str, row: dict) -> None:
        if self.dir is None or (name == "decisions" and not self.log_decisions):
            return
        self._writer(name, row).writerow(row)
        if self.auto_flush:
            self._handles[name][0].flush()

    def close(self) -> None:
        for fh, _ in self._handles.values():
            fh.close()
        self._handles.clear()
