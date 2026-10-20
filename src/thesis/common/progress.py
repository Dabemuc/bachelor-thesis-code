"""Schlanke Fortschrittsanzeige für lange Läufe (ohne Zusatzabhängigkeit).

Gibt höchstens alle ``every_s`` Sekunden eine Zeile auf stderr aus:
Anzahl, Prozent, Bilder/s, Restzeit. Die Ausgabe passiert außerhalb der
Latenzmessung des Edge-Runners und kostet daher keine gemessene Zeit.
"""

from __future__ import annotations

import sys
import time


class Progress:
    def __init__(self, total: int, label: str = "", every_s: float = 10.0) -> None:
        self.total, self.label, self.every_s = total, label, every_s
        self.n = 0
        self.t0 = self._last = time.monotonic()

    def step(self, k: int = 1) -> None:
        self.n += k
        now = time.monotonic()
        if now - self._last >= self.every_s or self.n >= self.total:
            self._last = now
            self._print(now)

    def _print(self, now: float) -> None:
        el = now - self.t0
        rate = self.n / el if el > 0 else 0.0
        rest = (self.total - self.n) / rate if rate > 0 else float("nan")
        eta = "–" if rest != rest else f"{int(rest // 3600)}h {int(rest % 3600 // 60):02d}m {int(rest % 60):02d}s"
        pct = 100.0 * self.n / self.total if self.total else 100.0
        print(f"[{time.strftime('%H:%M:%S')}] {self.label} {self.n}/{self.total} ({pct:.1f} %) "
              f"{rate:.1f} Bilder/s, Rest ≈ {eta}", file=sys.stderr, flush=True)
