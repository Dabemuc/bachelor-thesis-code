"""Temperatur und Drosselung protokollieren (Messvalidität).

Zwei unabhängige Quellen, weil der HAT über dem Active Cooler sitzt und die
Hailo-Temperatur nicht aus der SoC-Temperatur ableitbar ist:

* Pi-5-SoC:  ``vcgencmd measure_temp`` / ``vcgencmd get_throttled``
* Hailo-8:   ``control.get_chip_temperature().ts0_temperature`` (HailoRT)

⚠️ API-Pfad zum physischen Device (``VDevice.get_physical_devices()``) gegen die
installierte HailoRT-Version prüfen – bei Fehlern wird ``None`` geloggt statt
abzubrechen, damit ein Telemetriefehler keinen Messlauf killt.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict, dataclass


@dataclass
class Sample:
    t: float
    soc_temp_c: float | None
    throttled_hex: str | None
    hailo_temp_c: float | None

    def as_row(self) -> dict:
        return asdict(self)


def _vcgencmd(*args: str) -> str | None:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, check=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def soc_temp() -> float | None:
    out = _vcgencmd("measure_temp")          # "temp=48.3'C"
    m = re.search(r"temp=([\d.]+)", out or "")
    return float(m.group(1)) if m else None


def throttled() -> str | None:
    out = _vcgencmd("get_throttled")         # "throttled=0x0"
    m = re.search(r"throttled=(0x[0-9a-fA-F]+)", out or "")
    return m.group(1) if m else None


def hailo_temp(vdevice) -> float | None:
    if vdevice is None:
        return None
    try:
        dev = vdevice.get_physical_devices()[0]
        return float(dev.control.get_chip_temperature().ts0_temperature)
    except Exception:  # noqa: BLE001 – Telemetrie darf nie den Lauf abbrechen
        return None


def sample(vdevice=None) -> Sample:
    return Sample(t=time.time(), soc_temp_c=soc_temp(), throttled_hex=throttled(), hailo_temp_c=hailo_temp(vdevice))
