"""Run-Ordner + Metadaten.

Jeder Messlauf bekommt einen eigenen Ordner ``results/<zeit>_<host>_<experiment>/``
mit ``meta.json``. Darin steht alles, was man später im Kolloquium gefragt wird:
Git-Commit (und ob uncommitted Änderungen dabei waren), Host, Versionen, Config.
Rohdaten in diesem Ordner werden nachträglich nie verändert.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from .config import REPO_ROOT, RESULTS_DIR, ExperimentConfig


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> dict:
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def package_versions(names: tuple[str, ...] = ("numpy", "torch", "onnx", "hailort", "hailo-dataflow-compiler")) -> dict:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    # Auf dem Pi kommt HailoRT per apt – die Paketversion steht dann nur in dpkg.
    dpkg = _run(["dpkg-query", "-W", "-f=${Version}", "hailort"])
    if dpkg:
        out["hailort (dpkg)"] = dpkg
    return out


def new_run_dir(cfg: ExperimentConfig, tag: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname().split(".")[0]
    name = f"{stamp}_{host}_{cfg.name}" + (f"_{tag}" if tag else "")
    run_dir = RESULTS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_meta(run_dir: Path, cfg: ExperimentConfig, extra: dict | None = None) -> None:
    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "git": git_state(),
        "packages": package_versions(),
        "config": dataclasses.asdict(cfg),
        **(extra or {}),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if meta["git"]["dirty"]:
        print("⚠️  Uncommittete Änderungen – der Lauf ist keinem Commit eindeutig zuzuordnen.", file=sys.stderr)
