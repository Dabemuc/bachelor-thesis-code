"""Run-Ordner + Metadaten.

Jeder Messlauf bekommt einen eigenen Ordner ``results/<zeit>_<host>_<experiment>/``
mit ``meta.json``. Darin steht alles, was man später im Kolloquium gefragt wird:
Git-Commit (und ob uncommitted Änderungen dabei waren), Host, Versionen, Config.
Rohdaten in diesem Ordner werden nachträglich nie verändert.
"""

from __future__ import annotations

import dataclasses
import hashlib
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


def sha256_file(path: Path | str) -> str:
    """SHA-256 einer Datei (z. B. ONNX, HEF) – macht Artefakte eindeutig zuordenbar."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hef_provenance(hef_path: Path | str) -> dict:
    """Prüfsumme des HEF + (falls vorhanden) die compile_meta.json daneben.

    Damit steht in jedem Edge-Lauf, welches HEF mit welcher DFC-Version und welchem
    Model Script lief – auch wenn der Pi selbst keinen DFC hat.
    """
    hef_path = Path(hef_path)
    info: dict = {"path": str(hef_path), "sha256": sha256_file(hef_path)}
    cmeta = hef_path.parent / "compile_meta.json"
    if cmeta.exists():
        c = json.loads(cmeta.read_text(encoding="utf-8"))
        info["compile_meta"] = {k: c.get(k) for k in (
            "created", "hw_arch", "model_script", "calib_n", "onnx_sha256", "hef_sha256", "git")}
        info["compile_meta"]["dfc_version"] = (c.get("packages") or {}).get("hailo-dataflow-compiler")
        if c.get("hef_sha256") and c["hef_sha256"] != info["sha256"]:
            print("⚠️  HEF-Prüfsumme passt nicht zur compile_meta.json daneben – veraltete Metadaten?", file=sys.stderr)
    else:
        info["compile_meta"] = None
        print(f"⚠️  Keine compile_meta.json neben {hef_path} – DFC-Version des HEF unbekannt.", file=sys.stderr)
    return info


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
