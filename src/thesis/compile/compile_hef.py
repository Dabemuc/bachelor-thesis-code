"""ONNX → HAR → (quantisiert) → HEF mit dem Hailo Dataflow Compiler.

Aufruf im DFC-Container auf dem x86-Desktop:
    thesis-compile configs/resnet18_int8.yaml

Schritte (DFC-Python-API, ``ClientRunner``):
    1. translate_onnx_model  – Parsen; ``end_node_names`` bestimmt die HEF-Ausgänge.
                                Für Eigen-CAM auf der Hardware: Logits-Knoten UND
                                Ausgang der letzten Conv-Stufe angeben.
    2. load_model_script     – Normalisierung on-chip (optional), später Quant-Optionen
                                (z. B. 4-Bit-Gewichte für einzelne Layer, Mixed Mode).
    3. optimize(calib)       – Post-Training-Quantisierung mit Kalibrierdaten.
    4. compile()             – HEF erzeugen.

Zwischenstände (HAR) werden mitgespeichert: das quantisierte HAR ist die
Grundlage für Emulator-Läufe (SDK_QUANTIZED) und Layer-Analysen (TF4).

Logging: ``thesis-compile`` startet den eigentlichen Compile als Kindprozess und
schreibt dessen komplette Ausgabe (stdout + stderr, inkl. TensorFlow/CUDA-Meldungen
aus nativem Code) zusätzlich nach ``logs/compile_<name>.log``. Existiert die Datei
schon, bekommt der neue Lauf einen Zeitstempel im Namen. Nach Erfolg liegt eine Kopie
als ``compile.log`` neben dem HEF, dazu das ``hailo_sdk.core.log`` dieses Laufs.
``artifacts/<name>/.compile.lock`` verhindert, dass zwei Läufe gleichzeitig in
dasselbe Artefakt-Verzeichnis schreiben (passiert am 03.11.2026).

⚠️ Zu verifizieren in AP0:
  * ob ein Zwischenknoten, der zusätzlich weiterverarbeitet wird, als Ausgang
    zulässig ist (erwartet ja, aber am eigenen Modell prüfen);
  * exakte Befehlssyntax gegen den DFC-User-Guide der installierten Version.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from ..common.config import ARTIFACTS_DIR, LOGS_DIR, load_config
from ..common.preprocess import load_uint8, normalize, onchip_normalization_params
from ..common.runmeta import git_state, package_versions, sha256_file
from ..common.subset import read_subset


def build_calib_set(subset_file, n: int, normalize_on_chip: bool) -> np.ndarray:
    """Kalibrierdaten NHWC. Bei on-chip-Normalisierung: rohe 0–255-Werte (float32)."""
    items = read_subset(subset_file)[:n]
    imgs = [load_uint8(it.path) for it in items]
    if normalize_on_chip:
        return np.stack(imgs).astype(np.float32)
    return np.stack([normalize(im) for im in imgs])


# Gesetzt im Kindprozess: dort läuft der eigentliche Compile ohne erneutes Logging/Lock.
_CHILD_ENV = "THESIS_COMPILE_CHILD"
LOCK_NAME = ".compile.lock"


def _log_path(name: str) -> Path:
    """logs/compile_<name>.log; vorhandene Logs werden nie überschrieben."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"compile_{name}.log"
    if path.exists():
        path = LOGS_DIR / f"compile_{name}_{datetime.now():%Y%m%d-%H%M%S}.log"
    return path


def _acquire_lock(out_dir: Path) -> Path:
    lock = out_dir / LOCK_NAME
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        info = lock.read_text(encoding="utf-8", errors="replace").strip()
        sys.exit(
            f"Abbruch: In {out_dir} läuft bereits ein Compile ({lock.name}: {info}).\n"
            f"Falls kein Lauf mehr aktiv ist (z. B. Container hart beendet): {lock} löschen."
        )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                   "started": datetime.now().isoformat(timespec="seconds")}, f)
    return lock


def _run_logged(argv: list[str], config: str) -> int:
    """Startet den Compile als Kindprozess und schreibt die Ausgabe nach Terminal + Log."""
    cfg = load_config(config)
    out_dir = ARTIFACTS_DIR / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(out_dir)
    try:
        log_path = _log_path(cfg.name)
        cmd = [sys.executable, "-m", "thesis.compile.compile_hef", *argv]
        env = {**os.environ, _CHILD_ENV: "1", "PYTHONUNBUFFERED": "1"}
        with open(log_path, "wb") as log:
            log.write(f"# thesis-compile {' '.join(argv)}\n"
                      f"# Start: {datetime.now().isoformat(timespec='seconds')}"
                      f"  Host: {socket.gethostname()}\n".encode())
            log.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            assert proc.stdout is not None
            try:
                while chunk := os.read(proc.stdout.fileno(), 65536):
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    log.write(chunk)
                    log.flush()
                rc = proc.wait()
            except KeyboardInterrupt:
                rc = proc.wait()  # Kind hat SIGINT ebenfalls bekommen
                rc = rc if rc != 0 else 130
            log.write(f"\n# Ende: {datetime.now().isoformat(timespec='seconds')}"
                      f"  Exit-Code: {rc}\n".encode())
        if rc == 0:
            shutil.copy2(log_path, out_dir / "compile.log")
        print(f"Compile-Log → {log_path}" + ("" if rc == 0 else f"   (Exit-Code {rc})"))
        return rc
    finally:
        lock.unlink(missing_ok=True)


def _copy_core_log(out_dir: Path, config: str) -> None:
    """hailo_sdk.core.log (Allocator/Builder) dieses Laufs neben das HEF legen.

    Der DFC schreibt es ins Arbeitsverzeichnis und rotiert es bei jedem Lauf
    (core.log → core.1.log …). Kopiert wird nur, wenn die erste Zeile den eigenen
    Aufruf nennt – sonst stammt es von einem anderen, parallel laufenden Prozess.
    """
    core = Path.cwd() / "hailo_sdk.core.log"
    if not core.exists():
        return
    with core.open(encoding="utf-8", errors="replace") as f:
        first = f.readline()
    if config in first:
        shutil.copy2(core, out_dir / "hailo_sdk.core.log")
    else:
        print(f"Hinweis: {core} gehört nicht zu diesem Lauf, nicht kopiert.", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX → HEF (Hailo-8)")
    ap.add_argument("config")
    ap.add_argument("--hw-arch", default="hailo8")
    args = ap.parse_args()

    if os.environ.get(_CHILD_ENV) != "1":
        sys.exit(_run_logged(sys.argv[1:], args.config))

    try:
        from hailo_sdk_client import ClientRunner  # noqa: PLC0415
    except ImportError:
        sys.exit("hailo_sdk_client fehlt – dieses Skript läuft nur im DFC-Container (x86_64).")

    cfg = load_config(args.config)
    if not cfg.onnx_path or not cfg.onnx_path.exists():
        sys.exit(f"ONNX fehlt: {cfg.onnx_path}")
    out_dir = ARTIFACTS_DIR / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ClientRunner(hw_arch=args.hw_arch)
    runner.translate_onnx_model(
        str(cfg.onnx_path),
        cfg.model,
        end_node_names=cfg.end_node_names or None,
    )
    runner.save_har(str(out_dir / f"{cfg.model}_parsed.har"))

    script_lines = []
    if cfg.normalize_on_chip:
        mean, std = onchip_normalization_params()
        script_lines.append(f"normalization1 = normalization({mean}, {std})")
    # Hier später Quantisierungs-Optionen je Präzisionsstufe ergänzen (siehe README).
    if cfg.optimization_level is not None:
        script_lines.append(f"model_optimization_flavor(optimization_level={cfg.optimization_level})")
    script_lines.extend(cfg.model_script)
    if script_lines:
        runner.load_model_script("\n".join(script_lines) + "\n")

    if cfg.calib_subset_path is None or not cfg.calib_subset_path.exists():
        sys.exit(f"Kalibrier-Subset fehlt: {cfg.calib_subset_path} (calib_subset in der Config setzen)")
    if cfg.calib_subset == cfg.subset:
        sys.exit("calib_subset == subset – Kalibrierung auf den Evaluationsbildern ist unzulässig.")
    calib = build_calib_set(cfg.calib_subset_path, cfg.calib_size, cfg.normalize_on_chip)
    runner.optimize(calib)
    runner.save_har(str(out_dir / f"{cfg.model}_quantized.har"))

    hef = runner.compile()
    hef_path = out_dir / f"{cfg.model}.hef"
    hef_path.write_bytes(hef)
    (out_dir / "compile_meta.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "hw_arch": args.hw_arch,
        "model_script": script_lines,
        # Anzahl der an optimize() übergebenen Bilder. NICHT die Kalibriergröße des DFC:
        # die steht in calibset_size (Default 64) bzw. im Model Script oben.
        "calib_array_n": int(calib.shape[0]),
        "onnx_sha256": sha256_file(cfg.onnx_path),
        "hef_sha256": sha256_file(hef_path),
        "git": git_state(),
        "packages": package_versions(),
        "config": dataclasses.asdict(cfg),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    _copy_core_log(out_dir, args.config)
    print(f"HEF → {hef_path}   (in der Config als hef: {cfg.name}/{cfg.model}.hef eintragen)")


if __name__ == "__main__":
    main()
