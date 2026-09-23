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

⚠️ Zu verifizieren in AP0:
  * ob ein Zwischenknoten, der zusätzlich weiterverarbeitet wird, als Ausgang
    zulässig ist (erwartet ja, aber am eigenen Modell prüfen);
  * exakte Befehlssyntax gegen den DFC-User-Guide der installierten Version.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ..common.config import ARTIFACTS_DIR, DATA_DIR, load_config
from ..common.preprocess import load_uint8, normalize, onchip_normalization_params
from ..common.subset import read_subset


def build_calib_set(subset_file, n: int, normalize_on_chip: bool) -> np.ndarray:
    """Kalibrierdaten NHWC. Bei on-chip-Normalisierung: rohe 0–255-Werte (float32)."""
    items = read_subset(subset_file)[:n]
    imgs = [load_uint8(it.path) for it in items]
    if normalize_on_chip:
        return np.stack(imgs).astype(np.float32)
    return np.stack([normalize(im) for im in imgs])


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX → HEF (Hailo-8)")
    ap.add_argument("config")
    ap.add_argument("--calib-subset", default="calib_1024_seed1.txt",
                    help="Subset-Datei in data/subsets/ – DISJUNKT zum Evaluations-Subset!")
    ap.add_argument("--hw-arch", default="hailo8")
    args = ap.parse_args()

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
    if script_lines:
        runner.load_model_script("\n".join(script_lines) + "\n")

    calib = build_calib_set(DATA_DIR / "subsets" / args.calib_subset, cfg.calib_size, cfg.normalize_on_chip)
    runner.optimize(calib)
    runner.save_har(str(out_dir / f"{cfg.model}_quantized.har"))

    hef = runner.compile()
    hef_path = out_dir / f"{cfg.model}.hef"
    hef_path.write_bytes(hef)
    print(f"HEF → {hef_path}   (in der Config als hef: {cfg.name}/{cfg.model}.hef eintragen)")


if __name__ == "__main__":
    main()
