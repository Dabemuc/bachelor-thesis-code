"""Edge-Runner: HEF auf dem Hailo-8 über ein festes Subset laufen lassen.

Schreibt NUR Rohdaten – keine Metriken, keine Plots (die kosten CPU und
verfälschen die Latenz):

    results/<run>/meta.json         Git-Commit, Versionen, Config
    results/<run>/predictions.csv   image_id, label, top1, top5, latency_ms, repeat
    results/<run>/telemetry.csv     SoC-/Hailo-Temperatur, Drossel-Flags (1 Hz-ish)
    results/<run>/logits.npz        ids [N], logits [N, 1000]   (Reihenfolge = Subset)
    results/<run>/features.npz      ids [N], features [N, H, W, K] (falls 2. Ausgang im HEF)

Aufruf auf dem Pi:
    thesis-edge-run configs/resnet18_int8.yaml

Latenz hier = Host-seitige Wandzeit eines ``infer``-Aufrufs mit Batch 1
(inkl. PCIe-Transfer und Dequantisierung). Für reine Hardware-Latenz/FPS
zusätzlich ``hailortcli run <hef>`` bzw. ``hailortcli benchmark`` verwenden
und beide Zahlen getrennt berichten.

⚠️ API-Grundlage: HailoRT-Python-API (``hailo_platform``) mit InferVStreams,
wie in den HailoRT-Tutorials. Beim ersten Lauf gegen die installierte Version
prüfen (``python -c "import hailo_platform; print(hailo_platform.__file__)"``).
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ..common.config import load_config
from ..common.preprocess import load_uint8, normalize
from ..common.runmeta import new_run_dir, write_meta
from ..common.subset import read_subset
from . import telemetry


def _import_hailo():
    try:
        import hailo_platform as hp  # noqa: PLC0415
    except ImportError:
        sys.exit(
            "hailo_platform nicht gefunden. Auf dem Pi: `sudo apt install hailo-all` und das venv "
            "mit --system-site-packages anlegen (siehe README)."
        )
    return hp


class TelemetryLogger(threading.Thread):
    def __init__(self, path: Path, vdevice, interval_s: float = 1.0):
        super().__init__(daemon=True)
        self.path, self.vdevice, self.interval_s = path, vdevice, interval_s
        self._halt = threading.Event()

    def run(self) -> None:
        with open(self.path, "w", newline="") as f:
            w = None
            while not self._halt.is_set():
                row = telemetry.sample(self.vdevice).as_row()
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    w.writeheader()
                w.writerow(row)
                f.flush()
                self._halt.wait(self.interval_s)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)


def _classify_outputs(out: dict[str, np.ndarray], num_classes: int = 1000) -> tuple[np.ndarray, np.ndarray | None]:
    """Logits- und Feature-Ausgang anhand der Form trennen."""
    logits, feats = None, None
    for arr in out.values():
        a = np.asarray(arr)[0]
        if a.size == num_classes:
            logits = a.reshape(-1)
        elif a.ndim == 3:
            feats = a
    if logits is None:
        raise RuntimeError(f"Kein Logits-Ausgang ({num_classes} Werte) gefunden: { {k: v.shape for k, v in out.items()} }")
    return logits, feats


def main() -> None:
    ap = argparse.ArgumentParser(description="HEF auf dem Hailo-8 über ein Subset laufen lassen")
    ap.add_argument("config", help="configs/<experiment>.yaml")
    ap.add_argument("--limit", type=int, default=None, help="nur die ersten N Bilder (Smoke-Test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.hef_path is None or not cfg.hef_path.exists():
        sys.exit(f"HEF fehlt: {cfg.hef_path}")
    items = read_subset(cfg.subset_path)[: args.limit]

    hp = _import_hailo()
    run_dir = new_run_dir(cfg)
    write_meta(run_dir, cfg, extra={"n_images": len(items), "role": "edge"})

    hef = hp.HEF(str(cfg.hef_path))
    with hp.VDevice() as target:
        configure_params = hp.ConfigureParams.create_from_hef(hef, interface=hp.HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        ng_params = network_group.create_params()

        in_fmt = hp.FormatType.UINT8 if cfg.normalize_on_chip else hp.FormatType.FLOAT32
        in_params = hp.InputVStreamParams.make(network_group, format_type=in_fmt)
        # FLOAT32 → HailoRT dequantisiert die Ausgänge (Logits + Feature-Maps)
        out_params = hp.OutputVStreamParams.make(network_group, format_type=hp.FormatType.FLOAT32)
        input_name = hef.get_input_vstream_infos()[0].name

        tlog = TelemetryLogger(run_dir / "telemetry.csv", target)
        tlog.start()

        ids: list[str] = []
        logits_all: list[np.ndarray] = []
        feats_all: list[np.ndarray] = []
        with hp.InferVStreams(network_group, in_params, out_params) as pipe, network_group.activate(ng_params):
            # Warm-up (nicht gemessen)
            dummy = np.zeros((1, *load_uint8(items[0].path).shape), dtype=np.uint8 if cfg.normalize_on_chip else np.float32)
            for _ in range(cfg.warmup_iters):
                pipe.infer({input_name: dummy})

            with open(run_dir / "predictions.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "label", "top1", "top5", "latency_ms", "repeat"])
                for rep in range(cfg.repeats):
                    for it in items:
                        img = load_uint8(it.path)
                        x = img[None] if cfg.normalize_on_chip else normalize(img)[None]
                        t0 = time.perf_counter_ns()
                        out = pipe.infer({input_name: x})
                        dt_ms = (time.perf_counter_ns() - t0) / 1e6
                        logits, feats = _classify_outputs(out)
                        top5 = np.argsort(logits)[::-1][:5]
                        w.writerow([it.image_id, it.label, int(top5[0]), " ".join(map(str, top5)), f"{dt_ms:.3f}", rep])
                        if rep == 0:
                            ids.append(it.image_id)
                            logits_all.append(logits.astype(np.float32))
                            if feats is not None and cfg.save_features:
                                feats_all.append(feats.astype(np.float32))
        tlog.stop()

    np.savez_compressed(run_dir / "logits.npz", ids=np.array(ids), logits=np.stack(logits_all))
    if feats_all:
        np.savez_compressed(run_dir / "features.npz", ids=np.array(ids), features=np.stack(feats_all))
    print(f"fertig → {run_dir}")


if __name__ == "__main__":
    main()
