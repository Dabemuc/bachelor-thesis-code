"""FP32-Referenzlauf (PyTorch) mit demselben Datenformat wie der Edge-Runner.

    python -m thesis.reference.fp32_run configs/resnet18_fp32.yaml [--device cuda]

Schreibt ``logits.npz`` und ``features.npz`` (Feature-Maps als [H, W, K], also
bereits ins HailoRT-Layout transponiert) in einen eigenen Run-Ordner. Damit
lassen sich Edge- und Referenzlauf mit ``thesis.analysis`` direkt vergleichen.

Grad-CAM (braucht Gradienten → nur hier bzw. im Schattenmodell) folgt als
eigenes Modul; Grundlage ist gcqf/gradcam.py aus dem Prototyp.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from ..common.config import load_config
from ..common.preprocess import load_uint8, normalize, to_nchw_batch
from ..common.runmeta import new_run_dir, write_meta
from ..common.subset import read_subset

# Schicht, deren Ausgang als CAM-Feature-Map dient (muss dem HEF-Ausgang entsprechen!)
CAM_LAYER = {
    "resnet18": "layer4",
    "mobilenet_v2": "features",
    "mobilenet_v3_large": "features",
}


def main() -> None:
    import torch  # noqa: PLC0415
    import torchvision.models as tvm  # noqa: PLC0415

    from .export_onnx import MODELS  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="FP32-Referenz (PyTorch)")
    ap.add_argument("config")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    items = read_subset(cfg.subset_path)[: args.limit]
    fn_name, weights_name = MODELS[cfg.model]
    model = getattr(tvm, fn_name)(weights=getattr(tvm, weights_name).IMAGENET1K_V1).eval().to(args.device)

    captured = {}
    layer = getattr(model, CAM_LAYER[cfg.model])
    layer.register_forward_hook(lambda _m, _i, o: captured.__setitem__("feat", o))

    run_dir = new_run_dir(cfg)
    write_meta(run_dir, cfg, extra={"n_images": len(items), "role": "reference", "device": args.device})

    ids, logits_all, feats_all = [], [], []
    with torch.no_grad(), open(run_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label", "top1", "top5"])
        for it in items:
            x = torch.from_numpy(to_nchw_batch(normalize(load_uint8(it.path)))).to(args.device)
            logits = model(x)[0].cpu().numpy()
            feat = captured["feat"][0].cpu().numpy().transpose(1, 2, 0)   # CHW → HWC
            top5 = np.argsort(logits)[::-1][:5]
            w.writerow([it.image_id, it.label, int(top5[0]), " ".join(map(str, top5))])
            ids.append(it.image_id)
            logits_all.append(logits.astype(np.float32))
            feats_all.append(feat.astype(np.float32))

    np.savez_compressed(run_dir / "logits.npz", ids=np.array(ids), logits=np.stack(logits_all))
    np.savez_compressed(run_dir / "features.npz", ids=np.array(ids), features=np.stack(feats_all))
    print(f"fertig → {run_dir}")


if __name__ == "__main__":
    main()
