"""FP32-Referenzlauf (PyTorch) mit demselben Datenformat wie der Edge-Runner.

    python -m thesis.reference.fp32_run configs/resnet18_fp32.yaml [--device cuda]

Schreibt ``logits.npz`` und – bei ``save_features: true`` – ``features.npz`` (Feature-Maps als [H, W, K], also
bereits ins HailoRT-Layout transponiert) in einen eigenen Run-Ordner. Damit
lassen sich Edge- und Referenzlauf mit ``thesis.analysis`` direkt vergleichen.

Mit ``--gradients`` zusätzlich ``gradients.npz`` (ids, target [N], gradients [N, H, W, K] =
d logit[target] / d Feature-Map, ebenfalls HWC) – dasselbe Format wie ``thesis-emulator-run``.
Grad-CAM selbst rechnet ``thesis.analysis.gradcam`` aus Features + Gradienten, für beide Seiten
mit demselben Code. Zielklasse: eigene Top-1 (= FP32-Top-1, Vorschlag E-D) oder per
``--target-run`` aus der predictions.csv eines anderen Laufs.

    python -m thesis.reference.fp32_run configs/mobilenet_v3_large_fp32_imagenet.yaml \
        --limit 200 --gradients
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..common.config import REPO_ROOT, load_config
from ..common.preprocess import load_uint8, normalize, to_nchw_batch
from ..common.progress import Progress
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
    ap.add_argument("--gradients", action="store_true",
                    help="zusätzlich d logit[target] / d Feature-Map speichern (gradients.npz)")
    ap.add_argument("--target-run", type=Path, default=None,
                    help="Zielklasse = top1 aus <run>/predictions.csv (Default: eigene top1)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    items = read_subset(cfg.subset_path)[: args.limit]
    fn_name, weights_name = MODELS[cfg.model]
    model = getattr(tvm, fn_name)(weights=getattr(tvm, weights_name).IMAGENET1K_V1).eval().to(args.device)

    captured = {}
    layer = getattr(model, CAM_LAYER[cfg.model])
    layer.register_forward_hook(lambda _m, _i, o: captured.__setitem__("feat", o))

    run_dir = new_run_dir(cfg)
    targets: dict[str, int] = {}
    if args.target_run:
        run = args.target_run if args.target_run.is_absolute() else REPO_ROOT / args.target_run
        with open(run / "predictions.csv", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("repeat", "0") in ("", "0"):
                    targets[row["image_id"]] = int(row["top1"])
        lacking = [it.image_id for it in items if it.image_id not in targets]
        if lacking:
            raise SystemExit(f"{len(lacking)} Bilder fehlen in {run}/predictions.csv, z. B. {lacking[0]}")

    write_meta(run_dir, cfg, extra={
        "n_images": len(items), "role": "reference", "device": args.device,
        "gradients": args.gradients,
        "target_run": str(args.target_run) if args.target_run else None,
        "cam_layer": CAM_LAYER[cfg.model],
    })

    ids, logits_all, feats_all, grads_all, tgt_all = [], [], [], [], []
    grad_ctx = torch.enable_grad() if args.gradients else torch.no_grad()
    with grad_ctx, open(run_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_id", "label", "top1", "top5"])
        prog = Progress(len(items), label=cfg.name)
        for it in items:
            x = torch.from_numpy(to_nchw_batch(normalize(load_uint8(it.path)))).to(args.device)
            out = model(x)
            logits = out[0].detach().cpu().numpy()
            feat_t = captured["feat"]
            feat = feat_t[0].detach().cpu().numpy().transpose(1, 2, 0)   # CHW → HWC
            if args.gradients:
                t = targets.get(it.image_id, int(np.argmax(logits)))
                # Gradient nur nach der Feature-Map, nicht nach den Gewichten
                (g,) = torch.autograd.grad(out[0, t], feat_t)
                grads_all.append(g[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32))
                tgt_all.append(t)
            top5 = np.argsort(logits)[::-1][:5]
            w.writerow([it.image_id, it.label, int(top5[0]), " ".join(map(str, top5))])
            ids.append(it.image_id)
            logits_all.append(logits.astype(np.float32))
            if cfg.save_features:
                feats_all.append(feat.astype(np.float32))
            prog.step()

    np.savez_compressed(run_dir / "logits.npz", ids=np.array(ids), logits=np.stack(logits_all))
    if feats_all:
        np.savez_compressed(run_dir / "features.npz", ids=np.array(ids), features=np.stack(feats_all))
    if grads_all:
        np.savez_compressed(run_dir / "gradients.npz", ids=np.array(ids), target=np.array(tgt_all),
                            gradients=np.stack(grads_all))
    print(f"fertig → {run_dir}")


if __name__ == "__main__":
    main()
