"""Zwei Run-Ordner vergleichen (z. B. FP32-Referenz vs. Hailo-8).

    python -m thesis.analysis.compare_runs results/<fp32-run> results/<edge-run>

Gibt je Bild aus: Top-1-Übereinstimmung, Logit-Abweichung (Cosinus) und – falls
beide Läufe Feature-Maps enthalten – die Eigen-CAM-Metriken, und falls beide zusätzlich
``gradients.npz`` haben, die Grad-CAM-Metriken (``thesis.analysis.gradcam``). Leere Grad-CAM-Karten
(ReLU überall 0, z. B. gesättigtes Logit) werden als ``gradcam_empty`` markiert und nicht gemittelt. Schreibt
``compare_<edge-run>.csv`` in den Referenz-Ordner und eine Kurzfassung auf stdout.
Das ist der End-to-End-Check aus AP0.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .eigencam import eigencam
from .gradcam import gradcam
from .metrics import align_to_reference, compare


def _load(run: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = run / f"{name}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["ids"], d[name]


def _load_grads(run: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(target [N], gradients [N, H, W, K]) in der Reihenfolge von features.npz bzw. logits.npz."""
    p = run / "gradients.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["target"], d["gradients"]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_run", type=Path)
    ap.add_argument("q_run", type=Path)
    args = ap.parse_args()

    ref_ids, ref_logits = _load(args.ref_run, "logits")
    q_ids, q_logits = _load(args.q_run, "logits")
    q_index = {i: k for k, i in enumerate(q_ids)}
    ref_feat, q_feat = _load(args.ref_run, "features"), _load(args.q_run, "features")
    have_feat = ref_feat is not None and q_feat is not None
    ref_grad, q_grad = _load_grads(args.ref_run), _load_grads(args.q_run)
    have_grad = have_feat and ref_grad is not None and q_grad is not None

    rows = []
    for k, image_id in enumerate(ref_ids):
        j = q_index.get(image_id)
        if j is None:
            continue
        row = {
            "image_id": image_id,
            "top1_match": int(ref_logits[k].argmax() == q_logits[j].argmax()),
            "logit_cos": _cos(ref_logits[k], q_logits[j]),
        }
        if have_feat:
            cam_ref = eigencam(ref_feat[1][k])
            cam_q, flipped = align_to_reference(cam_ref, eigencam(q_feat[1][j]))
            row.update({f"eigencam_{m}": v for m, v in compare(cam_ref, cam_q).as_row().items()})
            row["eigencam_flipped"] = int(flipped)
        if have_grad:
            t_ref, g_ref = ref_grad[0][k], ref_grad[1][k]
            t_q, g_q = q_grad[0][j], q_grad[1][j]
            gc_ref, e_ref = gradcam(ref_feat[1][k], g_ref)
            gc_q, e_q = gradcam(q_feat[1][j], g_q)
            row["gradcam_target_ref"], row["gradcam_target_q"] = int(t_ref), int(t_q)
            row["gradcam_empty"] = int(e_ref or e_q)
            if not (e_ref or e_q):
                row.update({f"gradcam_{m}": v for m, v in compare(gc_ref, gc_q).as_row().items()})
        rows.append(row)

    if not rows:
        raise SystemExit("Keine gemeinsamen image_ids – gleiche Subset-Datei verwendet?")
    out = args.ref_run / f"compare_{args.q_run.name}.csv"
    with open(out, "w", newline="") as f:
        # Spalten aller Zeilen (leere Grad-CAM-Zeilen haben keine Metriken)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields, restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"n = {len(rows)}")
    print(f"Top-1-Match:     {np.mean([r['top1_match'] for r in rows]):.3f}")
    print(f"Logit-Cosinus:   {np.mean([r['logit_cos'] for r in rows]):.4f}")
    if have_feat:
        for m in ("cc", "topk_iou", "spearman"):
            print(f"Eigen-CAM {m:<9} {np.mean([r[f'eigencam_{m}'] for r in rows]):.3f}")
    if have_grad:
        ok = [r for r in rows if not r["gradcam_empty"]]
        diff_t = sum(r["gradcam_target_ref"] != r["gradcam_target_q"] for r in rows)
        print(f"Grad-CAM: {len(ok)} Bilder, {len(rows) - len(ok)} leer, {diff_t} mit verschiedener Zielklasse")
        if diff_t:
            print("⚠️  Zielklassen unterscheiden sich – beide Läufe mit derselben --target-run erzeugen?")
        for m in ("cc", "topk_iou", "spearman"):
            v = [r[f"gradcam_{m}"] for r in ok]
            print(f"Grad-CAM  {m:<9} Mittel {np.mean(v):.3f}  Median {np.median(v):.3f}  P5 {np.percentile(v, 5):.3f}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
