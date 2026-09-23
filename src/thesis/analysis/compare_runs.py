"""Zwei Run-Ordner vergleichen (z. B. FP32-Referenz vs. Hailo-8).

    python -m thesis.analysis.compare_runs results/<fp32-run> results/<edge-run>

Gibt je Bild aus: Top-1-Übereinstimmung, Logit-Abweichung (Cosinus) und – falls
beide Läufe Feature-Maps enthalten – die Eigen-CAM-Metriken. Schreibt
``compare_<edge-run>.csv`` in den Referenz-Ordner und eine Kurzfassung auf stdout.
Das ist der End-to-End-Check aus AP0.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .eigencam import eigencam
from .metrics import align_to_reference, compare


def _load(run: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = run / f"{name}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["ids"], d[name]


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
        rows.append(row)

    if not rows:
        raise SystemExit("Keine gemeinsamen image_ids – gleiche Subset-Datei verwendet?")
    out = args.ref_run / f"compare_{args.q_run.name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"n = {len(rows)}")
    print(f"Top-1-Match:     {np.mean([r['top1_match'] for r in rows]):.3f}")
    print(f"Logit-Cosinus:   {np.mean([r['logit_cos'] for r in rows]):.4f}")
    if have_feat:
        for m in ("cc", "topk_iou", "spearman"):
            print(f"Eigen-CAM {m:<9} {np.mean([r[f'eigencam_{m}'] for r in rows]):.3f}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
