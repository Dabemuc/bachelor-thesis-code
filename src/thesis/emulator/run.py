"""Emulator-Lauf (P10+): quantisiertes HAR im DFC-Emulator auf einem festen Subset.

Beantwortet die drei Fragen aus Phase 0a (Vault: „Phase 0 – Defaults“):

  (a) Rechnet das trainierbare Keras-Modell (``get_keras_model(SDK_QUANTIZED, trainable=True)``)
      genauso wie ``runner.infer`` im selben Kontext?
  (b) Kommt man an die Gradienten der CAM-Schicht (Grad-CAM)?
  (c) Wie nah ist der Emulator am Chip (TF2)? → Die Ausgaben werden im selben Format wie die
      Pi-Läufe geschrieben, Vergleich auf dem Mac mit
      ``python -m thesis.analysis.compare_runs results/<chip-lauf> results/<emu-lauf>``.

Aufruf im DFC-Container auf dem x86-PC (gleiche Config wie für Compile und Pi-Lauf):

    thesis-emulator-run configs/resnet18_int8_lvl2.yaml --limit 200 \\
        --target-run results/20261021-012047_roosevelt_resnet18_fp32_imagenet

⚠️ Emulator und Chip müssen aus DEMSELBEN Compile stammen (Compile ist bei Stufe 2 nicht
reproduzierbar). Verwendet wird ``artifacts/<name>/<model>_quantized.har``; dessen SHA-256 und
der des HEF daneben werden protokolliert.

Je Kontext entsteht ein Run-Ordner ``results/<zeit>_<host>_<name>_emu_<kontext>/`` mit
    meta.json, predictions.csv, logits.npz, features.npz   (Format wie thesis.edge.runner)
    gradients.npz   ids, target [N], gradients [N, H, W, K] = d logit[target] / d A   (mit --gradients)
    checks.json     Ergebnisse der Prüfungen (a) und (b), Namen der Schichten, Eingabeformat

Grundlage der API (DFC 3.33.1, Quelltext des Wheels, gelesen 03.11.2026):
  * ``ClientRunner.infer`` → Liste der Ausgaben in Reihenfolge ``HailoModel.flow.output_nodes``
    (``hailo_model_optimization/flows/inference_flow.py``, ``InferenceModel.run``)
  * ``ClientRunner.get_keras_model`` → ``SimulationTrainingModel`` bzw. ``SimulationInferenceModel``,
    das zugrunde liegende ``HailoModel`` unter ``.model`` (``sdk_backend.py``: ``get_emulation_model``)
  * ``HailoModel.set_output_interal_layers([...])`` → Aufruf liefert ``(outputs, [interne Ausgaben])``
    (``hailo_model.py``; Schreibweise „interal“ wie im DFC)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from ..common.config import ARTIFACTS_DIR, REPO_ROOT, load_config
from ..common.preprocess import load_uint8, normalize
from ..common.progress import Progress
from ..common.runmeta import hef_provenance, new_run_dir, sha256_file, write_meta
from ..common.subset import read_subset

CONTEXTS = ("native", "fp_optimized", "quantized", "bit_exact")
NUM_CLASSES = 1000


def _to_np(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _as_list(out) -> list[np.ndarray]:
    return [_to_np(o) for o in out] if isinstance(out, (list, tuple)) else [_to_np(out)]


def _split(outs: list[np.ndarray]) -> tuple[int, int | None]:
    """Index des Logits- und des Feature-Ausgangs (Form [N, H, W, K] mit H·W > 1)."""
    i_log = i_feat = None
    for i, a in enumerate(outs):
        per = a.shape[1:]
        if int(np.prod(per)) == NUM_CLASSES:
            i_log = i
        elif a.ndim == 4 and per[0] * per[1] > 1:
            i_feat = i
    if i_log is None:
        raise RuntimeError(f"Kein Logits-Ausgang gefunden: {[a.shape for a in outs]}")
    return i_log, i_feat


def _read_targets(run: Path) -> dict[str, int]:
    """image_id → top1 aus predictions.csv eines Referenzlaufs (erste Wiederholung)."""
    out: dict[str, int] = {}
    with open(run / "predictions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("repeat", "0") in ("", "0"):
                out[row["image_id"]] = int(row["top1"])
    return out


def _write_predictions(run_dir: Path, items, logits: np.ndarray) -> None:
    with open(run_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_id", "label", "top1", "top5"])
        for it, lg in zip(items, logits):
            top5 = np.argsort(lg)[::-1][:5]
            w.writerow([it.image_id, it.label, int(top5[0]), " ".join(map(str, top5))])


def _has_normalization(hailo_model) -> bool:
    return any("normalization" in name.split("/")[-1] for name in hailo_model.layers)


def run_context(runner, ctx_name: str, cfg, items, imgs_uint8: np.ndarray, har: Path, args, targets):
    import tensorflow as tf  # noqa: PLC0415
    from hailo_sdk_client import InferenceContext  # noqa: PLC0415

    ctx_enum = getattr(InferenceContext, f"SDK_{ctx_name.upper()}")
    checks: dict = {"context": str(ctx_enum)}
    n = len(items)

    with runner.infer_context(ctx_enum) as ctx:
        trainable = ctx_name == "quantized"
        km = runner.get_keras_model(ctx, trainable=trainable)
        hm = km.model
        checks["keras_model"] = {"class": type(km).__name__, "trainable": trainable}

        # Eingabe: Ist die Normalisierung aus dem Model Script in diesem Kontext Teil des Modells?
        # (SDK_NATIVE = Originalmodell ohne Modifikationen → evtl. ohne normalization1)
        norm_in_model = _has_normalization(hm)
        if cfg.normalize_on_chip and norm_in_model:
            x_all = imgs_uint8.astype(np.float32)
            checks["input"] = "uint8-Werte als float32 (0–255), Normalisierung im Modell"
        else:
            x_all = np.stack([normalize(im) for im in imgs_uint8])
            checks["input"] = "auf dem Host normalisiert (float32)"
        checks["normalization_layer_in_model"] = norm_in_model

        # --- Ausgaben über runner.infer (der offizielle Weg) -----------------------------------
        outs = _as_list(runner.infer(ctx, x_all, batch_size=args.batch_size))
        i_log, i_feat = _split(outs)
        logits = outs[i_log].reshape(n, -1).astype(np.float32)
        feats = outs[i_feat].astype(np.float32) if i_feat is not None else None
        out_nodes = list(hm.flow.output_nodes)
        checks["output_nodes"] = out_nodes
        checks["logits_node"] = out_nodes[i_log]
        checks["features_node"] = out_nodes[i_feat] if i_feat is not None else None
        checks["output_shapes"] = [list(a.shape) for a in outs]

        run_dir = new_run_dir(cfg, tag=f"emu_{ctx_name}")
        write_meta(run_dir, cfg, extra={
            "n_images": n,
            "role": "emulator",
            "emulator_context": str(ctx_enum),
            "har": {"path": str(har), "sha256": sha256_file(har)},
            "hef": hef_provenance(cfg.hef_path) if cfg.hef_path and cfg.hef_path.exists() else None,
            "target_run": str(args.target_run) if args.target_run else None,
        })
        ids = np.array([it.image_id for it in items])
        _write_predictions(run_dir, items, logits)
        np.savez_compressed(run_dir / "logits.npz", ids=ids, logits=logits)
        if feats is not None and cfg.save_features:
            np.savez_compressed(run_dir / "features.npz", ids=ids, features=feats)

        # --- (a) + (b): direkter Aufruf des Keras-Modells, mit interner CAM-Schicht -------------
        # bit_exact: get_keras_model liefert dafür laut sdk_backend.get_emulation_model ein NATIVES Modell
        # (nur SDK_QUANTIZED wird auf quantisiert gestellt) → Prüfungen (a)/(b) dort nicht aussagekräftig.
        if ctx_name == "bit_exact":
            checks["gradients_skipped"] = "bit_exact: Keras-Modell wäre nativ (sdk_backend.get_emulation_model)"
        elif args.gradients and i_feat is not None:
            try:
                feat_layer = hm.flow.predecessors_sorted(out_nodes[i_feat])[0]
                checks["cam_layer"] = feat_layer
                hm.set_output_interal_layers([feat_layer])
                # HailoModel direkt aufrufen statt über den Keras-Wrapper: dessen build()
                # (inference_flow.get_build_inputs) erwartet ein tf.data-Dataset und scheiterte am
                # Tensor-Batch ebenso wie an einem dict (03.11.2026). Gebaut wird daher selbst, mit
                # derselben Form wie get_build_inputs sie für Tensoren bildet: (1, H, W, C).
                if not hm.built:
                    hm.build((1, *x_all.shape[1:]))
                # training=True: Nur dann rundet Hailo mit STE (fake_quant_int, @tf.custom_gradient).
                # Mit training=False ist der Modus DISABLED → nacktes tf.round ohne Gradient → None
                # (acceleras/lossy_elements/quant_element.py, BaseQuantElement.quant_call; Lauf 03.11.2026:
                # 25/25 Batches ohne Gradient). Vorwärts ist STE = clip(round(x)), also gleiche Zahlen;
                # geprüft unten über (a) gegen runner.infer.
                checks["called"] = "HailoModel direkt (km.model), training=True (STE)"

                # Die interne Schicht liefert im quantisierten Kontext Werte im Integer-Bereich, der
                # Feature-Ausgang dagegen dequantisiert: real = (int - zp) * scale. Für Grad-CAM auf der
                # Skala der Chip-Feature-Maps: dY/dA_real = dY/dA_int / scale.
                lay = hm.layers[feat_layer]
                oi = hm.flow.get_edge_output_index(feat_layer, out_nodes[i_feat])
                oi = lay.resolve_output_index(oi) if hasattr(lay, "resolve_output_index") else oi
                scale = np.asarray(_to_np(lay.output_scales[oi]), dtype=np.float32).reshape(-1)
                zp = np.asarray(_to_np(lay.output_zero_points[oi]), dtype=np.float32).reshape(-1)
                checks["cam_layer_output_scale"] = {"n": int(scale.size), "min": float(scale.min()), "max": float(scale.max())}
                checks["cam_layer_output_zero_point"] = {"n": int(zp.size), "min": float(zp.min()), "max": float(zp.max())}

                if args.target_run:
                    tgt = np.array([targets[it.image_id] for it in items])
                    checks["target"] = f"top1 aus {args.target_run}"
                else:
                    tgt = logits.argmax(1)
                    checks["target"] = "eigene top1 dieses Kontexts"

                grads_all, d_log, d_feat, d_int, d_deq = [], [], [], [], []
                n_none = 0
                prog = Progress(n, label=f"{cfg.name} emu_{ctx_name} Gradienten")
                for s in range(0, n, args.batch_size):
                    xb = tf.constant(x_all[s:s + args.batch_size])
                    bs = int(xb.shape[0])
                    tb = tgt[s:s + args.batch_size]
                    with tf.GradientTape() as tape:
                        res = hm(xb, training=True)
                        outs_b, internals = res
                        outs_b = list(outs_b) if isinstance(outs_b, (list, tuple)) else [outs_b]
                        act = internals[0]
                        lg = tf.reshape(outs_b[i_log], (bs, -1))
                        sel = tf.gather(lg, tb, axis=1, batch_dims=1)
                        y = tf.reduce_sum(sel)  # Bilder sind unabhängig (Inferenz) → Summe = Einzelgradienten
                    g = tape.gradient(y, act)
                    if g is None:
                        n_none += 1
                        g = tf.zeros_like(act)
                    grads_all.append(_to_np(g).astype(np.float32))
                    act_np = _to_np(act).astype(np.float32)
                    # (a) gleiche Zahlen wie runner.infer?
                    d_log.append(np.abs(_to_np(lg) - logits[s:s + args.batch_size]).max())
                    if feats is not None:
                        fb = _to_np(outs_b[i_feat])
                        d_feat.append(np.abs(fb - feats[s:s + args.batch_size]).max())
                        # interne Schicht = Feature-Ausgang? (Ausgangsschicht nur Durchreichung/Dequantisierung?)
                        d_int.append(np.abs(act_np - fb).max())
                        d_deq.append(np.abs((act_np - zp) * scale - fb).max())
                    prog.step(bs)
                hm.reset_output_interal_layers()

                grads = np.concatenate(grads_all)
                # Umrechnung nur, wenn die Dequantisierung die interne Schicht tatsächlich auf den
                # Feature-Ausgang abbildet (im nativen Kontext ist beides schon gleich).
                if d_int and max(d_deq) < max(d_int):
                    grads = grads / scale
                    checks["gradient_scale"] = "dY/dA_real = dY/dA_int / scale (Feature-Ausgang-Skala)"
                else:
                    checks["gradient_scale"] = "unverändert (interne Schicht = Feature-Ausgang)"
                checks["internal_dequantized_vs_feature_output_max_abs_diff"] = float(max(d_deq)) if d_deq else None
                checks["a_max_abs_diff_logits_keras_vs_infer"] = float(max(d_log))
                checks["a_max_abs_diff_features_keras_vs_infer"] = float(max(d_feat)) if d_feat else None
                checks["internal_vs_feature_output_max_abs_diff"] = float(max(d_int)) if d_int else None
                checks["b_gradient_none_batches"] = n_none
                checks["b_gradient_finite"] = bool(np.isfinite(grads).all())
                checks["b_gradient_nonzero_fraction"] = float((grads != 0).mean())
                checks["b_gradient_abs_mean"] = float(np.abs(grads).mean())
                checks["b_gradient_shape"] = list(grads.shape)
                np.savez_compressed(run_dir / "gradients.npz", ids=ids, target=tgt, gradients=grads)
            except Exception as e:
                checks["error"] = f"{type(e).__name__}: {e}"
                (run_dir / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
                raise

    (run_dir / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print(f"fertig → {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantisiertes HAR im Hailo-Emulator laufen lassen (P10+)")
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None, help="nur die ersten N Bilder des Subsets")
    ap.add_argument("--contexts", default="native,quantized",
                    help=f"Komma-Liste aus {', '.join(CONTEXTS)} (bit_exact ist im DFC nur „preview“)")
    ap.add_argument("--no-gradients", dest="gradients", action="store_false",
                    help="keine Gradienten/Prüfungen (a), (b) – nur Ausgaben")
    ap.add_argument("--target-run", type=Path, default=None,
                    help="Run-Ordner mit predictions.csv, dessen top1 die Zielklasse für die Gradienten ist "
                         "(Vorschlag E-D: FP32-Referenz)")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    ctxs = [c.strip() for c in args.contexts.split(",") if c.strip()]
    bad = [c for c in ctxs if c not in CONTEXTS]
    if bad:
        sys.exit(f"Unbekannte Kontexte: {bad} (erlaubt: {CONTEXTS})")

    try:
        from hailo_sdk_client import ClientRunner  # noqa: PLC0415
    except ImportError:
        sys.exit("hailo_sdk_client fehlt – dieses Skript läuft nur im DFC-Container (x86_64).")

    cfg = load_config(args.config)
    har = ARTIFACTS_DIR / cfg.name / f"{cfg.model}_quantized.har"
    if not har.exists():
        sys.exit(f"Quantisiertes HAR fehlt: {har} (entsteht bei thesis-compile)")
    items = read_subset(cfg.subset_path)[: args.limit]
    missing = [it.path for it in items if not Path(it.path).exists()]
    if missing:
        sys.exit(f"{len(missing)} Bilder fehlen, z. B. {missing[0]} → scripts/sync_subset_images.sh {cfg.subset} <ziel> {len(items)}")

    targets: dict[str, int] = {}
    if args.target_run:
        run = args.target_run if args.target_run.is_absolute() else REPO_ROOT / args.target_run
        targets = _read_targets(run)
        lacking = [it.image_id for it in items if it.image_id not in targets]
        if lacking:
            sys.exit(f"{len(lacking)} Bilder fehlen in {run}/predictions.csv, z. B. {lacking[0]}")

    imgs = np.stack([load_uint8(it.path) for it in items])
    runner = ClientRunner(har=str(har))
    for c in ctxs:
        try:
            run_context(runner, c, cfg, items, imgs, har, args, targets)
        except Exception as e:  # ein Kontext (z. B. bit_exact „preview“) soll die anderen nicht verhindern
            print(f"⚠️  Kontext {c} fehlgeschlagen: {type(e).__name__}: {e}", file=sys.stderr)
            if c == "quantized":
                raise


if __name__ == "__main__":
    main()
