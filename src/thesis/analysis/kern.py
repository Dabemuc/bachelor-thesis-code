"""Kern-Auswertung (Phase 2): Inventar, Prüfung, Extraktion und Metriken je Bild.

Ablauf (Notebook ``notebooks/00_daten.ipynb``):

1. ``inventory``        alle Kern-Blöcke aus ``results/*/meta.json``
2. ``check_inventory``  Vollständigkeit, dirty, HEF-Hash, Bild-IDs
3. ``extract_all``      jeden Block einmal lesen, kleine Zwischendateien schreiben:
                        7×7-Rohkarten (Eigen-CAM, Grad-CAM), Ausgabegrößen je Bild,
                        Aktivierungsabstand Emulator ↔ Chip (TF2)
4. ``compute_tables``   Metriken je Bild und Vergleich als Tabellen (Parquet)

Alle Schritte sind wiederaufnehmbar: Vorhandene Zwischendateien werden übersprungen.

Festlegungen (Auswertungsplan, eingefroren 24.11.2026, Tag ``auswertungsplan-v1``):

* Zielklasse für Grad-CAM und für ΔP/Logit-Ziel ist die FP32-Top-1 (Abschn. 2).
* Eigen-CAM ohne Ausrichtung an FP32; ``eig_flip`` zeigt, ob die Ausrichtung umgedreht
  hätte, ``*_al``-Spalten sind die ausgerichtete Sensitivitätsvariante (Abschn. 8).
* Kollaps = konstante Karte (Abschn. 5). Leere Grad-CAM-Karten werden markiert
  (``grad_empty``); ob Sättigung die Ursache ist, zeigt ``sat`` (Abschn. 8).
* Emulator: Eigen-CAM und Logits aus ``runner.infer`` (``features.npz``, ``logits.npz``),
  Grad-CAM aus dem Keras-Modell (``features_keras.npz`` + ``gradients.npz``), damit
  Aktivierungen und Gradienten aus demselben Modell stammen.

Quellen (Spalte ``source``): ``fp32`` · ``chip_kNN`` (48 000) · ``emu_kNN`` (Teilmenge 5 000)
· ``emuvoll_kNN`` (48 000, nur k01). ``fp32_teil`` ist der separate FP32-Lauf auf der
Teilmenge und dient nur als Konsistenzprüfung.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import DATA_DIR, REPO_ROOT, RESULTS_DIR
from ..common.subset import read_subset
from .eigencam import first_pc_projection
from .gradcam import gradcam_raw
from .maps import (TOPK_FRACTIONS, is_constant, pairwise_cc, pairwise_iou, spearman_kld,
                   unit_maps, upsampled_range)

MODELS = ("resnet18", "mobilenet_v3_large")
CACHE_DIR = REPO_ROOT / "auswertung_cache"


@dataclass
class KernSpec:
    """Was im Kern erwartet wird. Die Defaults sind der tatsächliche Kern (16.–18.11.2026)."""

    results_dir: Path = RESULTS_DIR
    list_voll: Path = DATA_DIR / "subsets" / "imagenet_eval_48000.txt"
    list_teil: Path = DATA_DIR / "subsets" / "imagenet_eval_strat5000_seed2.txt"
    models: tuple[str, ...] = MODELS
    k: int = 10
    emuvoll_k: tuple[int, ...] = (1,)
    started_after: str = "2026-11-16T20:00"
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- Inventar

_NAME_Q = re.compile(r"^(?P<model>.+)_int8_lvl2_k(?P<k>\d\d)$")
_NAME_F = re.compile(r"^(?P<model>.+)_fp32_imagenet$")


def inventory(spec: KernSpec = KernSpec()) -> pd.DataFrame:
    """Eine Zeile je Kern-Block (Run-Ordner)."""
    voll, teil = Path(spec.list_voll).name, Path(spec.list_teil).name
    rows = []
    for meta in sorted(Path(spec.results_dir).glob("*/meta.json")):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        c = m.get("config", {})
        if m.get("started", "") < spec.started_after or c.get("subset") not in (voll, teil):
            continue
        role, name = m.get("role"), c.get("name", "")
        menge = "voll" if c["subset"] == voll else "teil"
        if role == "reference" and (hit := _NAME_F.match(name)):
            k, source = None, "fp32" if menge == "voll" else "fp32_teil"
        elif role in ("edge", "emulator") and (hit := _NAME_Q.match(name)):
            k = int(hit["k"])
            if role == "edge":
                source = f"chip_k{k:02d}"
            elif m.get("emulator_context") != "sdk_quantized":
                continue
            else:
                source = f"emu_k{k:02d}" if menge == "teil" else f"emuvoll_k{k:02d}"
        else:
            continue
        if hit["model"] not in spec.models:
            continue
        files = sorted(p.name for p in meta.parent.iterdir())
        rows.append({
            "run": meta.parent.name, "path": str(meta.parent), "model": hit["model"],
            "source": source, "role": role, "k": k, "menge": menge,
            "offset": int(m.get("offset") or 0), "n": int(m["n_images"]),
            "commit": m["git"]["commit"][:7], "dirty": bool(m["git"]["dirty"]),
            "hef_sha": (m.get("hef") or {}).get("sha256", "")[:12],
            "target_run": Path(m["target_run"]).name if m.get("target_run") else None,
            "files": " ".join(files),
        })
    inv = pd.DataFrame(rows)
    return inv.sort_values(["model", "source", "offset"]).reset_index(drop=True)


def _ids_of(path: Path) -> np.ndarray:
    with np.load(Path(path) / "logits.npz") as z:
        return z["ids"]


def _list_ids(path: Path) -> np.ndarray:
    return np.array([it.image_id for it in read_subset(path)])


def check_inventory(inv: pd.DataFrame, spec: KernSpec = KernSpec(), check_ids: bool = True) -> pd.DataFrame:
    """Prüfungen wie am 18.11.2026, als Tabelle (``ok`` = bestanden)."""
    out = []

    def add(name, ok, detail=""):
        out.append({"prüfung": name, "ok": bool(ok), "detail": detail})

    lists = {"voll": _list_ids(spec.list_voll), "teil": _list_ids(spec.list_teil)}
    need = {"reference": {"logits.npz", "features.npz", "gradients.npz"},
            "edge": {"logits.npz", "features.npz"},
            "emulator": {"logits.npz", "features.npz", "features_keras.npz", "gradients.npz", "consistency.npz"}}
    for model in spec.models:
        sub = inv[inv.model == model]
        expected = ["fp32", "fp32_teil"] + [f"chip_k{k:02d}" for k in range(1, spec.k + 1)] \
            + [f"emu_k{k:02d}" for k in range(1, spec.k + 1)] + [f"emuvoll_k{k:02d}" for k in spec.emuvoll_k]
        missing = sorted(set(expected) - set(sub.source))
        add(f"{model}: alle Quellen vorhanden", not missing, ", ".join(missing))
        for source, g in sub.groupby("source"):
            menge = g.menge.iloc[0]
            total = len(lists[menge])
            g = g.sort_values("offset")
            contiguous = (g.offset.values == np.r_[0, np.cumsum(g.n.values)[:-1]]).all()
            add(f"{model}/{source}: Blöcke lückenlos, ohne Doppel", contiguous and g.n.sum() == total
                and not g.offset.duplicated().any(), f"{len(g)} Blöcke, n = {g.n.sum()} von {total}")
            files_ok = all(need[r] <= set(f.split()) for r, f in zip(g.role, g.files))
            add(f"{model}/{source}: alle Dateien", files_ok)
            if check_ids:
                ids_ok = all((_ids_of(Path(p)) == lists[menge][o:o + n]).all()
                             for p, o, n in zip(g.path, g.offset, g.n))
                add(f"{model}/{source}: Bild-IDs = Listenausschnitt", ids_ok)
        add(f"{model}: kein Lauf dirty", not sub.dirty.any(), ", ".join(sub[sub.dirty].run))
        q = sub[sub.k.notna()]
        per_k = q.groupby("k").hef_sha.nunique()
        add(f"{model}: HEF-Hash je Compile gleich (Chip = Emulator)", (per_k == 1).all(),
            ", ".join(f"k{int(k):02d}" for k in per_k[per_k != 1].index))
        add(f"{model}: {spec.k} verschiedene Compiles", q.groupby("k").hef_sha.first().nunique() == spec.k)
        emu = sub[sub.role == "emulator"]
        teil_ref = set(sub[sub.source == "fp32_teil"].run)
        voll_ref = set(sub[sub.source == "fp32"].run)
        ok_t = emu[emu.menge == "teil"].target_run.isin(teil_ref).all()
        ok_v = emu[emu.menge == "voll"].target_run.isin(voll_ref).all()
        add(f"{model}: Emulator-Zielklassen aus dem FP32-Lauf derselben Liste", ok_t and ok_v)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- Extraktion

def _softmax_p(logits: np.ndarray, cls: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[np.arange(len(cls)), cls] / e.sum(axis=1)


def _outputs(logits: np.ndarray, ref_logits: np.ndarray, target: np.ndarray) -> dict:
    """Ausgabegrößen je Bild relativ zu FP32 (Zielklasse = FP32-Top-1)."""
    lg = logits.astype(np.float64)
    rf = ref_logits.astype(np.float64)
    cos = (lg * rf).sum(1) / (np.linalg.norm(lg, axis=1) * np.linalg.norm(rf, axis=1) + 1e-12)
    return {
        "top1": logits.argmax(1).astype(np.int16),
        "top5": np.argsort(-logits, axis=1)[:, :5].astype(np.int16),
        "logit_cos": cos,
        "p_t": _softmax_p(logits, target),
        "logit_t": logits[np.arange(len(target)), target].astype(np.float32),
        "logit_max": logits.max(1).astype(np.float32),
    }


def _eig_raw(features: np.ndarray) -> np.ndarray:
    return np.stack([first_pc_projection(f) for f in features]).astype(np.float32)


def _grad_raw(features: np.ndarray, grads: np.ndarray) -> np.ndarray:
    return np.stack([gradcam_raw(f, g) for f, g in zip(features, grads)]).astype(np.float32)


def _block_file(cache: Path, run: str) -> Path:
    return cache / "blocks" / f"{run}.npz"


def _extract_block(row: dict, cache: str, fp32_logits_file: str | None) -> str:
    """Einen Run-Ordner lesen, kleine Zwischendatei schreiben. Läuft auch im Unterprozess."""
    cache = Path(cache)
    out = _block_file(cache, row["run"])
    if out.exists():
        return f"übersprungen {row['run']}"
    t0 = time.time()
    run = Path(row["path"])
    with np.load(run / "logits.npz") as z:
        ids, logits = z["ids"], z["logits"]
    res = {"ids": ids}
    if row["role"] == "reference":
        res["logits"] = logits
        target = logits.argmax(1)
    else:
        with np.load(fp32_logits_file) as z:
            index = {i: n for n, i in enumerate(z["ids"])}
            sel = np.array([index[i] for i in ids])
            ref = z["logits"][sel]
        target = ref.argmax(1)
        res.update(_outputs(logits, ref, target))
    with np.load(run / "features.npz") as z:
        assert (z["ids"] == ids).all()
        feats = z["features"]
    res["eig_raw"] = _eig_raw(feats)
    if row["role"] == "emulator":
        with np.load(run / "features_keras.npz") as z:
            assert (z["ids"] == ids).all()
            feats = z["features"]
        with np.load(run / "consistency.npz") as z:
            res["keras_diff_logits"] = z["max_abs_diff_logits"]
            res["keras_diff_features"] = z["max_abs_diff_features"]
    if row["role"] in ("reference", "emulator"):
        with np.load(run / "gradients.npz") as z:
            assert (z["ids"] == ids).all()
            grads, gt = z["gradients"], z["target"]
        # Grad-CAM-Ziel muss die FP32-Top-1 sein. Abweichungen (z. B. Gleitkomma-Gleichstand im
        # separaten FP32-Lauf der Teilmenge) werden markiert statt abzubrechen.
        res["grad_target_ok"] = gt == target
        res["grad_raw"] = _grad_raw(feats, grads)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **res)
    tmp.rename(out)
    return f"fertig {row['run']} ({time.time() - t0:.0f} s)"


def _merge_source(inv: pd.DataFrame, cache: Path, model: str, source: str, list_ids: np.ndarray) -> Path:
    """Blockdateien einer Quelle in Listenreihenfolge zusammenfügen."""
    out = cache / model / f"{source}.npz"
    if out.exists():
        return out
    parts = [dict(np.load(_block_file(cache, r))) for r in
             inv[(inv.model == model) & (inv.source == source)].sort_values("offset").run]
    res = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    if not (res["ids"] == list_ids).all():
        raise ValueError(f"{model}/{source}: Bild-IDs nach dem Zusammenfügen ≠ Liste")
    for cam in ("eig", "grad"):
        if f"{cam}_raw" in res:
            lo, hi = upsampled_range(res[f"{cam}_raw"])
            res[f"{cam}_lo"], res[f"{cam}_hi"] = lo, hi
    if "grad_raw" in res:
        res["grad_empty"] = res["grad_raw"].reshape(len(res["ids"]), -1).max(1) <= 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **res)
    return out


class _LRU:
    def __init__(self, size: int):
        self.size, self.d = size, OrderedDict()

    def get(self, key, load):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        val = load(key)
        self.d[key] = val
        if len(self.d) > self.size:
            self.d.popitem(last=False)
        return val


def _activation_gap(chip_paths: list[str], emu_paths: list[str], out: str) -> str:
    """TF2 Aktivierungsebene: Feature-Map Emulator (runner.infer) ↔ Chip desselben Compiles."""
    out = Path(out)
    if out.exists():
        return f"übersprungen {out.name}"
    t0 = time.time()
    emu_ids = {p: _ids_of(Path(p)) for p in emu_paths}
    where = {i: p for p, ids in emu_ids.items() for i in ids}

    def load(p):
        with np.load(Path(p) / "features.npz") as z:
            return {i: n for n, i in enumerate(z["ids"])}, z["features"]

    cache = _LRU(3)
    res = {"ids": [], "rel_err": [], "cos": [], "frac_equal": []}
    for cp in chip_paths:
        with np.load(Path(cp) / "features.npz") as z:
            c_ids, c_feat = z["ids"], z["features"]
        for n, i in enumerate(c_ids):
            p = where.get(i)
            if p is None:
                continue
            idx, e_feat = cache.get(p, load)
            e = e_feat[idx[i]].astype(np.float64).ravel()
            c = c_feat[n].astype(np.float64).ravel()
            nc = np.linalg.norm(c)
            res["ids"].append(i)
            res["rel_err"].append(np.linalg.norm(e - c) / (nc + 1e-12))
            res["cos"].append(e @ c / (np.linalg.norm(e) * nc + 1e-12))
            res["frac_equal"].append(np.mean(e == c))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.asarray(v) for k, v in res.items()})
    return f"fertig {out.name} ({time.time() - t0:.0f} s)"


def extract_all(inv: pd.DataFrame, spec: KernSpec = KernSpec(), cache: Path = CACHE_DIR,
                n_jobs: int = 2, log=print) -> None:
    """Alle Blöcke extrahieren und je Quelle zusammenfügen, dann den Aktivierungsabstand.

    ``n_jobs`` = parallele Prozesse. Speicher je Prozess bis ≈ 3 GB (MobileNetV3-Chipblock).
    """
    cache = Path(cache)
    lists = {"voll": _list_ids(spec.list_voll), "teil": _list_ids(spec.list_teil)}
    rows = inv.to_dict("records")

    def run_all(jobs):
        if n_jobs <= 1:
            for fn, args in jobs:
                log(fn(*args))
        else:
            with ProcessPoolExecutor(n_jobs) as ex:
                for msg in ex.map(_call, jobs):
                    log(msg)

    # 1) FP32 zuerst: die anderen Quellen brauchen die FP32-Logits
    run_all([(_extract_block, (r, str(cache), None)) for r in rows if r["role"] == "reference"])
    for model in spec.models:
        for src, menge in (("fp32", "voll"), ("fp32_teil", "teil")):
            _merge_source(inv, cache, model, src, lists[menge])
    # 2) Chip und Emulator
    ref = {m: str(cache / m / "fp32.npz") for m in spec.models}
    run_all([(_extract_block, (r, str(cache), ref[r["model"]])) for r in rows if r["role"] != "reference"])
    for model in spec.models:
        for src, g in inv[(inv.model == model) & (inv.role != "reference")].groupby("source"):
            _merge_source(inv, cache, model, src, lists[g.menge.iloc[0]])
            log(f"zusammengefügt {model}/{src}")
    # 3) Aktivierungsabstand Emulator ↔ Chip (TF2)
    jobs = []
    for model in spec.models:
        sub = inv[inv.model == model]
        for k in range(1, spec.k + 1):
            chip = list(sub[sub.source == f"chip_k{k:02d}"].sort_values("offset").path)
            for prefix in ("emu", "emuvoll"):
                emu = list(sub[sub.source == f"{prefix}_k{k:02d}"].sort_values("offset").path)
                if emu:
                    jobs.append((_activation_gap, (chip, emu, str(cache / model / f"akt_{prefix}_k{k:02d}.npz"))))
    run_all(jobs)


def _call(job):
    fn, args = job
    return fn(*args)


# --------------------------------------------------------------------------- Tabellen

def load_source(cache: Path, model: str, source: str) -> dict:
    with np.load(Path(cache) / model / f"{source}.npz") as z:
        return {k: z[k] for k in z.files}


def _sources(cache: Path, model: str) -> list[str]:
    return sorted(p.stem for p in (Path(cache) / model).glob("*.npz") if not p.stem.startswith("akt_"))


def _k_of(source: str) -> float:
    m = re.search(r"_k(\d\d)$", source)
    return float(m[1]) if m else np.nan


def quellen_table(cache: Path, model: str, spec: KernSpec = KernSpec()) -> pd.DataFrame:
    """Je Quelle × Bild: Vorhersage, Ausgabegrößen, Karteneigenschaften, Sättigung."""
    labels = {it.image_id: it.label for it in read_subset(spec.list_voll)}
    teil = set(_list_ids(spec.list_teil))
    fp = load_source(cache, model, "fp32")
    t = fp["logits"].argmax(1)
    frames = []
    fp_out = _outputs(fp["logits"], fp["logits"], t)
    for source in _sources(cache, model):
        if source == "fp32_teil":
            continue
        d = fp if source == "fp32" else load_source(cache, model, source)
        o = fp_out if source == "fp32" else d
        ids = d["ids"]
        label = np.array([labels[i] for i in ids])
        f = pd.DataFrame({
            "model": model, "source": source, "k": _k_of(source), "image_id": ids,
            "label": label, "in_teil": [i in teil for i in ids],
            "top1": o["top1"], "top1_ok": o["top1"] == label,
            "top5_ok": (o["top5"] == label[:, None]).any(1),
            "logit_cos": o["logit_cos"], "p_t": o["p_t"], "logit_t": o["logit_t"], "logit_max": o["logit_max"],
            "eig_const": d["eig_hi"] - d["eig_lo"] <= 0,
        })
        if "grad_empty" in d:
            f["grad_empty"] = d["grad_empty"]
            f["grad_target_ok"] = d["grad_target_ok"]
        if "keras_diff_logits" in d:
            f["keras_diff_logits"] = d["keras_diff_logits"]
            f["keras_diff_features"] = d["keras_diff_features"]
        frames.append(f)
    q = pd.concat(frames, ignore_index=True)
    # Sättigung: Ziel-Logit am oberen Ende des Ausgabebereichs des Compiles
    # (Obergrenze = größter Ausgabewert über Chip und Emulator desselben Compiles, wie E2 Phase 1)
    top = q[q.k.notna()].groupby("k").logit_max.max()
    q["range_max"] = q.k.map(top)
    q["sat"] = q.k.notna() & (q.logit_t >= q.range_max)
    t_index = pd.Series(t, index=fp["ids"])
    q["fp32_top1"] = t_index.loc[q.image_id].values
    return q


_MAPS: dict = {}
_METRIC_KEYS = tuple(f"iou{round(f * 100)}" for f in TOPK_FRACTIONS) + ("cc",) \
    + tuple(f"iou{round(f * 100)}_al" for f in TOPK_FRACTIONS) + ("cc_al",)


def _load_maps(cache: str, model: str) -> dict:
    """Nur die Karten-Arrays aller Quellen, einmal je Prozess."""
    key = (cache, model)
    if key not in _MAPS:
        out = {}
        for s in _sources(Path(cache), model):
            if s == "fp32_teil":
                continue
            with np.load(Path(cache) / model / f"{s}.npz") as z:
                out[s] = {k: z[k] for k in ("ids", "eig_raw", "grad_raw", "grad_empty") if k in z.files}
        _MAPS.clear()
        _MAPS[key] = out
    return _MAPS[key]


def _compare_chunk(cache: str, model: str, idx: np.ndarray, with_rank: bool) -> pd.DataFrame:
    """Alle Vergleiche für einen Ausschnitt der 48 000 Bilder (Zeilen = Paar × Bild)."""
    data = _load_maps(cache, model)
    fp_ids = data["fp32"]["ids"]
    srcs = sorted(data)
    chips = [s for s in srcs if s.startswith("chip_")]
    emus = [s for s in srcs if s.startswith("emu_")]
    emuvoll = [s for s in srcs if s.startswith("emuvoll_")]
    pos = {s: {i: n for n, i in enumerate(data[s]["ids"])} for s in emus}
    cols: dict[str, list] = {}

    def put(key, arr):
        cols.setdefault(key, []).append(arr)

    for n in idx:
        iid = fp_ids[n]
        in_teil = bool(emus) and iid in pos[emus[0]]
        for cam in ("eig", "grad"):
            names = ["fp32"] + (chips if cam == "eig" else []) + emuvoll + (emus if in_teil else [])
            if len(names) == 1:
                continue
            rows = [n if not s.startswith("emu_") else pos[s][iid] for s in names]
            raw = np.stack([data[s][f"{cam}_raw"][r] for s, r in zip(names, rows)])
            empty = np.array([data[s]["grad_empty"][r] for s, r in zip(names, rows)]) if cam == "grad" else None
            x = unit_maps(raw, empty)
            mats = {f"iou{round(f * 100)}": pairwise_iou(x, f) for f in TOPK_FRACTIONS}
            mats["cc"] = pairwise_cc(x)
            if cam == "eig":
                flip = np.r_[False, pairwise_cc(x)[0, 1:] < 0]
                xa = np.where(flip[:, None], 1.0 - x, x)
                for f in TOPK_FRACTIONS:
                    mats[f"iou{round(f * 100)}_al"] = pairwise_iou(xa, f)
                mats["cc_al"] = pairwise_cc(xa)
            pa, pb = (np.array(v) for v in zip(*_pairs(tuple(names))))
            bad = np.zeros(len(pa), bool) if empty is None else (empty[pa] | empty[pb])
            put("cam", np.full(len(pa), cam))
            put("a", np.array(names)[pa])
            put("b", np.array(names)[pb])
            put("image_id", np.full(len(pa), iid))
            for key in _METRIC_KEYS:
                m = mats.get(key)
                put(key, np.full(len(pa), np.nan) if m is None else np.where(bad, np.nan, m[pa, pb]))
            if with_rank:
                rho, kld = spearman_kld(x[0], x[1:])
                sp, kl = np.full(len(pa), np.nan), np.full(len(pa), np.nan)
                eff = pa == 0
                sp[eff], kl[eff] = rho[pb[eff] - 1], kld[pb[eff] - 1]
                put("spearman", np.where(bad, np.nan, sp))
                put("kld", np.where(bad, np.nan, kl))
    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df.insert(0, "model", model)
    return df


_PAIRS: dict = {}


def _pairs(names: tuple[str, ...]) -> list[tuple[int, int]]:
    """Welche Paare gebraucht werden (Auswertungsplan Abschn. 4), Index 0 = FP32."""
    if names in _PAIRS:
        return _PAIRS[names]
    out = [(0, b) for b in range(1, len(names))]                 # Effekt: FP32 ↔ X
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(names[1:], start=1):
        groups.setdefault(s.split("_")[0], []).append(i)
    for g in groups.values():                                    # Compile-Streuung je Quellenart
        out += [(i, j) for n, i in enumerate(g) for j in g[n + 1:]]
    by_k: dict[str, list[int]] = {}
    for i, s in enumerate(names[1:], start=1):
        by_k.setdefault(s.split("_")[1], []).append(i)
    for members in by_k.values():                                # Emulator ↔ Chip desselben Compiles
        chip = [i for i in members if names[i].startswith("chip_")]
        out += [(chip[0], i) for i in members if chip and names[i].startswith("emu")]
    _PAIRS[names] = out
    return out


def vergleiche_table(cache: Path, model: str, n_jobs: int = 2, chunk: int = 1000,
                     with_rank: bool = True, log=print) -> pd.DataFrame:
    """Je Vergleich × Bild: Top-15/20-IoU, CC (+ ausgerichtet bei Eigen-CAM), Spearman/KLD für FP32 ↔ X."""
    n = len(load_source(cache, model, "fp32")["ids"])
    chunks = [np.arange(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    frames, t0 = [], time.time()
    if n_jobs <= 1:
        for c in chunks:
            frames.append(_compare_chunk(str(cache), model, c, with_rank))
    else:
        with ProcessPoolExecutor(n_jobs) as ex:
            futs = [ex.submit(_compare_chunk, str(cache), model, c, with_rank) for c in chunks]
            for done, f in enumerate(futs, 1):
                frames.append(f.result())
                if done % 5 == 0 or done == len(futs):
                    log(f"{model}: {done}/{len(futs)} Abschnitte ({time.time() - t0:.0f} s)")
    v = pd.concat(frames, ignore_index=True)
    for col in ("model", "cam", "a", "b"):
        v[col] = v[col].astype("category")
    return v


def aktivierung_table(cache: Path, model: str) -> pd.DataFrame:
    frames = []
    for p in sorted((Path(cache) / model).glob("akt_*.npz")):
        with np.load(p) as z:
            f = pd.DataFrame({k: z[k] for k in z.files})
        f.insert(0, "source", p.stem[4:])
        f.insert(0, "model", model)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def save_table(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def compute_tables(spec: KernSpec = KernSpec(), cache: Path = CACHE_DIR, n_jobs: int = 2,
                   with_rank: bool = True, log=print) -> dict[str, Path]:
    """Tabellen ``quellen``, ``vergleiche``, ``aktivierung`` je Modell schreiben (übersprungen, wenn vorhanden)."""
    out = {}
    for model in spec.models:
        for name, fn in (("quellen", lambda: quellen_table(cache, model, spec)),
                         ("aktivierung", lambda: aktivierung_table(cache, model)),
                         ("vergleiche", lambda: vergleiche_table(cache, model, n_jobs, with_rank=with_rank, log=log))):
            p = Path(cache) / "tabellen" / f"{name}_{model}.parquet"
            if not p.exists():
                save_table(fn(), p)
                log(f"geschrieben {p.name}")
            out[f"{name}_{model}"] = p
    return out


def provenance() -> dict:
    """Git-Stand und Plan-Tag für die erste Notebook-Zelle."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "?"
    # dirty zählt nur Code und Konfiguration: Notebooks ändern sich bei jedem Lauf (Ausgaben),
    # sie werden getrennt aufgeführt.
    code = git("status", "--porcelain", "--", ".", ":(exclude)notebooks")
    nbs = git("status", "--porcelain", "--", "notebooks")
    return {"commit": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(code), "dirty_files": [l.split(maxsplit=1)[1] for l in code.splitlines()],
            "notebooks_geändert": [l.split(maxsplit=1)[1] for l in nbs.splitlines()],
            "plan_tag": git("rev-parse", "--short", "auswertungsplan-v1^{}"),
            "zeit": time.strftime("%Y-%m-%d %H:%M")}
