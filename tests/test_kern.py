"""Tests für die Kern-Auswertung: Stapel-Metriken = Einzelmetriken, Pipeline auf Spielzeugdaten."""

import json

import numpy as np
import pytest

from thesis.analysis import kern
from thesis.analysis.eigencam import upsample_normalize
from thesis.analysis.maps import pairwise_cc, pairwise_iou, unit_maps, upsampled_range
from thesis.analysis.metrics import compare, topk_iou


def test_batched_metrics_match_scalar():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(5, 7, 7))
    raw[3] = 0.4                      # konstante Karte
    x = unit_maps(raw)
    iou, cc = pairwise_iou(x, 0.15), pairwise_cc(x)
    for a in range(5):
        for b in range(5):
            m = compare(x[a].reshape(224, 224), x[b].reshape(224, 224))
            assert np.isclose(iou[a, b], m.topk_iou)
            assert np.isclose(cc[a, b], m.cc)
            assert np.isclose(pairwise_iou(x, 0.2)[a, b], topk_iou(x[a], x[b], 0.2))


def test_unit_maps_equal_upsample_normalize_and_range_trick():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(3, 7, 7)).astype(np.float32)
    x = unit_maps(raw)
    lo, hi = upsampled_range(raw)
    for i in range(3):
        assert np.allclose(x[i], upsample_normalize(raw[i]).ravel())
        # normierte Karte = Hochskalieren der vorab normierten 7×7-Karte (für Konsens-Karten)
        r = (raw[i] - lo[i]) / (hi[i] - lo[i])
        assert np.allclose(_up(r), x[i], atol=1e-5)


def _up(r):
    from PIL import Image
    return np.asarray(Image.fromarray(r.astype(np.float32), mode="F").resize((224, 224), Image.BILINEAR),
                      dtype=np.float64).ravel()


def _write_run(root, name, *, role, model, cfg_name, subset, offset, ids, logits, feats, grads=None,
               target=None, hef="h", context=None, target_run=None):
    d = root / name
    d.mkdir()
    meta = {"started": "2026-11-17T00:00:00", "role": role, "offset": offset, "n_images": len(ids),
            "git": {"commit": "abcdef1234", "dirty": False},
            "config": {"name": cfg_name, "model": model, "subset": subset}}
    if role != "reference":
        meta["hef"] = {"sha256": hef}
    if context:
        meta["emulator_context"] = context
    if target_run:
        meta["target_run"] = f"results/{target_run}"
    (d / "meta.json").write_text(json.dumps(meta))
    np.savez(d / "logits.npz", ids=ids, logits=logits)
    np.savez(d / "features.npz", ids=ids, features=feats)
    if grads is not None:
        np.savez(d / "gradients.npz", ids=ids, target=target, gradients=grads)
    if role == "emulator":
        np.savez(d / "features_keras.npz", ids=ids, features=feats)
        np.savez(d / "consistency.npz", ids=ids, max_abs_diff_logits=np.zeros(len(ids), np.float32),
                 max_abs_diff_features=np.zeros(len(ids), np.float32))


@pytest.fixture
def toy(tmp_path):
    rng = np.random.default_rng(5)
    nv, nt, c = 12, 4, 10
    ids = np.array([f"img/{i:03d}.JPEG" for i in range(nv)])
    labels = rng.integers(0, c, nv)
    sub = tmp_path / "subsets"
    sub.mkdir()
    (sub / "voll.txt").write_text("# toy\n" + "".join(f"{i}\t{l}\n" for i, l in zip(ids, labels)))
    teil_idx = np.array([1, 4, 7, 10])
    (sub / "teil.txt").write_text("".join(f"{ids[i]}\t{labels[i]}\n" for i in teil_idx))
    res = tmp_path / "results"
    res.mkdir()
    fl = rng.normal(size=(nv, c)).astype(np.float32)
    ff = rng.normal(size=(nv, 7, 7, 8)).astype(np.float32)
    fg = rng.normal(size=(nv, 7, 7, 8)).astype(np.float32)
    t = fl.argmax(1)
    for b in range(2):
        s = slice(6 * b, 6 * b + 6)
        _write_run(res, f"r_fp32_{b}", role="reference", model="m1", cfg_name="m1_fp32_imagenet",
                   subset="voll.txt", offset=6 * b, ids=ids[s], logits=fl[s], feats=ff[s], grads=fg[s], target=t[s])
    _write_run(res, "r_fp32_teil", role="reference", model="m1", cfg_name="m1_fp32_imagenet", subset="teil.txt",
               offset=0, ids=ids[teil_idx], logits=fl[teil_idx], feats=ff[teil_idx], grads=fg[teil_idx],
               target=t[teil_idx])
    for k in (1, 2):
        feats = ff if k == 1 else ff + rng.normal(0, 0.5, ff.shape).astype(np.float32)   # k01 = FP32 exakt
        lq = fl + (0 if k == 1 else rng.normal(0, 0.1, fl.shape).astype(np.float32))
        for b in range(2):
            s = slice(6 * b, 6 * b + 6)
            _write_run(res, f"r_chip_k{k}_{b}", role="edge", model="m1", cfg_name=f"m1_int8_lvl2_k{k:02d}",
                       subset="voll.txt", offset=6 * b, ids=ids[s], logits=lq[s], feats=feats[s], hef=f"h{k}")
        _write_run(res, f"r_emu_k{k}", role="emulator", model="m1", cfg_name=f"m1_int8_lvl2_k{k:02d}",
                   subset="teil.txt", offset=0, ids=ids[teil_idx], logits=lq[teil_idx], feats=feats[teil_idx],
                   grads=fg[teil_idx], target=t[teil_idx], hef=f"h{k}", context="sdk_quantized",
                   target_run="r_fp32_teil")
    for b in range(3):
        s = slice(4 * b, 4 * b + 4)
        _write_run(res, f"r_emuvoll_{b}", role="emulator", model="m1", cfg_name="m1_int8_lvl2_k01",
                   subset="voll.txt", offset=4 * b, ids=ids[s], logits=fl[s], feats=ff[s], grads=fg[s],
                   target=t[s], hef="h1", context="sdk_quantized", target_run="r_fp32_0")
    spec = kern.KernSpec(results_dir=res, list_voll=sub / "voll.txt", list_teil=sub / "teil.txt",
                         models=("m1",), k=2, started_after="2026-11-16")
    return spec, tmp_path / "cache"


def test_pipeline_on_toy_data(toy):
    spec, cache = toy
    inv = kern.inventory(spec)
    assert len(inv) == 2 + 1 + 4 + 2 + 3
    checks = kern.check_inventory(inv, spec)
    assert checks.ok.all(), checks[~checks.ok]
    kern.extract_all(inv, spec, cache, n_jobs=1, log=lambda *_: None)
    q = kern.quellen_table(cache, "m1", spec)
    assert set(q.source) == {"fp32", "chip_k01", "chip_k02", "emu_k01", "emu_k02", "emuvoll_k01"}
    assert len(q) == 12 * 4 + 4 * 2
    assert q.grad_target_ok.dropna().all()
    v = kern.vergleiche_table(cache, "m1", n_jobs=1, chunk=5, log=lambda *_: None)
    eig = v[v.cam == "eig"]
    same = eig[(eig.a == "fp32") & (eig.b == "chip_k01")]
    assert len(same) == 12 and np.allclose(same.iou15, 1.0) and np.allclose(same.cc, 1.0)
    # Paare je Bild: Eigen-CAM 2 Chip + 1 voll (+ 2 Emu auf der Teilmenge) gegen FP32,
    # Chip-Paar, Emu-Paar, Emu ↔ Chip je Compile
    per_img = eig.groupby("image_id").size()
    assert (per_img[~per_img.index.isin(q[q.in_teil].image_id)] == 3 + 1 + 1).all()
    assert (per_img[per_img.index.isin(q[q.in_teil].image_id)] == 5 + 2 + 3).all()
    assert v.spearman.notna().sum() == (v.a == "fp32").sum() - v[(v.a == "fp32")].iou15.isna().sum()
    a = kern.aktivierung_table(cache, "m1")
    k1 = a[a.source == "emu_k01"]
    assert len(k1) == 4 and np.allclose(k1.frac_equal, 1.0) and np.allclose(k1.rel_err, 0.0)
    assert len(a[a.source == "emuvoll_k01"]) == 12
