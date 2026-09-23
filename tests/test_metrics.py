import numpy as np

from thesis.analysis.eigencam import eigencam, first_pc_projection
from thesis.analysis.metrics import align_to_reference, compare


def test_identical_maps_are_perfect():
    rng = np.random.default_rng(0)
    cam = rng.random((224, 224))
    m = compare(cam, cam)
    assert np.isclose(m.cc, 1.0)
    assert np.isclose(m.spearman, 1.0)
    assert np.isclose(m.topk_iou, 1.0)
    assert m.kld < 1e-9
    assert not m.collapsed


def test_constant_map_is_collapsed():
    rng = np.random.default_rng(1)
    assert compare(rng.random((32, 32)), np.full((32, 32), 0.5)).collapsed


def test_sign_alignment():
    rng = np.random.default_rng(2)
    cam = rng.random((16, 16))
    aligned, flipped = align_to_reference(cam, 1.0 - cam)
    assert flipped and np.allclose(aligned, cam)


def test_eigencam_finds_hot_region():
    # Feature-Map mit einem „Objekt" oben links: dort sollen die Werte hoch sein
    feats = np.random.default_rng(3).normal(0, 0.05, (7, 7, 64))
    feats[:3, :3, :] += 1.0
    proj = first_pc_projection(feats)
    assert proj[:3, :3].mean() > proj[4:, 4:].mean()
    cam = eigencam(feats)
    assert cam.shape == (224, 224) and 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-9
