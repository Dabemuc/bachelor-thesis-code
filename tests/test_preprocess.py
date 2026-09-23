import numpy as np
from PIL import Image

from thesis.common.preprocess import load_uint8, normalize, onchip_normalization_params
from thesis.common.subset import SubsetItem, read_subset, write_subset


def test_load_shape_and_dtype(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 500, 3), dtype=np.uint8)).save(p)
    img = load_uint8(p)
    assert img.shape == (224, 224, 3) and img.dtype == np.uint8


def test_onchip_normalization_matches_host():
    """HEF-Normalisierung (0–255-Skala) muss dasselbe ergeben wie der Host-Pfad."""
    img = np.random.default_rng(1).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    mean, std = onchip_normalization_params()
    onchip = (img.astype(np.float32) - np.array(mean, np.float32)) / np.array(std, np.float32)
    assert np.allclose(onchip, normalize(img), atol=1e-5)


def test_subset_roundtrip(tmp_path):
    items = [SubsetItem("a/1.jpg", 3), SubsetItem("b/2.jpg", 7)]
    p = tmp_path / "s.txt"
    write_subset(p, items, header="seed=0")
    assert read_subset(p) == items
