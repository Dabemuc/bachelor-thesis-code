"""Eigen-CAM aus gespeicherten Feature-Maps (reines NumPy).

Gleiche Projektion wie im Prototyp (gcqf/eigencam.py), aber ohne PyTorch-Hook:
Eingabe sind die Feature-Maps, die der Pi als zweiten HEF-Ausgang mitschreibt
(bzw. die FP32-Referenz auf der Spark). So läuft die Auswertung für beide Seiten
mit exakt demselben Code.

Layout-Hinweis: HailoRT liefert Feature-Maps NHWC, PyTorch NCHW.
``eigencam`` erwartet [H, W, K] – PyTorch-Karten vorher transponieren.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..common.config import IMAGE_SIZE

_EPS = 1e-8


def first_pc_projection(features_hwk: np.ndarray) -> np.ndarray:
    """[H, W, K] → [H, W]: Projektion auf die erste Hauptkomponente."""
    h, w, k = features_hwk.shape
    x = features_hwk.reshape(h * w, k).astype(np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    proj = xc @ vt[0]
    # deterministische Orientierung: Region hoher Aktivierungsenergie = heiß
    energy = (x ** 2).sum(axis=1)
    if proj.std() > 0 and energy.std() > 0 and np.corrcoef(proj, energy)[0, 1] < 0:
        proj = -proj
    return proj.reshape(h, w)


def upsample_normalize(cam2d: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Bilinear auf Eingabegröße skalieren und auf [0, 1] normieren."""
    im = Image.fromarray(cam2d.astype(np.float32), mode="F").resize((size, size), Image.BILINEAR)
    cam = np.asarray(im, dtype=np.float64)
    cam -= cam.min()
    return cam / (cam.max() + _EPS)


def eigencam(features_hwk: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    return upsample_normalize(first_pc_projection(features_hwk), size)
