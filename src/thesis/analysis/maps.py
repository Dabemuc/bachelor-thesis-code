"""Kartenoperationen für die Kern-Auswertung, stapelweise je Bild (reines NumPy + PIL).

Alle Karten eines Bildes (FP32 und die quantisierten Quellen) werden als Stapel [S, P]
verarbeitet, damit alle Paarvergleiche eines Bildes mit zwei Matrixprodukten fallen.
Die Definitionen sind dieselben wie in ``metrics.py`` (Top-k-Maske mit ``>=`` an der
k-größten Stelle, Pearson mit 0 bei konstanter Karte); ``tests/test_maps.py`` prüft das.
"""

from __future__ import annotations

import numpy as np

from ..common.config import IMAGE_SIZE
from .eigencam import upsample_normalize
from .metrics import kl_divergence, spearman

TOPK_FRACTIONS = (0.15, 0.20)   # Hauptmaß und Robustheit (Auswertungsplan 3.1)
_EPS = 1e-8


def unit_maps(raw7: np.ndarray, empty: np.ndarray | None = None, size: int = IMAGE_SIZE) -> np.ndarray:
    """[S, h, w] Rohkarten → [S, size*size] wie ``upsample_normalize`` (leere Karten = 0)."""
    out = np.zeros((raw7.shape[0], size * size), dtype=np.float64)
    for i, r in enumerate(raw7):
        if empty is not None and empty[i]:
            continue
        out[i] = upsample_normalize(r, size).ravel()
    return out


def upsampled_range(raw7: np.ndarray, size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Minimum und Maximum der hochskalierten (noch nicht normierten) Karten.

    Bilineares Hochskalieren ist linear und erhält Konstanten. Mit diesen beiden Werten
    ist die normierte 224er-Karte = Hochskalieren von (raw − min) / (max − min). So lassen
    sich Konsens-Karten (Mittel normierter Karten) exakt aus den 7×7-Karten bilden.
    """
    from PIL import Image

    lo = np.empty(raw7.shape[0], dtype=np.float64)
    hi = np.empty(raw7.shape[0], dtype=np.float64)
    for i, r in enumerate(raw7):
        im = Image.fromarray(r.astype(np.float32), mode="F").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float64)
        lo[i], hi[i] = a.min(), a.max()
    return lo, hi


def topk_masks(x: np.ndarray, frac: float) -> np.ndarray:
    """[S, P] → bool [S, P]; wie ``metrics.topk_mask`` (Gleichstände an der Schwelle zählen mit)."""
    k = max(1, int(round(frac * x.shape[1])))
    thr = np.partition(x, -k, axis=1)[:, -k]
    return x >= thr[:, None]


def pairwise_iou(x: np.ndarray, frac: float) -> np.ndarray:
    """[S, P] → [S, S] Top-k-IoU aller Paare."""
    m = topk_masks(x, frac).astype(np.float32)
    inter = m @ m.T
    size = np.diag(inter)
    union = size[:, None] + size[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float64)


def pairwise_cc(x: np.ndarray) -> np.ndarray:
    """[S, P] → [S, S] Pearson aller Paare (0, wenn eine Karte konstant ist)."""
    xc = x - x.mean(axis=1, keepdims=True)
    ss = (xc * xc).sum(axis=1)
    num = xc @ xc.T
    den = np.sqrt(ss[:, None] * ss[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den < _EPS, 0.0, num / den)


def is_constant(x: np.ndarray, std_threshold: float = 1e-4) -> np.ndarray:
    """[S, P] → bool [S]; wie ``metrics.is_collapsed``."""
    return x.std(axis=1) < std_threshold


def spearman_kld(ref: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spearman und KLD einer Referenzkarte [P] gegen mehrere Karten [S, P]."""
    rho = np.array([spearman(ref, o) for o in others])
    kld = np.array([kl_divergence(ref, o) for o in others])
    return rho, kld
