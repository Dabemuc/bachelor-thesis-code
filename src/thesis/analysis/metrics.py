"""Heatmap-Vergleichsmetriken (reines NumPy).

Übernommen aus dem Machbarkeits-Prototyp ``grad-cam-quant-feasibility``
(gcqf/metrics.py), damit Simulation und Hardware mit identischem Code
ausgewertet werden.

* ``cc``        – Pearson-Korrelation (1 = gleicher Fokus)
* ``kld``       – KL-Divergenz, Heatmaps als Verteilungen (0 = gleich)
* ``topk_iou``  – IoU der salientesten Pixel (Überlappung der Fokusregion)
* ``spearman``  – Rangkorrelation; ≈ 0 markiert kollabierte Heatmaps

Konvention: erstes Argument = FP32-Referenz, zweites = quantisierte Karte.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

TOPK_FRACTION = 0.15
COLLAPSE_SPEARMAN = 0.2
COLLAPSE_CC = 0.0
COLLAPSE_IOU_TOL = 1.05

_EPS = 1e-8


@dataclass
class HeatmapMetrics:
    cc: float
    kld: float
    topk_iou: float
    spearman: float
    collapsed: bool

    def as_row(self) -> dict:
        return asdict(self)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < _EPS:
        return 0.0
    return float((x * y).sum() / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Ränge mit Mittelwert bei Gleichstand (wie scipy.stats.rankdata 'average')."""
    a = np.asarray(a, dtype=np.float64).ravel()
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    sorted_a = a[sorter]
    obs = np.r_[True, sorted_a[1:] != sorted_a[:-1]]
    dense = obs.cumsum()[inv]
    counts = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (counts[dense] + counts[dense - 1] + 1)


def pearson_cc(ref: np.ndarray, q: np.ndarray) -> float:
    return _pearson(ref, q)


def spearman(ref: np.ndarray, q: np.ndarray) -> float:
    return _pearson(_rankdata(ref), _rankdata(q))


def kl_divergence(ref: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(ref.ravel().astype(np.float64), 0, None) + _EPS
    r = np.clip(q.ravel().astype(np.float64), 0, None) + _EPS
    p /= p.sum()
    r /= r.sum()
    return float(np.sum(p * np.log(p / r)))


def topk_mask(heatmap: np.ndarray, frac: float = TOPK_FRACTION) -> np.ndarray:
    a = heatmap.ravel()
    k = max(1, int(round(frac * a.size)))
    threshold = np.partition(a, -k)[-k]
    return heatmap >= threshold


def topk_iou(ref: np.ndarray, q: np.ndarray, frac: float = TOPK_FRACTION) -> float:
    a, b = topk_mask(ref, frac), topk_mask(q, frac)
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def align_to_reference(reference: np.ndarray, cam: np.ndarray) -> tuple[np.ndarray, bool]:
    """Vorzeichen der Eigen-CAM-Hauptkomponente an der Referenz ausrichten.

    Für eine auf [0, 1] normierte Karte ist die vorzeichen-gespiegelte Projektion
    genau ``1 - cam``. Rückgabe: (ausgerichtete Karte, wurde gespiegelt).
    """
    if _pearson(reference, cam) < 0:
        return 1.0 - cam, True
    return cam, False


def is_collapsed(heatmap: np.ndarray, std_threshold: float = 1e-4) -> bool:
    return bool(np.std(heatmap) < std_threshold)


def is_broken(cc: float, iou: float, rho: float, q: np.ndarray) -> bool:
    return bool(
        is_collapsed(q)
        or rho < COLLAPSE_SPEARMAN
        or cc <= COLLAPSE_CC
        or iou <= TOPK_FRACTION * COLLAPSE_IOU_TOL
    )


def compare(ref: np.ndarray, q: np.ndarray) -> HeatmapMetrics:
    cc = pearson_cc(ref, q)
    iou = topk_iou(ref, q)
    rho = spearman(ref, q)
    return HeatmapMetrics(cc=cc, kld=kl_divergence(ref, q), topk_iou=iou, spearman=rho,
                          collapsed=is_broken(cc, iou, rho, q))
