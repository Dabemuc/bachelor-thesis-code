"""Grad-CAM aus gespeicherten Feature-Maps und Gradienten (reines NumPy).

Eingabe sind ``features.npz`` und ``gradients.npz`` eines Laufs (Layout [H, W, K], wie
HailoRT), egal ob sie aus PyTorch (``thesis.reference.fp32_run --gradients``) oder aus dem
Hailo-Emulator (``thesis-emulator-run``) stammen. So rechnen beide Seiten mit exakt
demselben Code.

Grad-CAM (Selvaraju et al. 2017, Gl. 1–2):
    alpha_k = mittlerer Gradient d y_c / d A_k über alle Positionen (global average pooling)
    L_c     = ReLU( sum_k alpha_k * A_k )
danach wie Eigen-CAM bilinear auf Eingabegröße skaliert und auf [0, 1] normiert.

Ist ``sum_k alpha_k * A_k`` überall ≤ 0, ist die Karte nach der ReLU leer. Das passiert u. a.,
wenn der Gradient 0 ist (z. B. gesättigtes Logit im quantisierten Modell, STE schneidet ab →
Vault: „E2 Phase 1 Voraussetzungen“, 03.11.2026). Solche Karten werden als ``empty`` markiert
und nicht in [0, 1] normiert.
"""

from __future__ import annotations

import numpy as np

from ..common.config import IMAGE_SIZE
from .eigencam import upsample_normalize


def gradcam_weights(grads_hwk: np.ndarray) -> np.ndarray:
    """[H, W, K] → [K]: räumliches Mittel der Gradienten (alpha_k)."""
    return grads_hwk.astype(np.float64).mean(axis=(0, 1))


def gradcam_raw(features_hwk: np.ndarray, grads_hwk: np.ndarray) -> np.ndarray:
    """[H, W] vor dem Skalieren: ReLU(sum_k alpha_k * A_k)."""
    cam = features_hwk.astype(np.float64) @ gradcam_weights(grads_hwk)
    return np.maximum(cam, 0.0)


def gradcam(features_hwk: np.ndarray, grads_hwk: np.ndarray, size: int = IMAGE_SIZE) -> tuple[np.ndarray, bool]:
    """Grad-CAM auf Eingabegröße. Rückgabe: (Karte in [0, 1], leer?)."""
    raw = gradcam_raw(features_hwk, grads_hwk)
    if raw.max() <= 0.0:
        return np.zeros((size, size)), True
    return upsample_normalize(raw, size), False
