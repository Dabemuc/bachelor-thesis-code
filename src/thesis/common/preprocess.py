"""Das EINE Preprocessing für alle Maschinen.

Wichtig für die Validität: Unterschiede zwischen FP32- und Hailo-Heatmaps
dürfen nur aus der Quantisierung kommen, nicht aus abweichendem Resize/Crop.
Deshalb gibt es hier genau eine Implementierung – ohne torchvision, damit sie
auch auf dem Pi und im DFC-Container ohne PyTorch läuft.

Pipeline (entspricht torchvision ``Resize(256) → CenterCrop(224)``):
    1. kürzere Seite auf 256 skalieren (bilinear, PIL)
    2. zentral 224×224 ausschneiden
    3. → uint8 HWC (RGB)
    4. optional: /255, (x - mean) / std  → float32 HWC

⚠️ PIL-Bilinear und torchvision-Bilinear auf Tensoren sind nicht bitgleich.
Die FP32-Referenz muss deshalb ebenfalls ``load_uint8`` benutzen und erst
danach in einen Tensor wandeln – nicht torchvision.transforms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SIZE


def load_uint8(path: str | Path, resize: int = RESIZE_SIZE, crop: int = IMAGE_SIZE) -> np.ndarray:
    """Bild laden → uint8-Array [crop, crop, 3] (RGB, HWC)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        # Größenberechnung wie torchvision.Resize(int): lange Seite wird abgeschnitten (int)
        if w <= h:
            new_w, new_h = resize, int(resize * h / w)
        else:
            new_w, new_h = int(resize * w / h), resize
        im = im.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - crop) // 2
        top = (new_h - crop) // 2
        im = im.crop((left, top, left + crop, top + crop))
        return np.asarray(im, dtype=np.uint8).copy()


def normalize(img_uint8: np.ndarray) -> np.ndarray:
    """uint8 HWC → float32 HWC, ImageNet-normalisiert."""
    x = img_uint8.astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    return (x - mean) / std


def onchip_normalization_params() -> tuple[list[float], list[float]]:
    """mean/std in 0–255-Skala für das DFC-Model-Script (``normalization``-Befehl)."""
    return [m * 255.0 for m in IMAGENET_MEAN], [s * 255.0 for s in IMAGENET_STD]


def to_nchw_batch(img_hwc: np.ndarray) -> np.ndarray:
    """HWC → [1, C, H, W] für PyTorch/ONNX."""
    return np.ascontiguousarray(img_hwc.transpose(2, 0, 1)[None])
