"""Experiment-Konfiguration.

Eine YAML-Datei in ``configs/`` beschreibt genau ein Experiment
(Modell × Quantisierungsstufe × Subset). Alle Maschinen lesen dieselbe Datei –
das ist die Garantie, dass FP32-Referenz, Compile und Edge-Lauf dasselbe meinen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

# ImageNet-Preprocessing (torchvision-Standard)
IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ExperimentConfig:
    name: str
    model: str                       # z. B. "resnet18"
    precision: str                   # "fp32" | "int8" | "a8w4" …
    subset: str                      # Pfad relativ zu data/subsets/
    onnx: str | None = None          # Pfad relativ zu artifacts/
    hef: str | None = None           # Pfad relativ zu artifacts/
    # ONNX-Knoten, die im HEF als Ausgänge erscheinen sollen (Logits + CAM-Layer)
    end_node_names: list[str] = field(default_factory=list)
    # True: Normalisierung (mean/std) wird per Model Script ins HEF gelegt,
    # der Host füttert uint8 (0–255). False: Host normalisiert, HEF bekommt float.
    normalize_on_chip: bool = True
    calib_size: int = 1024
    warmup_iters: int = 50
    repeats: int = 1                 # Wiederholungen des Subset-Durchlaufs (Hailo ist nicht deterministisch)
    save_features: bool = True       # Feature-Maps der CAM-Schicht mitschreiben
    notes: str = ""

    @property
    def subset_path(self) -> Path:
        return DATA_DIR / "subsets" / self.subset

    @property
    def onnx_path(self) -> Path | None:
        return ARTIFACTS_DIR / self.onnx if self.onnx else None

    @property
    def hef_path(self) -> Path | None:
        return ARTIFACTS_DIR / self.hef if self.hef else None


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(**raw)
