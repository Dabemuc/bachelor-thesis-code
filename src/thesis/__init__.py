"""Code zur Bachelorarbeit – Quantisierung (Hailo-8) und CAM-Erklärungen.

Aufteilung nach Maschine (siehe README):

* ``thesis.common``    – überall: Konfiguration, Preprocessing, Subset, Run-Metadaten
* ``thesis.analysis``  – überall (nur NumPy): CAM-Metriken, Eigen-CAM-Projektion
* ``thesis.edge``      – Raspberry Pi 5 + AI HAT+: HailoRT-Inferenz, Telemetrie
* ``thesis.compile``   – x86-Desktop im DFC-Container: ONNX → HAR → HEF
* ``thesis.reference`` – DGX Spark: PyTorch-FP32-Referenz, ONNX-Export
"""

__version__ = "0.1.0"
