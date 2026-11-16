#!/usr/bin/env bash
# FP32-Referenzen für den Kern – auf dem Mac (CPU, wie alle bisherigen FP32-Referenzen):
#     bash scripts/fp32_kern.sh
# 1) 5000er-Teilmenge mit Gradienten (Zielklassen + FP32-Grad-CAM für die Emulator-Läufe auf 5000)
# 2) alle 48 000 Bilder mit Features + Gradienten in Blöcken à 6000 (Speicher: alle Tensoren
#    landen bis zum Ende im RAM; 6000 MobileNetV3-Bilder ≈ 2,3 GB Features + Gradienten)
# Ausgabe: je Lauf ein Ordner in results/; Blöcke tragen ihren offset in meta.json.
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-python}
for cfg in configs/resnet18_fp32_imagenet.yaml configs/mobilenet_v3_large_fp32_imagenet.yaml; do
  $PY -m thesis.reference.fp32_run "$cfg" --subset imagenet_eval_strat5000_seed2.txt --gradients
  for off in 0 6000 12000 18000 24000 30000 36000 42000; do
    $PY -m thesis.reference.fp32_run "$cfg" --offset "$off" --limit 6000 --gradients
  done
done
