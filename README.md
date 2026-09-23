# thesis-code

Code zur Bachelorarbeit: Wie verändert **hardware-reale Quantisierung auf dem Hailo-8**
(Raspberry Pi 5 + AI HAT+ 26 TOPS) die CAM-Erklärungen eines Bildklassifikators?

Nachfolger des Machbarkeits-Prototyps [`grad-cam-quant-feasibility`](../grad-cam-quant-feasibility)
(Simulation mit Fake-Quantisierung). Die Metriken sind von dort übernommen, damit
Simulation und Hardware mit identischem Code ausgewertet werden.

## Drei Maschinen, ein Repo

| Maschine | Rolle | Paketteil | Install |
|---|---|---|---|
| **x86-Desktop** (Docker) | ONNX → Quantisierung → HEF, Emulator, Layer-Analyse | `thesis.compile` | `docker/dfc.Dockerfile`, dann `pip install -e ".[compile]"` |
| **DGX Spark** (aarch64) | FP32-Referenz, Grad-CAM, ONNX-Export | `thesis.reference` | `uv sync --extra reference` |
| **Raspberry Pi 5 + AI HAT+** | Inferenz auf dem Hailo-8, Latenz, Temperatur | `thesis.edge` | siehe unten |
| überall | Preprocessing, Subset, Run-Metadaten, Metriken | `thesis.common`, `thesis.analysis` | Basisinstallation |

**Merksatz: DFC = x86, HailoRT = überall.** Der Dataflow Compiler läuft nicht auf ARM,
also weder auf der Spark noch auf dem Pi.

```
x86 (DFC) ── .hef ──► Pi (HailoRT) ── results/ ──┐
Spark (PyTorch FP32) ─────────── results/ ───────┴─► thesis.analysis
```

## Regeln, die die Messung gültig halten

1. **Ein Preprocessing** (`thesis/common/preprocess.py`) für alle Maschinen, auch für die
   FP32-Referenz. Kein `torchvision.transforms`.
2. **Ein Subset** pro Experiment (`data/subsets/*.txt`), einmal erzeugt, nie verändert.
   Kalibrierdaten sind **disjunkt** zum Evaluations-Subset.
3. **Code per `git pull`, nicht per rsync** auf den Pi. Jeder Lauf schreibt den Commit
   und den dirty-Status in `meta.json`.
4. **Der Pi schreibt nur Rohdaten**, die Auswertung läuft woanders (sonst CPU-Last →
   verfälschte Latenz).
5. **HEF-Ausgänge werden beim Kompilieren festgelegt** (`end_node_names`). Für Eigen-CAM auf
   der Hardware braucht es zusätzlich zu den Logits die Feature-Map der letzten Conv-Stufe.
6. **Versionen passend wählen:** Der DFC muss zur HailoRT-Version auf dem Pi passen
   (für Hailo-8 die 3.x-Linie des DFC).

## Pi einrichten

```bash
# auf dem Pi
git clone <repo> ~/thesis-code && cd ~/thesis-code
scripts/pi_setup.sh            # apt install hailo-all → reboot
scripts/pi_setup.sh verify     # muss HAILO8 melden (nicht HAILO8L)

# hailo_platform kommt per apt → venv muss System-Pakete sehen
uv venv --system-site-packages && uv pip install -e .
# (ohne uv: python3 -m venv --system-site-packages .venv && .venv/bin/pip install -e .)
```

Test ohne eigenen Compile-Schritt: vorkompiliertes `resnet_v1_18.hef` (hailo8) aus dem
Hailo Model Zoo laden und `hailortcli run resnet_v1_18.hef --measure-temp` ausführen.

## AP0: einmal alles durchlaufen lassen

```bash
# Spark / beliebig mit torch
thesis-export-onnx resnet18
thesis-onnx-nodes artifacts/onnx/resnet18.onnx --grep layer4   # Namen → configs/resnet18_int8.yaml
python -m thesis.reference.fp32_run configs/resnet18_fp32.yaml --device cuda

# x86-Desktop, im DFC-Container
thesis-compile configs/resnet18_int8.yaml

# Laptop → Pi
scripts/sync_to_pi.sh push

# Pi (in tmux)
scripts/pi_bench_mode.sh
.venv/bin/thesis-edge-run configs/resnet18_int8.yaml --limit 20   # Smoke-Test
.venv/bin/thesis-edge-run configs/resnet18_int8.yaml

# zurück + vergleichen
scripts/sync_to_pi.sh pull
python -m thesis.analysis.compare_runs results/<fp32-run> results/<edge-run>
```

AP0 ist abgeschlossen, wenn Top-1-Match und Logit-Cosinus plausibel sind und die
Eigen-CAM-Metriken aus echten Hailo-Feature-Maps berechnet werden.

## Ausgabeformat (alle Runs)

```
results/<zeit>_<host>_<experiment>/
  meta.json          Commit, dirty-Flag, Versionen (inkl. HailoRT per dpkg), Config
  predictions.csv    image_id, label, top1, top5[, latency_ms, repeat]
  telemetry.csv      nur Edge: SoC-Temp, Throttle-Flags, Hailo-Temp (~1 Hz)
  logits.npz         ids [N], logits [N, 1000]
  features.npz       ids [N], features [N, H, W, K]   (HWC wie HailoRT)
```

## Offen / TODO

- [ ] `end_node_names` für ResNet18 aus dem echten ONNX eintragen
- [ ] prüfen, ob ein Zwischenknoten zusätzlich als HEF-Ausgang zulässig ist
- [ ] DFC-Version passend zur HailoRT-Version auf dem Pi festlegen
- [ ] Datensatz und Subsets festlegen (`data/subsets/README.md`)
- [ ] Grad-CAM-Modul aus dem Prototyp portieren (`thesis.reference`)
- [ ] Quantisierungsstufen über das Model Script (INT8, 4-Bit-Gewichte, Mixed Mode)
- [ ] Energiemessung (USB-C-Messgerät vs. `vcgencmd pmic_read_adc`)

## Tests

```bash
uv run --extra dev pytest
```
