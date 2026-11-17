#!/usr/bin/env bash
# Emulator-Läufe des Kerns (Grad-CAM, quantisierter Kontext) – IM DFC-CONTAINER auf dem PC:
#     bash scripts/emu_kern.sh subset      # Teil A: 5000er-Teilmenge, alle 20 Compiles
#     bash scripts/emu_kern.sh full        # Teil B: alle 48 000 Bilder, nur Compile k01 je Modell
#     bash scripts/emu_kern.sh subset resnet18   # optional: nur Configs mit diesem Namensteil
# Zielklasse = FP32-Top-1 (--target-run, Entscheidung 09.11.2026). Batch: MobileNetV3 4 (bitgleich
# zu 2, 16.11.2026), ResNet18 8. Blöcke, weil der Runner alle Eingaben als float32 im RAM hält
# (5000 Bilder ≈ 3 GB): Teilmenge in Blöcken à 2500, volle Liste à 3000 (teilt die FP32-Blöcke à 6000).
# Fertige Blöcke werden übersprungen → nach Abbruch neu starten.
# Voraussetzung: git pull (FP32-Run-Ordner mit predictions.csv), Bilder unter data/images/.
set -u
cd "$(dirname "$0")/.."
mode="${1:?subset|full}"
pattern="${2:-}"
summary="logs/emu_kern_${mode}_$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs

fp32_name() { case "$1" in resnet18*) echo resnet18_fp32_imagenet;; mobilenet*) echo mobilenet_v3_large_fp32_imagenet;; esac; }
batch() { case "$1" in mobilenet*) echo 4;; *) echo 8;; esac; }

run_block() {  # cfg name subset offset limit target_offset
  local cfg=$1 name=$2 subset=$3 off=$4 lim=$5 toff=$6 tsubset=$7
  if run=$(python3 scripts/find_run.py "$name" "$subset" "$off" --role emulator --context quantized); then
    echo "$(date +%T) SKIP $name $subset offset=$off ($run)" | tee -a "$summary"; return
  fi
  local fp; fp=$(fp32_name "$name")
  local target
  if ! target=$(python3 scripts/find_run.py "$fp" "$tsubset" "$toff" --role reference --need predictions.csv); then
    echo "$(date +%T) FAIL $name: FP32-Lauf $fp $tsubset offset=$toff fehlt (git pull?)" | tee -a "$summary"; return
  fi
  echo "$(date +%T) START $name $subset offset=$off (Ziel: $target)" | tee -a "$summary"
  local start; start=$(date +%s)
  if thesis-emulator-run "$cfg" --subset "$subset" --offset "$off" --limit "$lim" \
       --contexts quantized --batch-size "$(batch "$name")" --target-run "$target"; then
    echo "$(date +%T) OK   $name offset=$off ($(( $(date +%s) - start ))s)" | tee -a "$summary"
  else
    echo "$(date +%T) FAIL $name offset=$off" | tee -a "$summary"
  fi
}

for cfg in configs/kern/*_k[0-9][0-9].yaml; do
  [[ -n "$pattern" && "$cfg" != *"$pattern"* ]] && continue
  name=$(basename "$cfg" .yaml)
  if [[ "$mode" == subset ]]; then
    for off in 0 2500; do
      run_block "$cfg" "$name" imagenet_eval_strat5000_seed2.txt "$off" 2500 0 imagenet_eval_strat5000_seed2.txt
    done
  elif [[ "$mode" == full ]]; then
    [[ "$name" != *_k01 ]] && continue
    for ((off=0; off<48000; off+=3000)); do
      run_block "$cfg" "$name" imagenet_eval_48000.txt "$off" 3000 $(( off / 6000 * 6000 )) imagenet_eval_48000.txt
    done
  else
    echo "Modus subset|full"; exit 2
  fi
done
echo "$(date +%T) fertig → $summary" | tee -a "$summary"
