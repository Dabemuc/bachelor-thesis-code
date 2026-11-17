#!/usr/bin/env bash
# Chip-Läufe des Kerns – AUF DEM PI ausführen (in tmux):
#     bash scripts/chip_kern.sh             # alle 20 Kern-Configs
#     bash scripts/chip_kern.sh resnet18    # nur Configs mit diesem Namensteil
# Je Config alle 48 000 Eval-Bilder in Blöcken à BLOCK (Default 8000 → 6 Blöcke), weil der Runner
# alle Feature-Maps bis zum Ende im RAM hält (MobileNetV3: 8000 Bilder ≈ 1,5 GB).
# Fertige Blöcke werden übersprungen (scripts/find_run.py) → nach Abbruch einfach neu starten.
# Voraussetzung: HEFs unter artifacts/<name>/ und alle 48 000 Bilder unter data/images/.
set -u
cd "$(dirname "$0")/.."
pattern="${1:-}"
BLOCK=${BLOCK:-8000}
TOTAL=48000
SUBSET=imagenet_eval_48000.txt
summary="logs/chip_kern_$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs
bash scripts/pi_bench_mode.sh | tee -a "$summary"
for cfg in configs/kern/*_k[0-9][0-9].yaml; do
  [[ -n "$pattern" && "$cfg" != *"$pattern"* ]] && continue
  name=$(basename "$cfg" .yaml)
  for ((off=0; off<TOTAL; off+=BLOCK)); do
    if run=$(python3 scripts/find_run.py "$name" "$SUBSET" "$off" --role edge); then
      echo "$(date +%T) SKIP $name offset=$off ($run)" | tee -a "$summary"; continue
    fi
    echo "$(date +%T) START $name offset=$off" | tee -a "$summary"
    if thesis-edge-run "$cfg" --offset "$off" --limit "$BLOCK"; then
      echo "$(date +%T) OK   $name offset=$off $(vcgencmd measure_temp) $(vcgencmd get_throttled)" | tee -a "$summary"
    else
      echo "$(date +%T) FAIL $name offset=$off" | tee -a "$summary"
    fi
  done
done
echo "$(date +%T) fertig → $summary" | tee -a "$summary"
