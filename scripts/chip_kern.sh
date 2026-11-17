#!/usr/bin/env bash
# Chip-Läufe des Kerns – AUF DEM PI ausführen (in tmux):
#     bash scripts/chip_kern.sh             # alle 20 Kern-Configs
#     bash scripts/chip_kern.sh resnet18    # nur Configs mit diesem Namensteil
# Je Config alle 48 000 Eval-Bilder in Blöcken à BLOCK (Default 8000 → 6 Blöcke), weil der Runner
# alle Feature-Maps bis zum Ende im RAM hält (MobileNetV3: 8000 Bilder ≈ 1,5 GB).
# Fertige Blöcke werden übersprungen → nach Abbruch einfach neu starten. „Fertig“ heißt: in
# logs/chip_kern_done.log eingetragen (überlebt das Abholen durch den Mac) oder noch als Run-Ordner da.
# Speicher (Pi: 29 GB SD): auf dem Mac parallel `scripts/pull_pi_done.sh --watch` laufen lassen –
# holt nur Blöcke aus chip_kern_done.log ab (verschiebt sie) und gibt so Platz frei.
# Voraussetzung: HEFs unter artifacts/<name>/ und alle 48 000 Bilder unter data/images/.
set -u
cd "$(dirname "$0")/.."
# venv des Repos verwenden, falls nicht aktiviert (tmux-Sitzungen starten ohne)
[[ -x .venv/bin/thesis-edge-run ]] && PATH="$PWD/.venv/bin:$PATH"
command -v thesis-edge-run >/dev/null || { echo "thesis-edge-run nicht gefunden – venv aktivieren / pip install -e ." >&2; exit 1; }
pattern="${1:-}"
BLOCK=${BLOCK:-8000}
TOTAL=48000
SUBSET=imagenet_eval_48000.txt
summary="logs/chip_kern_$(date +%Y%m%d-%H%M%S).log"
done_log="logs/chip_kern_done.log"   # Zeilen: <name> <offset> <run-ordner>
mkdir -p logs
touch "$done_log"
bash scripts/pi_bench_mode.sh | tee -a "$summary"
for cfg in configs/kern/*_k[0-9][0-9].yaml; do
  [[ -n "$pattern" && "$cfg" != *"$pattern"* ]] && continue
  name=$(basename "$cfg" .yaml)
  for ((off=0; off<TOTAL; off+=BLOCK)); do
    if grep -q "^$name $off " "$done_log"; then
      echo "$(date +%T) SKIP $name offset=$off (laut $done_log fertig)" | tee -a "$summary"; continue
    fi
    if run=$(python3 scripts/find_run.py "$name" "$SUBSET" "$off" --role edge --need features.npz); then
      echo "$name $off $run" >> "$done_log"
      echo "$(date +%T) SKIP $name offset=$off ($run)" | tee -a "$summary"; continue
    fi
    echo "$(date +%T) START $name offset=$off" | tee -a "$summary"
    if thesis-edge-run "$cfg" --offset "$off" --limit "$BLOCK" \
       && run=$(python3 scripts/find_run.py "$name" "$SUBSET" "$off" --role edge --need features.npz); then
      echo "$name $off $run" >> "$done_log"   # erst jetzt darf der Mac den Ordner abholen
      echo "$(date +%T) OK   $name offset=$off $(vcgencmd measure_temp) $(vcgencmd get_throttled)" | tee -a "$summary"
    else
      echo "$(date +%T) FAIL $name offset=$off" | tee -a "$summary"
    fi
  done
done
echo "$(date +%T) fertig → $summary" | tee -a "$summary"
