#!/usr/bin/env bash
# Fertige Chip-Blöcke vom Pi auf den Mac VERSCHIEBEN – nur Ordner, die in logs/chip_kern_done.log
# auf dem Pi stehen (also vollständig geschrieben sind), nie einen laufenden Block.
#     scripts/pull_pi_done.sh            # einmal
#     scripts/pull_pi_done.sh --watch    # alle 5 min (INTERVAL), bis Strg+C
# Wie `sync_to_pi.sh pull`: rsync --remove-source-files, nur von git nicht getrackte Dateien.
set -euo pipefail
PI_HOST="${PI_HOST:-lee-local}"
PI_DIR="${PI_DIR:-~/bachelor-thesis-code}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${INTERVAL:-300}"   # 5 min: MobileNetV3 erzeugt ≈ 1,5 GB je 8000er-Block

pull_once() {
  local list; list="$(mktemp)"
  # Run-Ordner aus der Done-Liste, davon nur ungetrackte Dateien, die noch auf dem Pi liegen
  ssh "$PI_HOST" "cd $PI_DIR && [ -f logs/chip_kern_done.log ] && awk '{print \$3}' logs/chip_kern_done.log | sort -u | while read -r d; do [ -d \"\$d\" ] && git ls-files --others -- \"\$d\"; done" > "$list" || true
  if [ ! -s "$list" ]; then
    echo "$(date +%T) nichts Neues"; rm -f "$list"; return
  fi
  echo "$(date +%T) verschiebe $(wc -l < "$list" | tr -d ' ') Dateien"
  rsync -ah --remove-source-files --files-from="$list" "$PI_HOST:$PI_DIR/" "$ROOT/"
  ssh "$PI_HOST" "find $PI_DIR/results -mindepth 1 -type d -empty -delete; df -h ~ | tail -1"
  rm -f "$list"
}

if [[ "${1:-}" == --watch ]]; then
  while true; do pull_once; sleep "$INTERVAL"; done
else
  pull_once
fi
