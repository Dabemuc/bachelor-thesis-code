#!/usr/bin/env bash
# Artefakte (HEFs) und Bilder auf den Pi schieben, Ergebnisse zurückholen.
# Code kommt NICHT per rsync, sondern per `git pull` auf dem Pi – damit jeder
# Messlauf einem Commit zugeordnet ist.
#
#   scripts/sync_to_pi.sh push    # artifacts/ + data/ → Pi
#   scripts/sync_to_pi.sh pull    # neue Pi:results/ → lokal VERSCHIEBEN (Pi bleibt git-sauber)
set -euo pipefail

PI_HOST="${PI_HOST:-pi5}"                 # Host aus ~/.ssh/config
PI_DIR="${PI_DIR:-~/bachelor-thesis-code}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "${1:-}" in
  push)
    rsync -avh --progress --include='*/' --include='*.hef' --include='compile_meta.json' --exclude='*' \
          "$ROOT/artifacts/" "$PI_HOST:$PI_DIR/artifacts/"
    rsync -avh --progress "$ROOT/data/images/" "$PI_HOST:$PI_DIR/data/images/"
    ;;
  pull)
    # Neue Lauf-Ergebnisse vom Pi auf den Mac VERSCHIEBEN (nicht kopieren): Danach liegen sie nur
    # noch hier (meta.json + CSV werden hier committet, .npz bleiben lokal/Backup) und der Pi
    # bleibt sauber für `git pull`. Nur Dateien, die auf dem Pi NICHT von git getrackt sind –
    # bereits committete Läufe bleiben dort unangetastet. rsync löscht eine Quelldatei erst nach
    # erfolgreicher Übertragung.
    LIST="$(mktemp)"
    trap 'rm -f "$LIST"' EXIT
    ssh "$PI_HOST" "cd $PI_DIR && git ls-files --others -- results/" > "$LIST"
    if [ ! -s "$LIST" ]; then
      echo "Keine neuen Ergebnisse auf dem Pi."
      exit 0
    fi
    echo "Verschiebe $(wc -l < "$LIST" | tr -d ' ') Dateien vom Pi:"
    rsync -avh --progress --remove-source-files --files-from="$LIST" "$PI_HOST:$PI_DIR/" "$ROOT/"
    ssh "$PI_HOST" "find $PI_DIR/results -mindepth 1 -type d -empty -delete"
    ;;
  *)
    echo "usage: $0 push|pull" >&2; exit 1 ;;
esac
