#!/usr/bin/env bash
# Artefakte (HEFs) und Bilder auf den Pi schieben, Ergebnisse zurückholen.
# Code kommt NICHT per rsync, sondern per `git pull` auf dem Pi – damit jeder
# Messlauf einem Commit zugeordnet ist.
#
#   scripts/sync_to_pi.sh push    # artifacts/ + data/ → Pi
#   scripts/sync_to_pi.sh pull    # Pi:results/ → lokal
set -euo pipefail

PI_HOST="${PI_HOST:-pi5}"                 # Host aus ~/.ssh/config
PI_DIR="${PI_DIR:-~/thesis-code}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "${1:-}" in
  push)
    rsync -avh --progress --include='*/' --include='*.hef' --exclude='*' \
          "$ROOT/artifacts/" "$PI_HOST:$PI_DIR/artifacts/"
    rsync -avh --progress "$ROOT/data/images/" "$PI_HOST:$PI_DIR/data/images/"
    ;;
  pull)
    rsync -avh --progress "$PI_HOST:$PI_DIR/results/" "$ROOT/results/"
    ;;
  *)
    echo "usage: $0 push|pull" >&2; exit 1 ;;
esac
