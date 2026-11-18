#!/usr/bin/env bash
# Ergebnisse vom Windows-PC über den Pi auf den Mac holen (Gegenstück zu push_results_to_mac.ps1 -Via).
# AUF DEM MAC ausführen:
#     scripts/relay_from_pi.sh manifest   # Liste der vorhandenen results/-Dateien → Pi:~/relay/mac_manifest.txt
#     scripts/relay_from_pi.sh pull       # Pi:~/relay/results_relay.tar holen, prüfen, auspacken (nie überschreiben),
#                                         # danach auf dem Pi und lokal löschen
set -euo pipefail
PI_HOST="${PI_HOST:-lee-local}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-}" in
  manifest)
    (cd "$ROOT/results" && find . -type f | sed 's|^\./||') | ssh "$PI_HOST" "mkdir -p ~/relay && cat > ~/relay/mac_manifest.txt && wc -l < ~/relay/mac_manifest.txt"
    echo "Manifest liegt auf $PI_HOST:~/relay/mac_manifest.txt"
    ;;
  pull)
    tar_local="$ROOT/results_relay.tar"
    rsync -ah --progress "$PI_HOST:relay/results_relay.tar" "$tar_local"
    echo "Prüfe Archiv …"
    n=$(tar -tf "$tar_local" | wc -l | tr -d ' ')
    bad=$(tar -tf "$tar_local" | grep -v '^results/' || true)
    [ -z "$bad" ] || { echo "Abbruch: Pfade außerhalb von results/: $bad" >&2; exit 1; }
    (cd "$ROOT" && tar -xkf "$tar_local")
    echo "$n Dateien ausgepackt (vorhandene nicht überschrieben)."
    ssh "$PI_HOST" "rm -f ~/relay/results_relay.tar; df -h ~ | tail -1"
    rm -f "$tar_local"
    ;;
  *) echo "usage: $0 manifest|pull" >&2; exit 1 ;;
esac
