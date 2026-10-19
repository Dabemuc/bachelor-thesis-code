#!/usr/bin/env bash
# Nur die Bilder einer festen Liste (data/subsets/<name>) an ein Ziel kopieren.
# Für Rechner, die die Bilder nicht selbst herunterladen können (ImageNet: Login nötig).
#
#   scripts/sync_subset_images.sh imagenet_calib_2000_seed1.txt  <ziel>  [N]
#
#   <ziel> = lokaler Ordner oder rsync-Ziel "host:pfad/zu/data/images/"
#   N      = optional nur die ersten N Bilder der Liste (wie --limit)
#
# ⚠️ ImageNet-Bilder nur auf eigene Rechner kopieren (Nutzungsbedingungen Punkt 4).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBSET="$ROOT/data/subsets/${1:?Liste fehlt}"
DEST="${2:?Ziel fehlt}"
N="${3:-}"

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
# awk statt grep|cut|head: head würde die Pipe vorzeitig schließen (SIGPIPE + pipefail)
awk -F'\t' -v n="${N:-0}" '!/^#/ && NF { print $1; if (n > 0 && ++c >= n) exit }' "$SUBSET" > "$LIST"
echo "Kopiere $(wc -l < "$LIST" | tr -d ' ') Bilder aus $(basename "$SUBSET") → $DEST"
rsync -avh --files-from="$LIST" "$ROOT/data/images/" "$DEST"
