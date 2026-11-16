#!/usr/bin/env bash
# Kern-Compiles (20 = 10 je Modell) nacheinander – IM DFC-CONTAINER ausführen:
#     bash scripts/compile_kern.sh            # alle
#     bash scripts/compile_kern.sh resnet18   # nur Configs, deren Name das Muster enthält
# Bereits vorhandene HEFs werden übersprungen (Wiederaufnahme nach Abbruch möglich).
# Ein Fehler bricht nicht ab; Zusammenfassung am Ende und in logs/compile_kern_<zeit>.txt.
set -u
cd "$(dirname "$0")/.."
pattern="${1:-}"
summary="logs/compile_kern_$(date +%Y%m%d-%H%M%S).txt"
mkdir -p logs
ok=0; skip=0; fail=0
for cfg in configs/kern/*_k[0-9][0-9].yaml; do
  [[ -n "$pattern" && "$cfg" != *"$pattern"* ]] && continue
  name=$(basename "$cfg" .yaml)
  hef=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['hef'])" "$cfg")
  if [[ -f "artifacts/$hef" ]]; then
    echo "$(date +%T) SKIP $name (HEF vorhanden)" | tee -a "$summary"; skip=$((skip+1)); continue
  fi
  start=$(date +%s)
  echo "$(date +%T) START $name" | tee -a "$summary"
  if thesis-compile "$cfg"; then
    echo "$(date +%T) OK   $name ($(( $(date +%s) - start ))s)" | tee -a "$summary"; ok=$((ok+1))
  else
    echo "$(date +%T) FAIL $name ($(( $(date +%s) - start ))s)" | tee -a "$summary"; fail=$((fail+1))
  fi
done
echo "Fertig: $ok ok, $skip übersprungen, $fail fehlgeschlagen → $summary" | tee -a "$summary"
