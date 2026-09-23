#!/usr/bin/env bash
# Vor jeder Messreihe auf dem Pi: definierte Bedingungen herstellen.
set -euo pipefail

echo "== CPU-Governor → performance (kein Frequenz-Rampen während der Messung) =="
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

echo "== Ausgangszustand =="
vcgencmd measure_temp
vcgencmd get_throttled   # 0x0 erwartet; alles andere → Netzteil/Kühlung prüfen

echo "Hinweis: Läufe in tmux starten (SSH-Abbruch killt sonst den Lauf)."
