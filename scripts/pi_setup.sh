#!/usr/bin/env bash
# Einrichtung auf dem Raspberry Pi 5.
#   scripts/pi_setup.sh          # installieren, danach neu starten
#   scripts/pi_setup.sh verify   # nach dem Neustart: Gerät + Versionen prüfen
set -euo pipefail

if [[ "${1:-}" == "verify" ]]; then
  echo "== Gerät prüfen: muss HAILO8 melden, nicht HAILO8L =="
  hailortcli fw-control identify
  echo "== Installierte Hailo-Pakete (HailoRT-Version für die DFC-Wahl notieren!) =="
  dpkg -l | grep -i hailo || true
  echo "== PCIe-Link (Gen 3 = 8GT/s erwartet) =="
  sudo lspci -vv 2>/dev/null | grep -i -A30 hailo | grep -i 'LnkSta:' || true
  exit 0
fi

echo "== System aktualisieren =="
sudo apt update && sudo apt full-upgrade -y

echo "== Hailo-Stack (PCIe-Treiber, HailoRT, python3-hailort, Firmware) =="
sudo apt install -y hailo-all

echo ">> Jetzt neu starten (sudo reboot), danach: scripts/pi_setup.sh verify"
