#!/usr/bin/env python3
"""Run-Ordner zu (Config-Name, Subset, Offset[, Rolle/Kontext]) finden – für wiederaufnehmbare Skripte.

    python3 scripts/find_run.py NAME SUBSET OFFSET [--role edge|emulator|reference] [--context sdk_quantized]

Gibt den neuesten passenden Ordner (relativ zum Repo) aus, Exit-Code 1 wenn keiner existiert.
Ein Ordner zählt nur, wenn --need (Default logits.npz, also Lauf fertig) vorhanden ist.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("subset")
ap.add_argument("offset", type=int)
ap.add_argument("--role", default=None)
ap.add_argument("--context", default=None)
ap.add_argument("--need", default="logits.npz",
                help="Datei, die vorhanden sein muss (Default logits.npz = Lauf fertig; für Zielklassen-Läufe\n"
                     "aus Git reicht predictions.csv, die *.npz liegen dort nicht)")
a = ap.parse_args()

hits = []
for meta in (ROOT / "results").glob(f"*_{a.name}*/meta.json"):
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    c = m.get("config", {})
    if c.get("name") != a.name or c.get("subset") != a.subset or int(m.get("offset", 0)) != a.offset:
        continue
    if a.role and m.get("role") != a.role:
        continue
    if a.context and a.context not in str(m.get("emulator_context", "")):
        continue
    if not (meta.parent / a.need).exists():
        continue
    hits.append(meta.parent)
if not hits:
    sys.exit(1)
print(sorted(hits)[-1].relative_to(ROOT))
