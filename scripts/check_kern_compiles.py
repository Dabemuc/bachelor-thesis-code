#!/usr/bin/env python3
"""Prüft die Kern-Compiles (artifacts/*_kNN/) – auf dem Rechner ausführen, auf dem die Artefakte liegen:

    python3 scripts/check_kern_compiles.py            # Tabelle auf stdout
    python3 scripts/check_kern_compiles.py > logs/check_kern.txt

Je Compile: HEF vorhanden, SHA-256 (gekürzt), Git-Commit/dirty, Kontext-Flow (Single/Multi),
Kalibriergröße laut Log, SNR der Ausgänge, Mapping-Auslastung, Laufzeit. Am Ende: doppelte HEF-Hashes.
Liest nur compile_meta.json und compile.log; ändert nichts.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
for d in sorted((ROOT / "artifacts").glob("*_k[0-9][0-9]")):
    meta = d / "compile_meta.json"
    log = d / "compile.log"
    m = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    t = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    hef = next(d.glob("*.hef"), None)
    snr = re.findall(r"output_layer(\d) SNR:\s*([\d.]+) dB", t)
    # Kalibriergröße: DFC-Fortschrittszeile der Kalibrierung („… 64/64 …“) bzw. „… 64 entries/images“
    calib = re.findall(r"[Cc]alibration[^\n]*?\b(\d+)/\1\b", t) or re.findall(
        r"[Cc]alibration[^\n]*?(\d+) (?:entries|images)", t)
    ctx = "single" if "Using Single-context flow" in t else ("multi" if re.search(r"[Mm]ulti-context", t) else "?")
    util = re.search(r"\| Total\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%", t)
    start = re.search(r"# Start: (\S+)", t)
    end = re.search(r"# Ende: (\S+)\s+Exit-Code: (\d+)", t)
    rows.append({
        "name": d.name,
        "hef": bool(hef),
        "sha": (m.get("hef_sha256") or "")[:12],
        "git": ((m.get("git") or {}).get("commit") or "")[:7],
        "dirty": (m.get("git") or {}).get("dirty"),
        "ctx": ctx,
        "calib": "/".join(sorted(set(calib))) or "?",
        "snr": " ".join(f"{a}:{b}" for a, b in snr) or "?",
        "util": "/".join(util.groups()) if util else "?",
        "start": start.group(1)[11:16] if start else "?",
        "end": end.group(1)[11:16] if end else "?",
        "exit": end.group(2) if end else "?",
    })

hdr = ["name", "hef", "sha", "git", "dirty", "ctx", "calib", "snr", "util", "start", "end", "exit"]
print("\t".join(hdr))
for r in rows:
    print("\t".join(str(r[h]) for h in hdr))
dups = [s for s, n in Counter(r["sha"] for r in rows if r["sha"]).items() if n > 1]
print(f"\n{len(rows)} Compile-Ordner, {sum(r['hef'] for r in rows)} HEFs, doppelte Hashes: {dups or 'keine'}")
