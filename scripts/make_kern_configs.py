#!/usr/bin/env python3
"""Erzeugt die 20 Kern-Configs (k = 10 Compiles je Modell) unter configs/kern/.

    python scripts/make_kern_configs.py

Jede Config ist identisch zur Basis-Config bis auf name, hef, repeats und notes;
am Ende wird das geprüft. Vorhandene Dateien werden überschrieben (deterministisch).
"""
import glob
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASES = [("resnet18_int8_lvl2", "resnet18"), ("mobilenet_v3_large_int8_lvl2", "mobilenet_v3_large")]
K = 10

out = ROOT / "configs" / "kern"
out.mkdir(parents=True, exist_ok=True)
for base, model in BASES:
    s = (ROOT / "configs" / f"{base}.yaml").read_text(encoding="utf-8")
    body = s[s.index("name:"):]
    for k in range(1, K + 1):
        n = f"{base}_k{k:02d}"
        b = body.replace(f"name: {base}\n", f"name: {n}\n", 1)
        b = re.sub(r"hef: .*\n", f"hef: {n}/{model}.hef\n", b, count=1)
        b = re.sub(r"repeats: .*\n", "repeats: 1\n", b, count=1)
        b = re.sub(r"notes: .*\n",
                   f'notes: "Kern (Phase 2): Compile {k}/{K}, identisch zu {base}.yaml bis auf Name/HEF-Pfad"\n',
                   b, count=1)
        head = (f"# Kern (Phase 2), Compile {k} von {K} – erzeugt mit scripts/make_kern_configs.py am 16.11.2026.\n"
                f"# Identisch zu configs/{base}.yaml (DFC-Stufe 2, Kalibrierung 64 = DFC-Default, Default-Kompilierstufe)\n"
                f"# bis auf name, hef und repeats/notes. Die {K} Compiles unterscheiden sich nur durch den Finetune-Zufall (P3).\n")
        (out / f"{n}.yaml").write_text(head + b, encoding="utf-8")

for base, _ in BASES:
    ref = yaml.safe_load((ROOT / "configs" / f"{base}.yaml").read_text(encoding="utf-8"))
    files = sorted(glob.glob(str(out / f"{base}_k*.yaml")))
    assert len(files) == K, files
    for f in files:
        c = yaml.safe_load(Path(f).read_text(encoding="utf-8"))
        diff = {key for key in set(ref) | set(c) if ref.get(key) != c.get(key)}
        assert diff <= {"name", "hef", "repeats", "notes"}, (f, diff)
    print(f"{base}: {K} Configs geprüft")
