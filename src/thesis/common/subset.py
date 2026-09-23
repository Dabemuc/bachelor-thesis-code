"""Feste Bildliste (Subset) für alle Läufe.

Ein Subset ist eine Textdatei in ``data/subsets/`` mit einer Zeile pro Bild:

    <relativer Pfad unter data/images/>\t<Ground-Truth-Klassenindex>

Die Liste wird einmal mit festem Seed erzeugt und danach nie mehr verändert –
jede Maschine iteriert in exakt dieser Reihenfolge.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR

IMAGES_DIR = DATA_DIR / "images"


@dataclass(frozen=True)
class SubsetItem:
    image_id: str       # relativer Pfad, dient zugleich als stabile ID
    label: int          # Ground Truth (−1 = unbekannt)

    @property
    def path(self) -> Path:
        return IMAGES_DIR / self.image_id


def read_subset(path: str | Path) -> list[SubsetItem]:
    items: list[SubsetItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel, _, label = line.partition("\t")
        items.append(SubsetItem(rel, int(label) if label else -1))
    return items


def write_subset(path: str | Path, items: list[SubsetItem], header: str = "") -> None:
    lines = [f"# {h}" for h in header.splitlines()] if header else []
    lines += [f"{it.image_id}\t{it.label}" for it in items]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Subset aus einer Label-Datei ziehen (``<rel_path>\\t<label>`` je Zeile)."""
    ap = argparse.ArgumentParser(description="Festes Bild-Subset erzeugen")
    ap.add_argument("labels", help="Datei mit allen Kandidaten: <rel_path>\\t<label>")
    ap.add_argument("out", help="Zieldatei, z. B. data/subsets/val_1000_seed0.txt")
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = read_subset(args.labels)
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(pool, args.n), key=lambda it: it.image_id)
    write_subset(args.out, chosen, header=f"source={args.labels} n={args.n} seed={args.seed}")
    print(f"{len(chosen)} Bilder → {args.out}")


if __name__ == "__main__":
    main()
