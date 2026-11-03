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
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("-n", type=int, help="n Bilder zufällig aus allen Kandidaten")
    grp.add_argument("--per-class", type=int,
                     help="geschichtet: je Klasse genau so viele Bilder (alle Klassen gleich stark)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = read_subset(args.labels)
    rng = random.Random(args.seed)
    if args.per_class:
        by_class: dict[int, list[SubsetItem]] = {}
        for it in sorted(pool, key=lambda it: it.image_id):
            by_class.setdefault(it.label, []).append(it)
        short = {c: len(v) for c, v in by_class.items() if len(v) < args.per_class}
        if short:
            raise SystemExit(f"{len(short)} Klassen haben weniger als {args.per_class} Kandidaten, z. B. {next(iter(short.items()))}")
        chosen = [it for c in sorted(by_class) for it in rng.sample(by_class[c], args.per_class)]
        desc = f"per_class={args.per_class} classes={len(by_class)} n={len(chosen)}"
    else:
        chosen = rng.sample(pool, args.n)
        desc = f"n={args.n}"
    chosen = sorted(chosen, key=lambda it: it.image_id)
    write_subset(args.out, chosen, header=f"source={args.labels} {desc} seed={args.seed}")
    print(f"{len(chosen)} Bilder → {args.out}")


if __name__ == "__main__":
    main()
