"""Imagenette herunterladen und die festen Subsets für AP0 erzeugen.

    thesis-fetch-imagenette            # Standard: 1000 Eval-Bilder, 1024 Kalibrierbilder

Imagenette (fast.ai) = 10 leicht unterscheidbare ImageNet-Klassen. Verwendet wird
die Variante ``imagenette2-320`` (kürzere Seite bereits auf 320 px skaliert, ~330 MB).
160 px wäre zu klein: das gemeinsame Preprocessing skaliert auf 256 und würde
dann hochskalieren.

Erzeugt:
    data/images/imagenette2-320/{train,val}/<wnid>/*.JPEG    (nur die benötigten Dateien)
    data/subsets/imagenette_val_1000_seed0.txt      Evaluation  – aus dem Imagenette-VAL-Split
    data/subsets/imagenette_calib_1024_seed1.txt    Kalibrierung – aus dem Imagenette-TRAIN-Split
                                                     → disjunkt per Konstruktion

⚠️ Nur für AP0 (Funktionstest der Kette): Alle Imagenette-Bilder stammen aus dem
ImageNet-*Trainings*set, also auch aus den Trainingsdaten der torchvision-Gewichte.
Accuracy-Werte sind dadurch zu optimistisch. Für die Messungen der Arbeit ImageNet-Val verwenden.
"""

from __future__ import annotations

import argparse
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

from .config import DATA_DIR
from .subset import SubsetItem, write_subset

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
ROOT_NAME = "imagenette2-320"

# WordNet-ID → ImageNet-1k-Klassenindex (Standardreihenfolge der ILSVRC2012-Klassen)
WNID_TO_INDEX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Archiv aus Cache: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Lade {url}")

    def hook(block, block_size, total):
        if total > 0:
            print(f"\r  {min(100, 100 * block * block_size / total):5.1f} %", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.rename(dest)
    print()


def _members_by_split(tar: tarfile.TarFile) -> dict[str, list[tarfile.TarInfo]]:
    out: dict[str, list[tarfile.TarInfo]] = {"train": [], "val": []}
    for m in tar.getmembers():
        parts = m.name.split("/")
        # imagenette2-320/<split>/<wnid>/<file>.JPEG
        if m.isfile() and len(parts) == 4 and parts[1] in out and parts[2] in WNID_TO_INDEX:
            out[parts[1]].append(m)
    for v in out.values():
        v.sort(key=lambda m: m.name)
    return out


def _sample(members: list[tarfile.TarInfo], n: int, seed: int) -> list[tarfile.TarInfo]:
    if n > len(members):
        sys.exit(f"Nur {len(members)} Bilder verfügbar, {n} angefordert.")
    return sorted(random.Random(seed).sample(members, n), key=lambda m: m.name)


def _to_item(m: tarfile.TarInfo) -> SubsetItem:
    return SubsetItem(image_id=m.name, label=WNID_TO_INDEX[m.name.split("/")[2]])


def main() -> None:
    ap = argparse.ArgumentParser(description="Imagenette laden + Subsets erzeugen (AP0)")
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-calib", type=int, default=1024)
    ap.add_argument("--seed-eval", type=int, default=0)
    ap.add_argument("--seed-calib", type=int, default=1)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "thesis")
    args = ap.parse_args()

    tgz = args.cache / f"{ROOT_NAME}.tgz"
    _download(URL, tgz)

    images_dir = DATA_DIR / "images"
    subsets_dir = DATA_DIR / "subsets"
    images_dir.mkdir(parents=True, exist_ok=True)
    subsets_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tgz, "r:gz") as tar:
        by_split = _members_by_split(tar)
        print(f"Imagenette: {len(by_split['train'])} train / {len(by_split['val'])} val")
        eval_members = _sample(by_split["val"], args.n_eval, args.seed_eval)
        calib_members = _sample(by_split["train"], args.n_calib, args.seed_calib)
        wanted = {m.name for m in eval_members + calib_members}
        todo = [m for m in tar.getmembers() if m.name in wanted and not (images_dir / m.name).exists()]
        print(f"Entpacke {len(todo)} Dateien nach {images_dir}")

    # zweiter Durchlauf: gz-Archive sind nur sequentiell lesbar
    with tarfile.open(tgz, "r:gz") as tar:
        for m in tar:
            if m.name in wanted:
                target = images_dir / m.name
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(m).read())

    eval_file = subsets_dir / f"imagenette_val_{args.n_eval}_seed{args.seed_eval}.txt"
    calib_file = subsets_dir / f"imagenette_calib_{args.n_calib}_seed{args.seed_calib}.txt"
    note = "⚠️ Imagenette = ImageNet-TRAIN-Bilder → nur für AP0, nicht für Messwerte der Arbeit"
    write_subset(eval_file, [_to_item(m) for m in eval_members],
                 header=f"source={URL} split=val n={args.n_eval} seed={args.seed_eval}\n{note}")
    write_subset(calib_file, [_to_item(m) for m in calib_members],
                 header=f"source={URL} split=train n={args.n_calib} seed={args.seed_calib}\n{note}")
    print(f"→ {eval_file}\n→ {calib_file}")


if __name__ == "__main__":
    main()
