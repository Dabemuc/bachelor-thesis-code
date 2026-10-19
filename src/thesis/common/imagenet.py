"""ImageNet (ILSVRC2012) Val entpacken und die festen Subsets erzeugen.

    thesis-prepare-imagenet            # Standard: 2 Kalibrierbilder pro Klasse, Seed 1

Erwartet die Archive von image-net.org in ``data/images/imagenet/_archives/``:
    ILSVRC2012_img_val.tar          (MD5 29b22e2961454d5413ddabcf34fc5622)
    ILSVRC2012_devkit_t12.tar.gz    (MD5 fa75699e90414af021442c21a62c3abf)
(Prüfsummen wie in ``torchvision/datasets/imagenet.py``, ``ARCHIVE_META``.)

Erzeugt:
    data/images/imagenet/val/ILSVRC2012_val_XXXXXXXX.JPEG   (flach, wie im Archiv)
    data/subsets/imagenet_val_all_50000.txt     alle 50 000 Val-Bilder – NUR für den
                                                Sanity-Check der FP32-Top-1 gegen die
                                                torchvision-Referenzwerte
    data/subsets/imagenet_calib_2000_seed1.txt  2 Bilder pro Klasse: Kalibrierung + Finetune
    data/subsets/imagenet_eval_48000.txt        alle übrigen Val-Bilder: Evaluation
                                                → disjunkt zur Kalibrierliste per Konstruktion

Kalibrierung aus einem abgetrennten Teil von Val statt aus Train (Entscheidung
19.10.2026, siehe Vault „Phase 0 – Defaults“): Eval- und Kalibrierliste sind
disjunkt, und das DFC-Finetune nutzt keine Labels (Distillation).

Klassenindex = Position der WNID in der sortierten Liste der 1000 ILSVRC2012-Klassen
(Standardreihenfolge, wie torchvision ``ImageFolder`` / die vortrainierten Gewichte).
Zur Kontrolle wird gegen die Imagenette-Zuordnung (``imagenette.WNID_TO_INDEX``) geprüft.

Klassenzuordnung ILSVRC2012_ID → WNID: aus ``data/imagenet_ilsvrc2012_id_to_wnid.tsv``
(committet, einmalig aus ``meta.mat`` des Devkits erzeugt). Fehlt die Datei, wird
``meta.mat`` direkt gelesen – dafür ist ``scipy`` nötig; ``--write-class-map``
schreibt die TSV dann neu.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

from .config import DATA_DIR
from .imagenette import WNID_TO_INDEX as IMAGENETTE_WNID_TO_INDEX
from .subset import SubsetItem, write_subset

ARCHIVES = {
    "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
}
SOURCE = "https://image-net.org (ILSVRC2012)"
REL_ROOT = "imagenet/val"   # relativ zu data/images/
CLASS_MAP = DATA_DIR / "imagenet_ilsvrc2012_id_to_wnid.tsv"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_archives(archive_dir: Path, skip_md5: bool) -> dict[str, Path]:
    paths = {}
    for key, (name, md5) in ARCHIVES.items():
        p = archive_dir / name
        if not p.exists():
            sys.exit(f"Fehlt: {p}")
        if not skip_md5:
            got = _md5(p)
            if got != md5:
                sys.exit(f"MD5 falsch für {name}: {got} (erwartet {md5})")
            print(f"MD5 ok: {name}")
        paths[key] = p
    return paths


def _class_map_from_meta(meta: bytes) -> dict[int, tuple[str, str]]:
    try:
        from scipy.io import loadmat
    except ImportError:
        sys.exit(f"{CLASS_MAP} fehlt und scipy ist nicht installiert (pip install scipy)")
    synsets = loadmat(io.BytesIO(meta), squeeze_me=True)["synsets"]
    # Felder: ILSVRC2012_ID, WNID, words, gloss, num_children, …
    # nur die 1000 Blatt-Klassen (num_children == 0) haben ILSVRC2012_ID 1..1000
    return {int(s[0]): (str(s[1]), str(s[2])) for s in synsets if int(s[4]) == 0}


def _read_class_map() -> dict[int, str]:
    out = {}
    for line in CLASS_MAP.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            cid, wnid, _ = line.split("\t", 2)
            out[int(cid)] = wnid
    return out


def _val_labels(devkit: Path, write_class_map: bool) -> dict[str, str]:
    """Dateiname → WNID, aus Klassenzuordnung + ILSVRC2012_validation_ground_truth.txt."""
    with tarfile.open(devkit, "r:gz") as tar:
        meta = tar.extractfile("ILSVRC2012_devkit_t12/data/meta.mat").read()
        gt = tar.extractfile(
            "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt").read()
    if write_class_map or not CLASS_MAP.exists():
        full = _class_map_from_meta(meta)
        lines = ["# ILSVRC2012_ID\tWNID\twords  (aus ILSVRC2012_devkit_t12/data/meta.mat, "
                 "nur die 1000 Blatt-Klassen)"]
        lines += [f"{k}\t{w}\t{d}" for k, (w, d) in sorted(full.items())]
        CLASS_MAP.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Klassenzuordnung geschrieben: {CLASS_MAP}")
    id_to_wnid = _read_class_map()
    if len(id_to_wnid) != 1000:
        sys.exit(f"Erwartet 1000 Klassen in meta.mat, gefunden {len(id_to_wnid)}")
    ids = [int(x) for x in gt.decode().split()]
    if len(ids) != 50000:
        sys.exit(f"Erwartet 50000 Val-Labels, gefunden {len(ids)}")
    return {f"ILSVRC2012_val_{i:08d}.JPEG": id_to_wnid[c] for i, c in enumerate(ids, start=1)}


def _extract_val(val_tar: Path, out_dir: Path) -> int:
    """Entpackt idempotent (vorhandene Dateien bleiben); schreibt atomar über .part."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    with tarfile.open(val_tar, "r:") as tar:
        for m in tar:
            if not m.isfile():
                continue
            target = out_dir / Path(m.name).name
            if target.exists():
                continue
            tmp = target.with_suffix(".part")
            tmp.write_bytes(tar.extractfile(m).read())
            tmp.rename(target)
            n_new += 1
    return n_new


def main() -> None:
    ap = argparse.ArgumentParser(description="ImageNet-Val entpacken + Subsets erzeugen")
    ap.add_argument("--per-class-calib", type=int, default=2)
    ap.add_argument("--seed-calib", type=int, default=1)
    ap.add_argument("--archive-dir", type=Path, default=DATA_DIR / "images" / "imagenet" / "_archives")
    ap.add_argument("--skip-md5", action="store_true")
    ap.add_argument("--skip-extract", action="store_true", help="nur Listen schreiben")
    ap.add_argument("--write-class-map", action="store_true",
                    help="Klassenzuordnung neu aus meta.mat erzeugen (braucht scipy)")
    args = ap.parse_args()

    paths = _check_archives(args.archive_dir, args.skip_md5)
    file_to_wnid = _val_labels(paths["devkit"], args.write_class_map)
    wnids = sorted(set(file_to_wnid.values()))
    wnid_to_index = {w: i for i, w in enumerate(wnids)}
    for w, i in IMAGENETTE_WNID_TO_INDEX.items():   # Kontrolle der Indexreihenfolge
        if wnid_to_index[w] != i:
            sys.exit(f"Index-Kontrolle fehlgeschlagen: {w} → {wnid_to_index[w]}, erwartet {i}")
    print("Klassenindex-Kontrolle gegen Imagenette: ok")

    images_dir = DATA_DIR / "images"
    if not args.skip_extract:
        n = _extract_val(paths["val"], images_dir / REL_ROOT)
        print(f"{n} Dateien neu entpackt nach {images_dir / REL_ROOT}")
    missing = [f for f in file_to_wnid if not (images_dir / REL_ROOT / f).exists()]
    if missing:
        sys.exit(f"{len(missing)} Val-Bilder fehlen noch (z. B. {missing[0]}) – erneut ausführen")

    def item(f: str) -> SubsetItem:
        return SubsetItem(image_id=f"{REL_ROOT}/{f}", label=wnid_to_index[file_to_wnid[f]])

    by_class: dict[str, list[str]] = defaultdict(list)
    for f in sorted(file_to_wnid):
        by_class[file_to_wnid[f]].append(f)
    rng = random.Random(args.seed_calib)
    calib = sorted(f for w in wnids for f in rng.sample(by_class[w], args.per_class_calib))
    calib_set = set(calib)
    eval_ = [f for f in sorted(file_to_wnid) if f not in calib_set]

    subsets_dir = DATA_DIR / "subsets"
    n_cal = len(calib)
    all_file = subsets_dir / "imagenet_val_all_50000.txt"
    calib_file = subsets_dir / f"imagenet_calib_{n_cal}_seed{args.seed_calib}.txt"
    eval_file = subsets_dir / f"imagenet_eval_{len(eval_)}.txt"
    write_subset(all_file, [item(f) for f in sorted(file_to_wnid)],
                 header=f"source={SOURCE} split=val n=50000\n"
                        "⚠️ NUR für den Sanity-Check der FP32-Top-1 – enthält die Kalibrierbilder")
    write_subset(calib_file, [item(f) for f in calib],
                 header=f"source={SOURCE} split=val per_class={args.per_class_calib} "
                        f"seed={args.seed_calib} n={n_cal}\n"
                        f"Kalibrierung + Finetune, disjunkt zu {eval_file.name}")
    write_subset(eval_file, [item(f) for f in eval_],
                 header=f"source={SOURCE} split=val n={len(eval_)} (alle außer {calib_file.name})\n"
                        "Evaluation; --limit N nimmt die ersten N (nach Dateiname, Klassen gemischt)")
    print(f"→ {all_file}\n→ {calib_file}\n→ {eval_file}")


if __name__ == "__main__":
    main()
