# Subsets

Eine Datei pro fester Bildliste, Format je Zeile: `<Pfad unter data/images/>\t<Klassenindex>`.

Diese Dateien werden committet – die Bilder selbst (`data/images/`) nicht.

Geplant:

- `val_1000_seed0.txt` – Evaluations-Subset (ImageNet-Val o. Ä.)
- `calib_1024_seed1.txt` – Kalibrierdaten für die Quantisierung, **disjunkt** zum Evaluations-Subset
  (sonst kalibriert man auf den Testbildern)

Erzeugen: `thesis-make-subset <alle_labels.txt> data/subsets/val_1000_seed0.txt -n 1000 --seed 0`

⚠️ Datensatz noch offen (Methodik → „Welche konkreten Modelle/Datensätze?"). ImageNet-Val
erfordert Registrierung bei image-net.org; Nutzungsbedingungen für die Arbeit prüfen.

## Stand 19.10.2026: ImageNet (ILSVRC2012) Val

Erzeugt mit `thesis-prepare-imagenet` (Archive von image-net.org in `data/images/imagenet/_archives/`, MD5 geprüft):

| Datei | n | Zweck |
|---|---|---|
| `imagenet_val_all_50000.txt` | 50 000 | **nur** Sanity-Check der FP32-Top-1 gegen die torchvision-Referenzwerte (enthält die Kalibrierbilder) |
| `imagenet_calib_2000_seed1.txt` | 2 000 | 2 Bilder pro Klasse, Seed 1 – Kalibrierung + Finetune |
| `imagenet_eval_48000.txt` | 48 000 | alle übrigen Val-Bilder – Evaluation; `--limit N` = die ersten N (nach Dateiname, Klassen gemischt) |

Kalibrierung aus einem abgetrennten Teil von Val statt aus Train: Entscheidung vom 19.10.2026 (Vault: „Phase 0 – Defaults“).
Klassenindex = sortierte WNID-Reihenfolge (torchvision-Standard); Zuordnung ILSVRC2012_ID → WNID in `data/imagenet_ilsvrc2012_id_to_wnid.tsv`.
Die Imagenette-Listen bleiben für AP0 erhalten.
