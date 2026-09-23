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
