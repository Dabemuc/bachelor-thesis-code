# Anweisungen für KI-Agenten (bachelor-thesis-code)

Code zur Bachelorarbeit von Daniel: Einfluss hardware-realer Quantisierung (Hailo-8 auf Raspberry Pi 5 + AI HAT+) auf CAM-Erklärungen (Grad-CAM, Eigen-CAM). Aufbau und Befehle: `README.md`.

## Dokumentation liegt im Obsidian-Vault, nicht hier

`/Users/dbenner/Library/Mobile Documents/iCloud~md~obsidian/Documents/MainVault/01 - Knowledge/Data/Bachelorarbeit/Bachelorarbeit/`

- Einstieg: `🏠 Bachelorarbeit - Home.md`. Regeln für die Dokumentation: `AGENTS.md` im Vault.
- Hast du Zugriff auf den Vault, lies zu Beginn Home und den neuesten Eintrag in `Logbuch/`.
- **Nach einer Messreihe:** Ergebnisse als Notiz `3 Experimente/E<n> <Titel>.md` im Vault festhalten (Modell, Präzision, Datensatz + n, DFC-Version und Optimierungsstufe, Config, Commit, Run-Ordner in `results/`, Tabelle, Vorbehalte). Den Tag im Logbuch eintragen, Home aktualisieren.
- **Nach Änderungen am Aufbau** (Rechner, Versionen, Regeln, Befehle): `2 Setup/Entwicklungs-Workflow.md` bzw. `2 Setup/Hardware.md` im Vault nachziehen.
- Kein Vault-Zugriff: dem Nutzer am Ende auflisten, was im Vault nachgetragen werden muss.

## Feste Randbedingungen

- **Rechner-Rollen:** Windows-PC (x86) kompiliert mit dem Hailo DFC in Podman (`docker/dfc.Dockerfile`) · Pi „Lee“ misst mit HailoRT (`thesis.edge`) · Mac bzw. DGX Spark: FP32-Referenz und Auswertung (`thesis.reference`, `thesis.analysis`).
- **Versionen:** DFC 3.33.0 ↔ HailoRT 4.23.0 ↔ Model Zoo v2.17. Der DFC läuft nur auf x86_64, nicht auf ARM.
- **Validität:** ein gemeinsames Preprocessing (`thesis/common/preprocess.py`) für alle Rechner. Feste Bildlisten in `data/subsets/` werden nie geändert, Kalibrierdaten sind disjunkt zur Eval-Liste. Auf dem Pi nur Rohdaten schreiben, Auswertung woanders. Code per `git pull` auf den Pi, jeder Lauf protokolliert Commit + dirty-Flag.
- **Nicht ins Git:** `artifacts/` (ONNX, HEF, HAR), `data/images/`, `*.npz`/`*.npy`, `docker/*.whl` (proprietäres DFC-Wheel), `*.log`.
- Zeilenenden LF (`.gitattributes`), Skripte in `scripts/` sind ausführbar.
- Keine Messwerte erfinden oder schätzen. Funktionstests (z. B. Imagenette, n = 20) klar von Ergebnissen für die Arbeit trennen.
