<!-- Eingefrorene Kopie der Vault-Notiz '1 Forschung/Auswertungsplan.md' (Obsidian), Stand updated: 2026-11-16.
     Kopiert am 2026-11-24 vor Sichtung der Kern-Daten. SHA-256 der Vault-Datei: 448b680eb51665bbf88a0dc6fae84c7b4546c8e3378dfe14728f9f415916057c
     Nicht bearbeiten. Änderungen am Plan nach diesem Datum offen als Abweichung berichten. -->

---
tags: [bachelorarbeit, methodik, auswertungsplan]
status: aktiv
date: 2026-11-03
updated: 2026-11-16
---

# Auswertungsplan (Kern, Phase 2)

Zugehörig: [[🏠 Bachelorarbeit - Home]] · [[Versuchsaufbau#Vorgehen (entschieden 19.10.2026)]] · [[Methodik]] · [[Phase 0 – Defaults]] · Grundlage: [[E2 Phase 1 Voraussetzungen]] · Logbuch: [[2026-11-03]]

> **Status: Entwurf (Vorschlag Claude, 03.11.2026).** ✅ **Entschieden 03.11.2026 (Daniel):** k = 10 · Umfang Accuracy 48 000 / CAM 5 000 geschichtet (unter der Bedingung tragfähiger Begründung → 2.2) · Grad-CAM-Hauptvariante reiner Emulator · Kalibriergröße erst nach einem Test (→ [[Phase 0 – Defaults#Zusammenführung (am Ende von Phase 0)]]). Alles Übrige ist Vorschlag. Zweck: Messgrößen, Aggregation und Tests **festlegen, bevor die Kern-Daten angesehen werden** ([[Versuchsaufbau#Vorgehen (entschieden 19.10.2026)]]: „Metriken und Aggregation erst nach den Ergebnissen zu wählen, ist im Kolloquium angreifbar“). Die Zahlen aus Phase 1 (n = 200) dienen nur zur Planung, nicht als Ergebnis.
>
> ⚠️ Der genaue Wortlaut von TF1–TF4 steht im Exposé-Repo (`thesis-expose`), nicht im Vault. Die Zuordnung unten ist aus [[Exposé V2 Review]] (A4, A7) und [[Thema & Forschungsfrage]] rekonstruiert → gegen das Exposé prüfen.

## 1. Was der Kern beantworten soll

> ⚠️ **Abgleich mit dem Exposé V2 (10.11.2026, Claude):** Die rekonstruierte Tabelle unten stimmt bei **TF1 nicht**. Wortlaut aus `Thesis Expose/expose-v2/main.tex` (Kap. 1, Commit `7da5ac7` „[V2] submission“):
> - **FF:** „Verschiebt hardware-reale Quantisierung auf einer Edge-NPU nicht nur die Accuracy, sondern auch den visuellen Fokus eines Vision-Classifiers und wie verhält sich der hardware-real gemessene Drift zu dem in Software-Simulation vorhergesagten?“
> - **TF1:** „Lassen sich CAM-Verfahren auf einer kommerziellen Edge-NPU hardware-real berechnen, und welchen Durchsatz kostet der dafür nötige Zugriff auf Zwischenaktivierungen?“
> - **TF2:** „Wie groß ist die Abweichung zwischen Schattenmodell und realer NPU auf Ebene der Zwischenaktivierungen, und wie stark schlägt sie in die daraus berechneten CAMs durch?“
> - **TF3:** „Wie hängt die Fokusverschiebung mit dem Accuracy-Einbruch zusammen?“
> - **TF4** *(optional)*: „Treiben dieselben Schichten, die im Performance-Profiling als Bottleneck auftreten, auch die Fokusverschiebung?“
>
> Der Fokus-Effekt FP32 ↔ INT8 ist also **Teil 1 der Forschungsfrage**, nicht TF1. Lücken und Abweichungen → Abschnitt 1.1. Tabelle unten bleibt als Verlauf stehen.


| TF (rekonstruiert) | Frage im Kern | Hauptgröße |
|---|---|---|
| TF1 | Verschiebt hardware-reales INT8 den visuellen Fokus, und wie stark je Modell? | CAM-Ähnlichkeit FP32 ↔ INT8 je Bild (Eigen-CAM vom Chip, Grad-CAM aus dem Emulator) |
| TF2 | Wie weit weicht das Schattenmodell (Emulator) vom Chip ab, gemessen in CAM-Metriken? | Eigen-CAM-Ähnlichkeit Emulator ↔ Chip desselben Compiles (Grad-CAM auf dem Chip nicht berechenbar → [[Exposé V2 Review]] A4) |
| TF3 | Wie hängt die Fokusverschiebung mit dem Accuracy-Einbruch zusammen? (Zusammenhangs-Beobachtung, keine Korrelation über Stufen → A7) | je Bild: Ausgabeabweichung vs. CAM-Ähnlichkeit; Bilder mit/ohne Vorhersagewechsel |
| TF4 | Layer-Zuordnung | nicht im Kern (Phase 3, P6/P7/P9b) |

### 1.1 Abgleich Exposé V2 ↔ Plan (10.11.2026, Vorschläge Claude, ❓ Entscheidung Daniel)

> ✅ **Entschieden 10.11.2026 (Daniel):** alle Vorschläge der Tabelle übernommen: (1) TF1-Durchsatz über HEFs nur mit Logits (`configs/*_int8_lvl2_logits.yaml`, angelegt 10.11., noch nicht committet) gegen die HEFs mit Feature-Map, `hailortcli benchmark` + Host-Latenz; (2) TF2 auf Aktivierungsebene (relativer Fehler, Cosinus, Anteil bitgleicher Werte der dequantisierten Feature-Map Emuᵢ ↔ Chipᵢ) + Zusammenhang mit der CAM-Abweichung (Spearman); (3) FF Teil 2: Eigen-CAM IoU(FP32, Chipᵢ) gegen IoU(FP32, Emuᵢ) je Bild gepaart (Differenz, Korrelation); Top-5-Accuracy, Latenz, Durchsatz als Zusatzgrößen. Die Abweichungen vom Exposé werden in der Arbeit begründet. Überschrift bleibt wegen Links.

| Exposé | Stand im Plan | Bewertung / Vorschlag |
|---|---|---|
| FF Teil 1: Fokusverschiebung FP32 ↔ INT8 | Effekt FP32 ↔ Chipᵢ/Emuᵢ (Abschn. 4, 5) | ✅ abgedeckt (hier als „TF1“ geführt → umbenennen in „FF-1“) |
| FF Teil 2: hardware-realer Drift ↔ „in Software-Simulation vorhergesagter“ | nur Emuᵢ ↔ Chipᵢ (TF2) | ⚠️ **Lücke:** Drift gegen Drift fehlt explizit → Vorschlag: Eigen-CAM IoU(FP32, Chipᵢ) gegen IoU(FP32, Emuᵢ) je Bild (gepaart), daraus „Simulation sagt Drift vorher?“ (Differenz, Korrelation). Daten fallen ohnehin an. ⚠️ „Simulation“ ist jetzt der DFC-Emulator (Entscheidung 20.10.), im Exposé eine PyTorch-Fake-Quant-Pipeline → Abweichung begründen (Stufe 2 ändert Gewichte, Emulator = bestmögliche Simulation desselben Compiles); unabhängige Fake-Quant wie [[E0 Simulation (Prototyp)]] höchstens qualitativ / Phase 3 |
| TF1 Teil 1: CAM hardware-real berechenbar? | Eigen-CAM vom Chip läuft (M2 ✅, [[E1 AP0 End-to-End]]) | ✅ beantwortet, im Kern nur noch berichten |
| TF1 Teil 2: **Durchsatzkosten** des Zugriffs auf Zwischenaktivierungen | ✅ **gemessen 16.11.2026** → [[E3 TF1 Durchsatzkosten Feature-Ausgang]] (kein messbarer Verlust) · ~~fehlt im Kern~~ | ⚠️ **Lücke:** Vorschlag: je Modell ein HEF **nur mit Logits** und eines mit Logits + Feature-Map (aus demselben Compile-Setup), `hailortcli benchmark` (FPS) + Host-Latenz, je mehrere Wiederholungen. Billig (2 zusätzliche Compiles). Hinweis E1: PCIe begrenzt rechnerisch nicht, Kosten eher durch aufgebrochene Fusion |
| TF2: Abweichung Schattenmodell ↔ NPU **auf Ebene der Zwischenaktivierungen** + Durchschlag in CAMs | nur CAM-Ebene (Eigen-CAM Emuᵢ ↔ Chipᵢ) | ⚠️ **Lücke:** Vorschlag Aktivierungs-Maß je Bild auf der dequantisierten Feature-Map: relativer Fehler ‖A_emu − A_chip‖/‖A_chip‖ und Cosinus, dazu Anteil bitgleicher Werte; dann Zusammenhang Aktivierungs-Abweichung ↔ CAM-Abweichung (Spearman) |
| TF3 | Abschn. 7 | ✅ deckungsgleich |
| TF4 | Phase 3 | ✅ (Exposé: optional) |
| Methodik: Grad-CAM **hybrid** (Chip-Aktivierungen + Schattenmodell-Gradienten) als Hauptweg | Hauptvariante reiner Emulator, Hybrid Nebenvariante (03.11.) | ⚠️ **Abweichung** → in der Arbeit begründen (Konsistenz von A und Gradienten, Gradientenpfad gegen PyTorch bestätigt) |
| Methodik: Schattenmodell = PyTorch-Pipeline | DFC-Emulator | ⚠️ Abweichung, begründet 19./20.10. (ab Stufe 2 ändert der DFC die Gewichte) |
| Rauschen über **Mehrfachläufe** (Chip nicht deterministisch) | Chip deterministisch (P2); Zufall liegt im **Compile** → k = 10 | ⚠️ Abweichung mit Befund begründen (P2/P3) |
| n: 1000 (Accuracy/Durchsatz), 250 (CAM) je Konfiguration | 48 000 / 5 000 | ✅ Abweichung nach oben, begründet (2.2) |
| Metriken: CC, KLD, Top-k-IoU, Spearman gleichrangig | Hauptmaß IoU, CC zweit, Spearman Kollaps, KLD Anhang (09.11.) | ✅ Gewichtung neu, alle vier bleiben erhalten |
| Zusatzgrößen: Top-1/**Top-5**-Accuracy, Vorhersage-Äquivalenz, **Latenz, Durchsatz** | Top-1, Logit-Maße | ⚠️ Top-5 und Latenz/Durchsatz ergänzen (Top-5 steht schon in `predictions.csv`) |
| Zielklasse FP32-Top-1 + Teilmenge gleicher Vorhersagen | entschieden 09.11. | ✅ deckungsgleich |

## 2. Versuchsplan des Kerns

| Faktor | Stufen | Status |
|---|---|---|
| Modell | ResNet18 (`IMAGENET1K_V1`), MobileNetV3-Large (V1) | ✅ entschieden (E-B, 20.10.) |
| Präzision | FP32 (PyTorch, CPU) · INT8 Stufe 2 (Chip + Emulator) | ✅ (E-A, 20./21.10.) |
| Compiles je Modell | **k** unabhängige Compiles derselben Config | ✅ **k = 10** (Daniel, 03.11.2026; Bauer: „eher 5 oder 10“). Compiles und Emulator-Läufe werden über Nacht bzw. während Vorlesungen gebündelt. Vorschlag Claude war 5 |
| Bilder | Eval-Liste (disjunkt zur Kalibrierung) | ✅ Accuracy 48 000, CAM 5 000 geschichtet (Daniel, 03.11.2026, Bedingung: Begründung → 2.2) |
| Zielklasse Grad-CAM | FP32-Top-1 | ✅ **entschieden 09.11.2026 (Daniel)**: FP32-Top-1 für alle Bilder (Hauptanalyse) + strenge Teilmenge gleicher Vorhersagen als Ergänzung (→ Abschnitt 7). Verworfen: eigene Top-1 je Modell (vermischt Klassenwechsel und Erklärungsänderung) · Ground Truth (andere Frage; FP32 bei ~30 % falsch) · nur Teilmenge (Selektionseffekt) · Teilmenge je Compile (passt nicht zur Aggregation je Bild über Compiles) |
| Kalibriergröße | **64 (DFC-Default)** | ✅ entschieden 09.11.2026 (Daniel) nach Test → [[E2 Phase 1 Voraussetzungen#Kalibrier-Test 64 vs. 1024 (09.11.2026, Auswertung Claude)]]. Folge: Sättigung bleibt → Umgang hier festlegen |
| ~~Kalibriergröße,~~ Kompilierstufe | → [[Phase 0 – Defaults#Zusammenführung (am Ende von Phase 0)]] | ❓ offen (im HEF eingebacken → **vor** den Kern-Compiles entscheiden) |

### 2.1 Umfang (n)

Kosten (gemessen bzw. hochgerechnet aus Phase 1, ⚠️ Hochrechnungen):
- Chip: ≈ 9 ms/Bild (MobileNetV3, Host-Latenz) → 48 000 Bilder ≈ 7–8 min je Lauf → billig.
- ✅ **Gemessen 16.11.2026** (Windows-PC, RTX 3070, 200 Bilder, Konsolenausgabe von Daniel; Läufe `results/20261116-17*_6ce6c0168cbf_*_emu_*`, nicht committet): Emulator **mit Gradienten** MobileNetV3 nativ 1,6 Bilder/s, **quantisiert 1,1 Bilder/s** (beide `--batch-size 2`); ResNet18 nativ 14,6, **quantisiert 15,5 Bilder/s** (Standard-Batch). Vorwärtslauf allein 14,6–84 Bilder/s (vernachlässigbar). → je Compile quantisiert: MobileNetV3 5 000 Bilder ≈ 1,3 h, 48 000 ≈ 12–13 h; ResNet18 5 000 ≈ 6 min, 48 000 ≈ 1 h.
  - ✅ **Nachtrag 16.11.2026: `--batch-size 4` läuft** (kein Speicherfehler): MobileNetV3 quantisiert **2,1 Bilder/s** (nativ 3,0) → 5 000 ≈ 40 min, 48 000 ≈ 6,4 h je Compile. Top-1/Top-5 auf allen 200 Bildern identisch zu Batch 2, Gradienten-Kennwerte in `checks.json` identisch (Mittel |∂|, Anteil ≠ 0); ✅ **Quantisierter Kontext bitgleich** zu Batch 2 (Logits, Features, Gradienten, alle 200 Bilder, geprüft 16.11.); nativer Kontext minimal abweichend (Gleitkomma, rel. ≤ 3·10⁻², im Kern nicht genutzt). → **Batch 4 für alle MobileNetV3-Emulator-Läufe im Kern** (Daniel, 16.11.2026). Gesamtzeit Option 3 damit ≈ 13 h (MobileNetV3) + 2 h (ResNet18).
- ~~Emulator mit Gradienten: nativ MobileNetV3 ≈ 5,8 Bilder/s (Batch 8); quantisiert langsamer und nur mit Batch 2 (GPU-Speicher). 48 000 Bilder ≈ ≥ 2,3 h **je Compile** → bei k = 5 × 2 Modelle ≈ ≥ 25 h.~~ (Hochrechnung vom 03.11., durch Messung ersetzt)
- Speicher: `features.npz` FP32-MobileNetV3 ≈ 31 MB je 200 Bilder → ≈ 7 GB bei 48 000; INT8 ≈ 7 MB je 200 → ≈ 1,7 GB je Lauf; Gradienten ähnlich. Bei k = 5 und allen Läufen auf 48 000: grob 30–50 GB.

**Vorschlag:**
- **Accuracy und Logits:** alle **48 000** Eval-Bilder (Chip, Emulator ohne Gradienten, FP32). Top-1 ±0,2 Pp. (1σ).
- **CAM-Metriken (Eigen-CAM, Grad-CAM):** feste, **klassengeschichtete Teilmenge von 5 000** Bildern (5 je Klasse, fester Seed) aus der Eval-Liste, als neue Liste `imagenet_eval_strat5000_seed2.txt` (bestehende Listen bleiben unverändert). Begründung: Bei gepaarten Vergleichen je Bild ist n = 5 000 für Mediane und P5 reichlich (Bootstrap-KI wird schmal), hält aber die Emulator-Zeit bei ≈ 1/10.
- Alternative: alles auf 48 000 (≈ 10× Emulator-Zeit und Speicher) · alles auf 5 000 (Accuracy dann ±0,65 Pp., schlechter vergleichbar mit torchvision-/Zoo-Werten).
- ✅ Liste erzeugt: `data/subsets/imagenet_eval_strat5000_seed2.txt` (Commit `d7d2391`; 1000 Klassen × 5, Seed 2, Teilmenge von `imagenet_eval_48000.txt`, disjunkt zur Kalibrierliste, geprüft).

### 2.2 Begründung für 5 000 Bilder bei den CAM-Metriken (für den Methodikteil)

Daniel will die Teilmenge nur, wenn sie sich in der Arbeit gut begründen lässt. Argumente:

1. **Präzision reicht für die Aussagen, die der Kern macht** (Planungsrechnung Claude aus Phase 1, ⚠️ Bootstrap aus nur 200 Werten je Modell, daher grob, v. a. bei Quantilen): Die Differenz e − s (Effekt minus Compile-Streuung, Eigen-CAM-IoU, je Bild) hatte in Phase 1 den Median −0,006 (ResNet18) bzw. −0,014 (MobileNetV3). Die Breite des 95 %-Bootstrap-KI dieses Medians wäre bei n = 5 000 etwa 0,0006 bzw. 0,006, bei n = 10 000 0,0004 bzw. 0,003. → Bei 5 000 Bildern ist das KI bei ResNet18 rund zehnmal schmaler als der Effekt, bei MobileNetV3 gut halb so breit wie der Effekt. Die Aussage „Effekt größer als die Compile-Streuung?“ ist damit bei beiden Modellen entscheidbar.
2. **Schichtung statt Zufallsauswahl:** genau 5 Bilder je Klasse, alle 1 000 Klassen gleich gewichtet. Die bisherigen „ersten N“ der Eval-Liste sind unausgewogen (erste 200 Bilder: 181 Klassen).
3. **Einordnung in die Literatur:** Kerkouri et al. werten auf **10 000 zufälligen** ImageNet-Val-Bildern aus ([[Quantization Effects on Neural Networks Perception]]), Quant-XAI auf **20** Bildern ([[Quant-XAI 2026 (Preprint)]]). 5 000 liegt in derselben Größenordnung wie Kerkouri, und dafür mit **k = 10 Compiles**, also 50 000 Bild-Compile-Paare je Modell.
4. **Kosten begründen die Wahl, nicht die Ergebnisse:** Der Emulator mit Gradienten ist der Engpass (MobileNetV3 quantisiert nur mit Batch 2). Die Wahl fällt vor den Kern-Daten.
5. ✅ **Entschieden 16.11.2026 (Daniel) – Umfang und Repräsentativitäts-Check:**
   - **Eigen-CAM FP32 ↔ Chip auf allen 48 000 Bildern, alle k = 10 Compiles** (Chip-Features fallen bei den Accuracy-Läufen ohnehin an; zusätzlich FP32-Features auf 48 000 ≈ 10 GB, Chip-Features ≈ 26 GB).
   - **Grad-CAM (Emulator) auf der 5 000er-Teilmenge für alle 10 Compiles**, zusätzlich **für einen Compile je Modell auf allen 48 000** (Repräsentativitäts-Check direkt für Grad-CAM, Ränder der Verteilung). Alles mit Emulator (TF2, FF Teil 2) und Vergleiche Eigen- ↔ Grad-CAM auf der Teilmenge.
   - **Kriterium (vorab):** Teilmenge repräsentativ, wenn Median, P5 und Anteil gebrochener Karten auf 48 000 je Modell und Verfahren im 95 %-Bootstrap-KI der Teilmenge liegen; sonst als Einschränkung berichten.
   - Rechenzeit Emulator (gemessen 16.11.): Batch 2 ≈ 26 h MobileNetV3 + 2 h ResNet18; **Batch 4 ≈ 13 h + 2 h**. Verworfen: Grad-CAM auf 48 000 für alle Compiles (≈ 6 Tage) · nur 5 000 ohne Check · 10 000 für alle (kein direkter Repräsentativitätsnachweis).
   - Ursprünglicher Vorschlag: **Repräsentativitäts-Check (Vorschlag, billig):** Eigen-CAM braucht keine Gradienten. Für **einen** Compile je Modell FP32 ↔ Chip auf allen 48 000 rechnen und die Verteilung (Median, P5, Anteil gebrochener Karten) mit der 5 000er-Teilmenge vergleichen. Liegen die Kennzahlen innerhalb der Bootstrap-KI, ist die Teilmenge nachweislich repräsentativ. Kosten: Chip ≈ 8 min, FP32-Features ≈ 7 GB (MobileNetV3) + 2,8 GB (ResNet18) auf dem Mac.

Ehrliche Grenze: Für die **Ränder** der Verteilung (P5, Minimum, seltene Kollapsfälle) ist mehr n immer besser. Wer diese Ränder als Hauptaussage braucht, müsste auf 10 000 gehen (Kerkouri-Größe, ≈ doppelte Emulator-Zeit).

## 3. Messgrößen je Bild

**CAM-Erzeugung** (fest, Code in `thesis.analysis`): Eigen-CAM = erste Hauptkomponente der Feature-Map, Vorzeichen an FP32 ausgerichtet; Grad-CAM = ReLU(Σ αₖ·Aₖ); beide bilinear auf 224×224, auf [0, 1] normiert. Zielschicht = letzte Conv-Stufe ([[Methodik#Zielschicht für Grad-CAM und Eigen-CAM]]).

| Rolle | Maß | Begründung |
|---|---|---|
| **Hauptmaß** ✅ entschieden 09.11.2026 (Daniel, → 3.1) | **Top-15 %-IoU** (Robustheit: Top-20 %-IoU) | misst direkt die Überlappung der Fokusregion („wohin schaut das Modell“); Vorlage Quant-XAI ([[Quant-XAI 2026 (Preprint)]]); interpretierbarer Zufallsboden k/(2−k) ≈ 0,081 ([[Exposé V2 Review]] B2) |
| Zweitmaß ✅ 09.11. | Pearson-CC | Vorlage Kerkouri ([[Quantization Effects on Neural Networks Perception]]); misst die ganze Karte, aber sättigt nahe 1 (Phase 1: Median 0,99–1,00) und trennt schlecht |
| Kollaps-Indikator ✅ 09.11. / Anhang | Spearman (Kollaps) / KLD (nur Anhang) | Spearman ≈ 0 markiert Kollaps (Quant-XAI); KLD wie Kerkouri |
| Ausgabe (für TF3) | Logit-Cosinus, Top-1-Wechsel, ΔProbability der FP32-Klasse | stetige Ausgabeabweichung auch ohne Vorhersagewechsel (ResNet18: 97 % gleiche Top-1) |

- ⚠️ Einschränkung im Methodikteil nennen: effektive Auflösung 7×7 (Faktor 32 bei beiden Modellen, Zielschicht letzte Conv-Stufe) → IoU springt grob, Verschiebungen innerhalb einer Zelle sieht nur CC. 14×14 (eine Stufe früher) höchstens als Phase-3-Achse.
- ⚠️ Begründung für IoU statt CC als Hauptmaß ist eigene Abwägung aus den Phase-1-Zahlen. Das ist genau die Art Entscheidung „nach Datenlage“, die der Plan vermeiden soll → in der Arbeit offen sagen, dass Phase-1-Funktionstests (n = 200, nicht Teil der Kern-Daten) die Wahl informiert haben.

### 3.1 Hauptmaß: Optionen (Zusammenstellung Claude, 09.11.2026)

**Was die Referenzpaper nutzen** (CAM(INT8) gegen CAM(FP32)):

| Paper | Maße | Anmerkung |
|---|---|---|
| [[Quant-XAI 2026 (Preprint)]] | Top-k-IoU (k = 15 %), Top-k-Dice, Spearman | Hauptzahl 0,126 ist Top-15 %-IoU |
| [[Same Predictions Different Reasons 2026 (Preprint)]] | PCC, SSIM, Top-20 %-IoU, Deletion/Insertion | nur Gewichte quantisiert |
| [[Quantization Effects on Neural Networks Perception]] | SIM, CC, KLD (Grad-CAM++) | SIM zeigt dort selbst ein Artefakt |

Saliency-Literatur allgemein: Bylinskii et al. 2019 (TPAMI) vergleichen diese Maße systematisch.

**Phase-1-Daten** (FP32 ↔ Emulator quantisiert, Compile A, n = 200 bzw. 196/197 ohne leere Grad-CAM; nur zur Planung, Rechnung Claude 09.11.2026 mit `thesis.analysis`):

| | Top-15 %-IoU Median / P5 | CC Median / P5 | Spearman Median / P5 |
|---|---|---|---|
| ResNet18 Eigen-CAM | 0,963 / 0,798 | 1,000 / 0,990 | 0,999 / 0,984 |
| ResNet18 Grad-CAM | 0,956 / 0,859 | 0,999 / 0,995 | 0,999 / 0,995 |
| MobileNetV3 Eigen-CAM | 0,888 / 0,473 | 0,993 / 0,856 | 0,987 / 0,897 |
| MobileNetV3 Grad-CAM | 0,755 / 0,400 | 0,964 / 0,800 | 0,934 / 0,776 |

Rangkorrelation der Maße über die Bilder: IoU ↔ CC 0,72–0,84, IoU ↔ Spearman 0,34–0,66, CC ↔ Spearman 0,51–0,90. → ⚠️ **Korrektur zur Zeile „Zweitmaß“ oben:** CC liegt nahe 1, ordnet die Bilder aber ähnlich wie IoU. „Trennt schlecht“ betrifft die Skala (schwer interpretierbar), weniger die Rangfolge. Spearman misst etwas anderes (Ordnung des ganzen Bildes, vor allem Hintergrund).

**Optionen** (Vor-/Nachteile eigene Abwägung, ⚠️ nicht aus der Literatur übernommen, wo nicht anders angegeben):

| Maß | Vorteile | Nachteile |
|---|---|---|
| **Top-k-IoU** | direkt „wohin schaut das Modell“; Zufallsboden bekannt (k/(2−k) ≈ 0,081 bei 15 %); unempfindlich gegen Normierung/Skalierung; vergleichbar mit Quant-XAI (k = 15) und Same Predictions (k = 20) | k willkürlich; ignoriert alles außerhalb der Fokusregion; bei flachen/kollabierten Karten schlecht definiert (Gleichstände); effektive Auflösung nur 7×7 = 49 Zellen (15 % ≈ 7 Zellen) |
| **Pearson-CC** | ganze Karte, kein Schwellwert; Standard der Saliency-Literatur; vergleichbar mit Kerkouri und Same Predictions | Werte nahe 1, schwer interpretierbar; von glatter Struktur der hochskalierten Karte dominiert; linear, ausreißerempfindlich |
| **Spearman** | rangbasiert; zeigt Kollaps (ρ ≈ 0) direkt; vergleichbar mit Quant-XAI | stark vom Hintergrund bestimmt; Gleichstände bei flachen Karten |
| **KLD** | Standard (Kerkouri) | asymmetrisch, unbeschränkt, empfindlich für Werte nahe 0 (ε), kaum interpretierbar |
| **SSIM** | strukturell (Same Predictions) | für natürliche Bilder entworfen, Fensterparameter; auf hochskalierten 7×7-Karten fraglich |
| **SIM** | – | Artefakt laut Kerkouri selbst |
| **Dice** | – | monotone Funktion von IoU (Dice = 2·IoU/(1+IoU)), redundant |
| **Deletion/Insertion** | misst Faithfulness (ist die Erklärung kausal richtig?) | andere Frage als „hat sich der Fokus verschoben“; viele zusätzliche Forward-Pässe |

**Vorschlag (Claude):** Hauptmaß **Top-15 %-IoU** (Vergleichbarkeit mit der nächsten Arbeit, interpretierbar, Zufallsboden) · Robustheit **Top-20 %-IoU** (Vergleich Same Predictions) · Zweitmaß **CC** (ganze Karte, Vergleich Kerkouri/Same Predictions) · Kollaps-Indikator **Spearman** · KLD nur Anhang · SSIM, SIM, Dice nicht · Deletion/Insertion höchstens Phase 3. ✅ **entschieden 09.11.2026 (Daniel)**, Vorschlag unverändert übernommen.

## 4. Vergleiche

Mit FP32 = PyTorch-Referenz, Cᵢ = Compile i (i = 1…k), Chipᵢ = Pi-Lauf von Cᵢ, Emuᵢ = Emulator von Cᵢ:

| Vergleich | Eigen-CAM | Grad-CAM | dient |
|---|---|---|---|
| **Effekt:** FP32 ↔ Chipᵢ | ✅ (hardware-real) | – | TF1 |
| **Effekt:** FP32 ↔ Emuᵢ | ✅ | ✅ (Hauptvariante, Vorschlag) | TF1 (Grad-CAM) |
| **Compile-Streuung:** Chipᵢ ↔ Chipⱼ, Emuᵢ ↔ Emuⱼ (i < j) | ✅ | ✅ | Referenz für TF1 |
| **Emulator-Lücke:** Emuᵢ ↔ Chipᵢ | ✅ | – | TF2 |
| Hybrid-Grad-CAM (A vom Chipᵢ, Gradienten aus Emuᵢ) vs. FP32 | – | ✅ (Nebenvariante) | TF2-Diskussion |

**Grad-CAM-Hauptvariante: reiner Emulator** (✅ entschieden 03.11.2026, Daniel). Begründung: in sich konsistent (A und Gradienten aus demselben Modell), Gradientenpfad gegen PyTorch bestätigt ([[E2 Phase 1 Voraussetzungen#Grad-CAM: PyTorch gegen Emulator (03.11.2026, Auswertung Claude)]]). Die Hybrid-Variante mischt Chip-Aktivierungen mit Gradienten eines anderen (Emulator-)Modells; sie wird als Nebenvariante berichtet, weil sie näher am „Gerät“ ist. ❓ offen

## 5. Aggregation und Umgang mit dem Compile-Zufall

Je Bild zuerst über die Compiles mitteln, dann über Bilder:
- **Effekt je Bild:** e = Mittel über i von IoU(FP32, Xᵢ)
- **Streuung je Bild:** s = Mittel über i < j von IoU(Xᵢ, Xⱼ)
- Berichtet je Modell und CAM-Verfahren: **Median, P5 und Minimum** von e (Median *und* schlechteste Bilder, wie 19.10. festgelegt), dieselben Kennzahlen für s, dazu die Spannweite der Mediane über die k Compiles (zeigt, wie stark ein einzelner Compile das Ergebnis verschiebt).
- **Anteil „gebrochener“ Karten** (`is_broken`: Kollaps, Spearman < 0,2, CC ≤ 0, IoU ≤ 1,05 · 0,15) je Variante.

**Interpretation von s (wichtig, aus [[2026-11-03]]):** s ist kein reiner „Zufallsboden“ für e. In s steckt nur der compile-abhängige Anteil (zweimal), in e der systematische plus der compile-abhängige Anteil (einmal). Ist e ≈ s, ist die Abweichung überwiegend compile-abhängig; ist e deutlich unter s, gibt es einen systematischen Anteil. Ergänzend (Vorschlag): **Konsens-Vergleich** – CAM aus den über die k Compiles gemittelten Feature-Maps (bzw. Gradienten) gegen FP32; mittelt den compile-abhängigen Anteil heraus und schätzt den systematischen. ⚠️ eigene Überlegung, nicht aus der Literatur → in der Arbeit als solche kennzeichnen.

✅ **Entschieden 09.11.2026 (Daniel):**
- **Hauptlogik: e gegen s.** Nullhypothese „kein systematischer Anteil“ = jeder Compile ist FP32 plus unabhängiges Rauschen. Dann steckt das Rauschen in e einmal, in s zweimal → unter H₀ gilt e ≥ s (näherungsweise; IoU ist nichtlinear). **e < s spricht für einen systematischen Anteil.** ⚠️ eigene Herleitung, in der Arbeit als solche kennzeichnen.
- **Ergänzung: Konsens-Vergleich** zur Größenschätzung des systematischen Anteils: **Karten** der k Compiles mitteln (nicht Feature-Maps; für Eigen- und Grad-CAM gleich), Konsens-Karte gegen FP32. ⚠️ eigenes Konstrukt, entspricht keinem realen Modell; Restrauschen bei endlichem k.
- **Verworfen:** Varianzzerlegung (gemischtes Modell): beantwortet die Schwankung zwischen Compiles, nicht die Größe der Verschiebung; Aufwand hoch. Konsens über Feature-Maps: methodenabhängig, in Phase 1 gleiches Ergebnis.
- **Phase 1 zur Planung** (Eigen-CAM Chip, Top-15 %-IoU, n = 200, k = 2, Rechnung Claude 09.11.): ResNet18 e 0,963 · s 0,971 · Konsens 0,967; MobileNetV3 e 0,885 · s 0,916 · Konsens 0,892 (Features und Karten gleich); e < s bei 62 % bzw. 66 % der Bilder. → Bei MobileNetV3 deutet alles auf einen überwiegend **systematischen** Anteil (Mitteln bringt kaum etwas). ⚠️ Korrigiert die Einordnung vom 03.11. („Effekt kaum über dem Compile-Zufall“), dort war s als Boden behandelt.

Zwei Böden zur Einordnung: **Zufallsboden** ≈ 0,081 (unabhängige Top-15 %-Masken) unten, **Compile-Streuung** s oben.

## 6. Tests

- **Effekt über der Compile-Streuung?** Gepaarter Wilcoxon-Vorzeichen-Rang-Test je Bild auf e − s (H₀: Median 0), je Modell und Verfahren; Effektstärke = Median(e − s) mit Bootstrap-95 %-KI (10 000 Ziehungen über Bilder). Bei n = 5 000 wird fast jeder Unterschied signifikant → **Effektstärke und KI sind die eigentliche Aussage**, der p-Wert nur Formalie.
- **ResNet18 vs. MobileNetV3:** Differenz der Mediane von e mit Bootstrap-KI (ungepaart über Modelle, gepaart über dieselben Bilder möglich, da gleiche Liste).
- **TF2:** Median/P5 von IoU(Emuᵢ, Chipᵢ) mit KI, relativ zu s.
- Mehrfachtests: wenige, vorab benannte Tests (je Modell × Verfahren) → Bonferroni über diese, Rest beschreibend.

## 7. TF3: Zusammenhang Fokus ↔ Ausgabe

- Je Bild und Modell: Spearman-Korrelation zwischen Ausgabeabweichung (1 − Logit-Cosinus, gemittelt über Compiles) und CAM-Abweichung (1 − e), mit Bootstrap-KI.
- Gruppenvergleich: Bilder **mit** Top-1-Wechsel (in mindestens einem Compile) vs. **ohne** → Verteilung von e je Gruppe.
- Auswertung zusätzlich auf der **Teilmenge gleicher Vorhersagen** (alle Compiles = FP32-Top-1), wie in [[Methodik#🔴 Offene Entscheidung: Zielklasse beim Grad-CAM-Vergleich]] vorgeschlagen. ✅ **entschieden 09.11.2026 (Daniel): strenge Definition** (alle k = 10 Compiles), Größe je Modell berichten. Phase 1 zur Planung: beide Compiles = FP32 bei ResNet18 96,5 %, MobileNetV3 86,0 % (n = 200; bei 10 Compiles kleiner, nicht hochgerechnet). ⚠️ Selektionseffekt benennen: Die Teilmenge enthält absichtlich die „einfachen“ Bilder; sie zeigt die Änderung **ohne** Vorhersagewechsel, nicht die Harmlosigkeit der Quantisierung.
- Über Modelle hinweg nur beschreibend (2 Punkte, keine Korrelation → A7).

## 8. Sonderfälle (vorab festgelegt)

| Fall | Umgang (Vorschlag) |
|---|---|
| **Logit-Sättigung** (Ziel-Logit am Ende des Ausgabebereichs, Phase 1: ~2 % der Bilder; STE-Gradient 0 → Grad-CAM leer) | ✅ **entschieden 09.11.2026 (Daniel):** (1) Ausschluss **nur aus den Grad-CAM-Metriken**, je Compile; Accuracy, Top-1 und Eigen-CAM behalten alle Bilder. (2) **Ausfallrate** (Anteil leerer Karten je Modell und Compile) als eigenes Ergebnis berichten (FP32 hat keine Ausfälle → Quantisierungseffekt). (3) **Sensitivitätsanalyse** mit leerer Karte = IoU 0. (4) Robustheitscheck auf der **Schnittmenge** (Bilder in keinem der k Compiles gesättigt). Selektionseffekt (höchste FP32-Logits) als Einschränkung nennen. Verworfen: leere Karte als IoU 0 im Hauptmaß (vermischt Ausfall mit Fokusverschiebung) · Bild aus der ganzen Analyse nehmen (entfernt ohne Grund die sichersten Bilder) · Kalibrierung 1024 (s. Kalibrier-Test). ~~Bild markieren, Anteil je Modell und Compile berichten. Grad-CAM-Metriken ohne leere Karten, **zusätzlich** Sensitivitätsanalyse (leere Karte als IoU = 0 gewertet). Eigen-CAM betrifft es nicht. Ursache (Ausgabebereich aus 64 Kalibrierbildern?) in P8 prüfen~~ |
| Eigen-CAM-Vorzeichen | an FP32 ausrichten (wie bisher), Flips zählen und berichten |
| Bild in einem Compile anders vorhergesagt | bleibt drin (Zielklasse ist FP32-Top-1), geht in TF3-Gruppen ein |
| Lauf mit `dirty: true` oder abweichendem HEF-Hash zwischen Emulator und Chip | nicht verwenden |

### Logit-Sättigung: Mechanismus und Literatur (Recherche Claude, 09.11.2026)

- **Mechanismus:** Der DFC legt den INT8-Bereich des Ausgangs per Kalibrierung fest (64 Bilder). Logits darüber werden auf den Höchstwert abgeschnitten (Clipping). Der STE gibt den Gradienten nur **innerhalb** des Bereichs durch, außerhalb ist er 0: Nagel et al. 2021, Gl. (37), Abschn. 4.1 → Ziel-Logit abgeschnitten ⇒ ∂y^c/∂A = 0 ⇒ Grad-CAM leer. Eigen-CAM nutzt nur die Feature-Map und ist nicht betroffen.
- **Zielkonflikt Bereich ↔ Auflösung** ist Lehrbuchstoff: „The key trade-off in range setting is between clipping and rounding error“ (Nagel et al. 2021, Abschn. 3.1, S. 9); Min-Max ist ausreißerempfindlich (ebd.). Der eigene Kalibrier-Test (64 vs. 1024) zeigt genau das ([[E2 Phase 1 Voraussetzungen#Kalibrier-Test 64 vs. 1024 (09.11.2026, Auswertung Claude)]]).
- **Hailos eigenes Paper** ([[Fighting Quantization Bias (Finkelstein 2019)]]) nutzt Min-Max auf einem **64-Bilder**-Kalibrierbatch und behandelt Rundungsrauschen und Clipping-Verzerrung gemeinsam als „Rauschen“ (S. 2 bzw. Abschn. Experimente, `1906.03193v1.pdf`). → stützt 64 als Herstellerpraxis.
- **Referenzpaper:** Keines berichtet Logit-Sättigung oder leere Grad-CAM-Karten (Volltextsuche nach saturat/clip/empty, 09.11.2026):
  - [[Quant-XAI 2026 (Preprint)]]: Fake-Quant mit MinMax-Kalibrierung, „rounding and clamping“ im Forward, STE im Backward → Sättigung wäre möglich, wird aber nicht erwähnt (ob der Ausgang selbst quantisiert ist, steht nicht da).
  - [[Same Predictions Different Reasons 2026 (Preprint)]]: **nur Gewichte** quantisiert (GPFQ) → Aktivierungen/Logits nicht abgeschnitten, Effekt dort ausgeschlossen.
  - [[Quantization Effects on Neural Networks Perception]]: „adaptive quantization“, Skala laut Beschreibung aus Min/Max des jeweiligen Eingabetensors → Bereich passt sich jedem Bild an, Clipping am Ausgang damit weitgehend ausgeschlossen (⚠️ falls die Beschreibung die Implementierung trifft).
  - Eigene Simulation [[E0 Simulation (Prototyp)]]: Aktivierungen auf den **Eval-Bildern selbst** kalibriert → auf diesen Bildern kein Clipping möglich. Erklärt, warum der Effekt dort nicht auftrat (eigene Schlussfolgerung).
- **XAI-Literatur allgemein:** Gesättigte Gradienten sind ein bekanntes Problem gradientenbasierter Attribution, aber dort durch sättigende Nichtlinearitäten, nicht durch Quantisierung: Sundararajan et al. 2017 (Integrated Gradients) motivieren IG damit. Grad-CAM nimmt bewusst den Score **vor** dem Softmax (Selvaraju et al. 2017), um Softmax-Sättigung zu vermeiden; das Clipping durch die Quantisierung trifft aber genau diesen Score.
- **Arbeit, die Quantisierungs-Clipping als Ursache leerer Grad-CAM-Karten beschreibt: nicht gefunden** (Websuche 09.11.2026, nicht erschöpfend) → potenziell eigener kleiner Befund (hardware-reale PTQ mit fester Kalibrierung vs. Simulationen mit adaptivem/auf Testdaten kalibriertem Bereich).
- ⚠️ **Selektionseffekt bei Ausschluss:** Betroffen sind die Bilder mit den **höchsten** FP32-Logits (28–37), also die sichersten Vorhersagen. Ausschluss verschiebt die Stichprobe leicht zu unsichereren Bildern → Sensitivitätsanalyse (Vorschlag oben) und Anteil berichten.
- Weitere denkbare Option (⚠️ Idee Claude, ungeprüft): im Emulator am letzten Quantisierer einen STE **ohne** Clipping-Maske verwenden → Grad-CAM auch bei gesättigtem Logit definiert, weicht aber vom realen Modell ab → höchstens als Nebenvariante.

## 9. Was nach Sichtung der Kern-Daten noch erlaubt ist

- Zusätzliche, **als explorativ gekennzeichnete** Auswertungen (z. B. andere k-Fraktion, Schwerpunkt-Distanz, Deletion/Insertion).
- Nicht erlaubt: Hauptmaß, Aggregation, Zielklasse oder Ausschlussregeln nachträglich ändern, ohne das offen zu berichten.

## Offene Entscheidungen (Daniel)

- [x] k ✅ 10 (Daniel, 03.11.2026)
- [x] Umfang n ✅ Accuracy 48 000, CAM 5 000 geschichtet (Daniel, 03.11.2026; Begründung 2.2 von Daniel noch abzunehmen)
- [x] Repräsentativitäts-Check ✅ Eigen-CAM Chip 48 000 × 10, Grad-CAM 5 000 × 10 + 48 000 × 1 je Modell (Daniel, 16.11.2026) → 2.2 Punkt 5
- [x] Hauptmaß ✅ Top-15 %-IoU, Robustheit Top-20 %, Zweitmaß CC, Kollaps Spearman, KLD Anhang (Daniel, 09.11.2026) → 3.1
- [x] Grad-CAM-Hauptvariante ✅ reiner Emulator, Hybrid als Nebenvariante (Daniel, 03.11.2026)
- [x] Zielklasse ✅ FP32-Top-1 + strenge Teilmenge (Daniel, 09.11.2026)
- [x] Umgang mit der Logit-Sättigung ✅ entschieden 09.11.2026 (Daniel), s. Tabelle Sonderfälle
- [x] Konsens-Vergleich ✅ als Ergänzung (Karten mitteln), Hauptlogik e gegen s (Daniel, 09.11.2026)
- [ ] TF-Wortlaut gegen das Exposé prüfen
- [x] **Umgang mit dem Emulator-Abstand bei MobileNetV3** ✅ **entschieden 09.11.2026 (Daniel):** Vorschlag unten übernommen → [[2026-11-09]]
  - Datenlage (Funktionstests, 200 Bilder, [[E2 Phase 1 Voraussetzungen]]): Emulator ↔ Chip Top-1 94,5 %, Eigen-CAM-IoU 0,937; Compile A ↔ B Top-1 90,5 %, IoU 0,915 → Emulator-Abstand etwas **kleiner** als die Compile-Streuung
  - **Vorschlag Claude (09.11.2026), von Daniel übernommen:** als Einschränkung berichten („Grad-CAM gilt für das emulierte Modell“) **und** im Kern Eigen-CAM Emulator ↔ Chip für alle k = 10 Compiles mitmessen, um den Abstand zu beziffern. Verworfene Alternative: Grad-CAM bei MobileNetV3 weglassen (schwächt die Arbeit; ohne Gradienten auf dem Chip gibt es keinen Weg ohne Emulator)
  - ⚠️ Annahme benennen: Abstand wird nur über Eigen-CAM gemessen; Grad-CAM reagiert empfindlicher (FP32 ↔ INT8: 0,755 vs. ≈ 0,89), Übertragbarkeit also nicht geprüft

## Quellen

- Bylinskii et al. 2019, What do different evaluation metrics tell us about saliency models? (IEEE TPAMI): https://arxiv.org/abs/1604.03605

- Nagel et al. 2021, A White Paper on Neural Network Quantization (arXiv, nicht peer-reviewed): https://arxiv.org/abs/2106.08295 (Abschn. 2.2, 3.1, 4.1 Gl. 36–37)
- Sundararajan et al. 2017, Axiomatic Attribution for Deep Networks (ICML): https://proceedings.mlr.press/v70/sundararajan17a.html
- Finkelstein et al. 2019 (Hailo): [[Fighting Quantization Bias (Finkelstein 2019)]]

- Selvaraju et al. 2017, Grad-CAM: [[Grad-CAM (Selvaraju 2017)]]
- Muhammad & Yeasin 2020, Eigen-CAM: [[Eigen-CAM (Muhammad 2020)]]
- Top-k-IoU, Spearman, Kollaps-Kriterium: [[Quant-XAI 2026 (Preprint)]] (Preprint, nicht peer-reviewed)
- CC, KLD: [[Quantization Effects on Neural Networks Perception]]
- Zufallsboden k/(2−k): [[Exposé V2 Review]] (B2, eigene Herleitung)
- Planungsrechnung 2.2: Ad-hoc-Rechnung von Claude am 03.11.2026 (Bootstrap, 2 000 Ziehungen, Eigen-CAM-IoU aus `thesis.analysis`), ⚠️ Code nicht im Repo gesichert. Daten: Chip-Läufe A/B und FP32-Referenzen in [[E2 Phase 1 Voraussetzungen]]
