# Detektor-Labor — Arbeitsweise

KI-Bild-Detektor fuer Anime und digitale Kunst. Produktion (`imagesort.py`) bleibt
unangetastet; hier wird nur gemessen und trainiert. Datenquellen: `<DATASETS>\QUELLEN.md`.

## Modell

Eingefrorene Merkmale aus den **mittleren Schichten** von CLIP ViT-L/14 (Bloecke
7/11/15/19/23) und DINOv2-L (Schichten 8/12/16/20/24), darauf ein kleiner MLP-Kopf,
**fuenf Koepfe mit verschiedenen Seeds gemittelt** (`kopf_<tag>_s0..4.npz`). Jedes Bild
wird vorher gleich zugerichtet: Telegram-Simulation (1280 px, JPEG q87) beim Download,
dann `norm.normiere` (512 px, q95). Trainiert wird nur auf gestoerten Bildern
(`norm.stoere`), gemessen auf ungestoerten.

Die Schwelle kalibriert `train3.py` so, dass menschliche Danbooru-Kunst von 2026 (val,
Kuenstler getrennt vom Test) 1 % Fehlalarm hat; mit `--kalib-p2` zusaetzlich auf
menschlicher Nicht-Anime-Kunst.

## Einen Ordner pruefen (nur lesen)

    .venv\Scripts\python.exe lab\detektor\pruefe.py "<Ordner>"            # Kopf p8sx (Standard seit 28.09.; nur SigLIP 2, gleich gut wie drei Rueckgrate, doppelt so schnell)
    .venv\Scripts\python.exe lab\detektor\pruefe.py "<Ordner>" --kopf p8x # drei Rueckgrate; aeltere: p6x, v3

Ergebnis: CSV und HTML in `lab\detektor\pruefung\`. Nichts wird verschoben.

## Einen neuen Telegram-KI-Kanal aufnehmen

1. In Telegram Desktop exportieren: **JSON**, nur Fotos und Dateien.
2. `einsortieren_json.py "<Export>" <kurzname>` — kopiert nach
   `<DATASETS>\telegram_ki\<kurzname>`, ohne Vorschaubilder, ohne Bot-Posts, mit
   Tag-Sperre und Beschriftungs-Sperre (`schutz.py`). Abgebrochener Export:
   `result.json` auf die letzte vollstaendige Nachricht kuerzen und als dritten
   Parameter angeben (`result_repariert.json`).
3. `werbung.py <kurzname>` — Werbe-/Screenshot-Kandidaten; **Kontaktbogen ansehen**,
   dann `werbung.py <kurzname> --raus <ids>`. Eigene Kanal-Links: `--eigen <teil>`.
4. `kanaele_test.py <tag>` und `danbooru_test.py <tag>` messen ihn mit.
5. Zeile in `<DATASETS>\QUELLEN.md` ergaenzen.

Kanaele sind **Tests**, nie Training (sonst waeren die Zahlen nicht ehrlich).

## Neues Trainingsmaterial nachschieben (inkrementell)

    aufbau3.py --stamm v3zN --ohne v3z2,v3z3     # nur neue Bilder
    tags_pack.py v3zN                             # Rating + Schutz-Sperre
    embed.py --rueckgrat clip_mid --pack v3zN --aug   (und --nur-split test, und dino_mid)
    train3.py --zusatz v3z2,v3z3,v3zN --tag <neu> --rating-ausgleich

Ketten, die das automatisch machen: `v3_kette.sh`, `v31_kette.sh`, `p2_kette2.sh`
(Status in `daten/*_status.log`).

## Pflichtregeln

- **schutz.py** auf jede Web-/Telegram-Quelle: keine Darstellungen Minderjaehriger
  (Booru-Tags, WD-Tags ab 0,2, Beitragstexte gegen `schutz.FIGUREN`). Explizites
  Material ist erlaubt, solange es nicht darunter faellt.
- Keine Quellen, deren Abruf gegen AGB verstoesst (Cara, Newgrounds; keine
  Datensaetze aus solchen Scrapes).
- Beide Klassen muessen dieselbe Vorgeschichte haben (Kompression, Groesse, Zeitraum,
  Stil, Rating) — sonst lernt der Kopf die Abweichung statt der Herkunft. Bisher
  gefundene Fallen: Bildgroesse, Seitenverhaeltnis, Resampling, Vorschaubilder,
  Mensch-vor-2022 gegen KI-danach.
- Vergleiche nur zwischen Ensembles; einzelne Koepfe streuen um bis zu 0,04 AUC.
- Verschieden kalibrierte Koepfe bei GLEICHEM Fehlalarm vergleichen: `gleich_fa.py v3 p2 ...`.
- Schwelle eines neuen Kopfes: `embed.py --pack p2z --nur-split val` (einmalig, ungestoert), dann
  `kalibriere.py <tag>` (Danbooru-Anime, Wallhaven, Danbooru-Nicht-Anime gleich gewichtet, 1 %).
- Nach jedem neuen Kopf: `kontrolle3.py <tag>` (Massstab-Eingriff, Robustheit).
