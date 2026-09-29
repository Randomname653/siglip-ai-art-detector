r"""sperrliste_bereinigen.py — daten/gesperrt_quellen.txt mit den Manifesten abgleichen.

Wird die Schutzregel gelockert (2026-09-26: Figuren-/Franchise-Liste nur noch fuer nicht
jugendfreie Bilder), kommen Bilder zurueck ins Lab. Ihre Pfade stehen aber noch in der
Sperrliste, und lesen.Pack wuerde sie weiter ausschliessen. Hier fliegt jeder Pfad aus der
Liste, der in einem aktuellen Manifest wieder als gueltige Datei gefuehrt wird. Pfade, die
in keinem Manifest mehr gueltig sind, bleiben gesperrt.

    python lab/detektor/sperrliste_bereinigen.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from lesen import DATEN                          # noqa: E402

FELDER = ("datei", "telegram", "original")


def main():
    liste = os.path.join(DATEN, "gesperrt_quellen.txt")
    gesperrt = [z.strip() for z in open(liste, encoding="utf-8") if z.strip()]
    gueltig = set()
    for mf in glob.glob(f"{DATASETS}/**/manifest.jsonl", recursive=True):
        if "_zum_loeschen" in mf:
            continue
        for z in open(mf, encoding="utf-8"):
            e = json.loads(z)
            if e.get("grund"):
                continue
            for f in FELDER:
                if e.get(f):
                    gueltig.add(e[f].lower())
    bleibt = sorted({p for p in gesperrt if p.lower() not in gueltig})
    zurueck = len(set(gesperrt)) - len(bleibt)
    with open(liste + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(bleibt) + "\n")
    os.replace(liste + ".tmp", liste)
    print(f"{len(set(gesperrt))} Eintraege, {zurueck} wieder gueltig und entsperrt, {len(bleibt)} bleiben gesperrt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
