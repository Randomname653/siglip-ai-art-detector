r"""nas_profikunst.py — professionelle Konzept- und Game-Art vom NAS als MENSCHLICHE TESTGRUPPE.

Quelle: die lokale Sammlung des Projekt-Eigentuemers, Wallpaper von CGWallpapers.com (Ordner "2.0", nach
Kuenstler) und GameWallpapers.com (Ordner "gw", offizielles Spiel-Artwork). Jedes Bild traegt am unteren Rand
eine Textzeile: Titel, (c) Kuenstler/Entwickler, bei CGWallpapers auch die verwendete Software.
Die Dateidaten sind 2025/26 (neu kopiert) -> kein Beleg "vor 2022" -> NUR Test, nie Training.

Je Bild:
  - Fusszeile (untere 5,5 %) per OCR lesen: Software/Entwickler ins Manifest; nennt sie ein KI-Werkzeug
    (Midjourney, Stable Diffusion, DALL-E, Firefly, Leonardo, NovelAI, "AI") -> verworfen
  - Fusszeile ABSCHNEIDEN (sonst waere das Wasserzeichen ein Merkmal "menschlich")
  - Foto-Filter und schutz.py wie ueberall; gespeichert als Telegram-Simulation (1280 px, q87 4:2:0)
Ordner mit Fan-Art werden ausgelassen (Herkunft unklar).

    python lab/detektor/nas_profikunst.py
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS, NAS                  # noqa: E402
from civitai import REAL_MIN, hochladen_bytes    # noqa: E402
from schutz import SCHWELLE, figurenregel_gilt, gesperrt_text, gesperrt_wd   # noqa: E402

QUELLEN = {"2.0": "cgwallpapers", "gw": "gamewallpapers"}
ZIEL = f"{DATASETS}/nas_profikunst"
FUSS = 0.055
KI_WORTE = re.compile(r"midjourney|stable ?diffusion|dall[- ]?e|firefly|leonardo|novel ?ai|\bai\b|"
                      r"generat|neural|diffusion|comfy|automatic1111", re.I)


def main():
    import easyocr
    from imgutils.tagging import get_wd14_tags
    from imgutils.validate import anime_real
    ocr = easyocr.Reader(["en"], gpu=True, verbose=False)
    os.makedirs(ZIEL, exist_ok=True)
    mf = open(os.path.join(ZIEL, "manifest.jsonl"), "w", encoding="utf-8")
    stand = {}
    t0 = time.time()
    for ordner, name in QUELLEN.items():
        wurzel = f"{NAS}/{ordner}"
        dateien = sorted(os.path.join(p, f) for p, _, fs in os.walk(wurzel) for f in fs
                         if f.lower().endswith((".jpg", ".jpeg", ".png")) and "fan art" not in p.lower())

        def lesen(pfad):
            # sofort auf hoechstens 2048 px verkleinern: 4K-PNGs zu Tausenden im RAM gaben einen MemoryError
            try:
                im = Image.open(pfad)
                im.draft("RGB", (2048, 2048))
                im = im.convert("RGB")
                if max(im.size) > 2048:
                    im.thumbnail((2048, 2048), Image.LANCZOS)
                return pfad, im
            except Exception:
                return pfad, None

        def stapelweise():
            with ThreadPoolExecutor(6) as ex:
                for s in range(0, len(dateien), 24):
                    yield from ex.map(lesen, dateien[s:s + 24])
        if True:
            for n, (pfad, im) in enumerate(stapelweise()):
                rel = os.path.relpath(pfad, wurzel).replace("\\", "/")
                e = {"quelle": name, "datei_nas": f"{ordner}/{rel}", "gruppe": rel.split("/")[0]}
                if im is None:
                    e["grund"] = "fehler"
                else:
                    w, h = im.size
                    fuss = im.crop((0, int(h * (1 - FUSS)), w, h))
                    fuss.thumbnail((2400, 400))
                    text = " ".join(t for _, t, c in ocr.readtext(__import__("numpy").asarray(fuss)) if c > 0.2)
                    e["fusszeile"] = text[:300]
                    m = re.search(r"software\s+(.{2,40}?)(?:\s{2,}|©|$|wallpaper)", text, re.I)
                    e["software"] = m.group(1).strip() if m else None
                    bild = im.crop((0, 0, w, int(h * (1 - FUSS))))
                    if KI_WORTE.search(text):
                        e["grund"] = "ki-werkzeug in der fusszeile"
                    else:
                        lab, sc = anime_real(bild)
                        real = float(sc if lab == "real" else 1 - sc)
                        if real >= REAL_MIN:
                            e["grund"] = "foto"
                        else:
                            rating, alle, ch = get_wd14_tags(bild, model_name="EVA02_Large",
                                                             general_threshold=SCHWELLE, character_threshold=0.75)
                            e["rating"] = max(rating, key=rating.get)
                            if gesperrt_wd(alle) or (figurenregel_gilt(e["rating"])
                                                     and gesperrt_text(" ".join(ch).replace("_", " ") + " " + rel)):
                                e = {"quelle": name, "gruppe": e["gruppe"], "grund": "gesperrt"}
                            else:
                                daten, gr = hochladen_bytes(bild)
                                ziel = os.path.join(ZIEL, name, f"{n:05d}.jpg")
                                os.makedirs(os.path.dirname(ziel), exist_ok=True)
                                with open(ziel, "wb") as f:
                                    f.write(daten)
                                e["datei"], e["gw"], e["gh"] = ziel, gr[0], gr[1]
                mf.write(json.dumps(e, ensure_ascii=False) + "\n")
                mf.flush()
                k = e.get("grund", "ok")
                stand[(name, k)] = stand.get((name, k), 0) + 1
                if n % 500 == 0:
                    print(f"  {name} {n}/{len(dateien)}  {(time.time()-t0)/60:.0f} min", flush=True)
    mf.close()
    print("FERTIG:", {f"{a}/{b}": v for (a, b), v in sorted(stand.items())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
