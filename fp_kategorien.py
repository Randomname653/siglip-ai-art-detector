r"""fp_kategorien.py — sort the head's false positives on provably human posts into kinds, like imagesort's comic
heuristic does with WD tags: photo, photo background (drawn character on a photo), meme/text, sheet/comic, anime
screencap-look, and the rest = genuine false positives on art. Duplicates (same image posted again) are grouped via
32x32 pixel thumbnails, so each picture counts once.

Human = Telegram post date in the file name before 2022-08-01 (era alibi). Scores come from the production cache
(features.db, key sig2:p8sx_v1), i.e. the folder must have been scanned by imagesort ai-cleanup.

    python fp_kategorien.py [--ordner "//NAS/.../Sorted Stuff"] [--schwelle 0.5]
"""
import argparse
import collections
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import imagesort as I   # noqa: E402

HIER = os.path.dirname(os.path.abspath(__file__))
FOTO_TAGS = {"photo_(medium)": 0.5, "realistic": 0.7, "photorealistic": 0.5}
FOTO_HG_TAGS = {"photo_background": 0.3}
TEXT_TAGS = {"meme": 0.3, "text_focus": 0.3, "english_text": 0.8, "caption": 0.4, "subtitled": 0.4}
SHEET_TAGS = {"multiple_views": 0.5, "reference_sheet": 0.4, "expression_chart": 0.4, "variations": 0.5,
              "character_sheet": 0.4, "sprite_sheet": 0.4}
SCREENCAP_TAGS = {"anime_screencap": 0.3, "screencap": 0.3, "anime_coloring": 0.5, "retro_artstyle": 0.4,
                  "1990s_(style)": 0.4, "1980s_(style)": 0.4}


def trifft(tags, regeln):
    return any(tags.get(t, 0) >= s for t, s in regeln.items())


def daumen(p):
    try:
        im = Image.open(p).convert("L").resize((32, 32), Image.BILINEAR)
    except Exception:
        return None
    v = np.asarray(im, np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def gruppiere(pfade, vek):
    """Dubletten-Gruppen (Cosinus >= 0.97 auf 32x32 Graustufen), greedy."""
    ok = [i for i, v in enumerate(vek) if v is not None]
    V = np.stack([vek[i] for i in ok])
    gruppe = {}
    for a, i in enumerate(ok):
        if pfade[i] in gruppe:
            continue
        sim = V[a:] @ V[a]
        for b in np.flatnonzero(sim >= 0.97):
            gruppe.setdefault(pfade[ok[a + b]], pfade[i])
    for i, v in enumerate(vek):
        gruppe.setdefault(pfade[i], pfade[i])
    return gruppe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordner", required=True, help="gescannter Ordner (Pfad wie im Cache)")
    ap.add_argument("--schwelle", type=float, default=0.5)
    a = ap.parse_args()
    from imgutils.tagging import get_wd14_tags
    c = sqlite3.connect(f"file:{I.CACHE_DB_PATH}?mode=ro", uri=True)
    sc = {p: s for p, s in c.execute("SELECT path, score FROM ai_scores WHERE model_name='sig2:p8sx_v1' AND path LIKE ?",
                                     (a.ordner + "%",)) if "_ai_quarantine" not in p}
    hum = sorted(p for p in sc if (d := I.telegram_post_date(p)) and d < I.AI_ERA_CUTOFF)
    print(f"{len(hum)} menschliche Posts (vor 08/2022), Dubletten-Abgleich ...", flush=True)
    with ThreadPoolExecutor(12) as ex:
        vek = list(ex.map(daumen, hum))
    gruppe = gruppiere(hum, vek)
    n_unik = len(set(gruppe.values()))
    fp = [p for p in hum if sc[p] >= a.schwelle]
    pc = I.RealPhotoCache()
    kat = {}
    for p in fp:
        foto = I.is_real_photo(p, cache=pc)
        try:
            rating, tags, _ = get_wd14_tags(Image.open(p).convert("RGB"), model_name="EVA02_Large",
                                            general_threshold=0.25)
        except Exception:
            rating, tags = {}, {}
        if foto or trifft(tags, FOTO_TAGS):
            k = "Foto"
        elif trifft(tags, FOTO_HG_TAGS):
            k = "Zeichnung auf Foto-Hintergrund"
        elif trifft(tags, TEXT_TAGS):
            k = "Meme / Text"
        elif trifft(tags, SHEET_TAGS) or I.is_comic_heuristic(tags):
            k = "Sheet / Comic"
        elif trifft(tags, SCREENCAP_TAGS):
            k = "Kunst: Anime-Screenshot-Look"
        else:
            k = "Kunst: Illustration u. a."
        kat[p] = k
    pc.close()
    roh = collections.Counter(kat.values())
    unik = collections.Counter()
    gesehen = set()
    for p in fp:
        if gruppe[p] not in gesehen:
            gesehen.add(gruppe[p])
            unik[kat[p]] += 1
    print(f"\nFehlalarme bei Score >= {a.schwelle}: {len(fp)} Posts = {len(gesehen)} verschiedene Bilder "
          f"(von {len(hum)} Posts = {n_unik} verschiedenen Bildern)")
    print(f"{'Art':<34}{'Posts':>7}{'Bilder':>8}{'Anteil an allen Bildern':>26}")
    for k in ("Foto", "Zeichnung auf Foto-Hintergrund", "Meme / Text", "Sheet / Comic",
              "Kunst: Anime-Screenshot-Look", "Kunst: Illustration u. a."):
        print(f"{k:<34}{roh.get(k, 0):>7}{unik.get(k, 0):>8}{unik.get(k, 0) / n_unik:>25.3%}")
    echt = unik["Kunst: Anime-Screenshot-Look"] + unik["Kunst: Illustration u. a."]
    print(f"\n=> echte Fehlalarme auf Kunst: {echt} von {n_unik} Bildern = {echt / n_unik:.3%} "
          f"(roh inkl. Fotos/Memes/Dubletten: {len(fp) / len(hum):.3%})")
    json.dump({"schwelle": a.schwelle, "posts": len(hum), "bilder": n_unik, "fp_posts": len(fp),
               "fp_bilder": len(gesehen), "kategorien_posts": roh, "kategorien_bilder": unik,
               "fp": {p: [kat[p], sc[p], gruppe[p]] for p in fp}},
              open(os.path.join(HIER, "daten", "fp_kategorien.json"), "w", encoding="utf-8"), indent=0)


if __name__ == "__main__":
    main()
