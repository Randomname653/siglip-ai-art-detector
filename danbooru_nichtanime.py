r"""danbooru_nichtanime.py — human non-anime art (2022 to 06/2026).

For digital art in general the human side needs non-anime art from the same period as the AI images, or
"non-anime = AI" is learned. Style tags per year (realistic, painting, watercolor, 3d, pixel_art,
traditional_media, landscape, concept_art, cartoon, western_comics, card, character_sheet, promotional_art).
The style filter is OFF (realistic/3d are wanted here); the photo filter stays. Split by artist 80/5/15;
artists from the anime test set are excluded.

    python danbooru_nichtanime.py
"""
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from civitai import REAL_MIN, STIL_MERKEN, STIL_RAUS, hochladen_bytes   # noqa: E402
from danbooru import API, H, KI_TAGS                                   # noqa: E402
from danbooru import MANIFEST as TEST_MANIFEST                         # noqa: E402
from schutz import SCHWELLE, gesperrt_tags, gesperrt_text, gesperrt_wd                # noqa: E402

ZIEL = f"{DATASETS}/danbooru_nichtanime"
MANIFEST = os.path.join(ZIEL, "manifest.jsonl")
JAHRE = [("2022-01-01", "2023-01-01"), ("2023-01-01", "2024-01-01"), ("2024-01-01", "2025-01-01"),
         ("2025-01-01", "2026-01-01"), ("2026-01-01", "2026-07-01")]
# Soll je Tag und Jahr
TAGS = {"realistic": 2000, "painting_(medium)": 2000, "watercolor_(medium)": 1600, "3d": 2000,
        "pixel_art": 1200, "traditional_media": 1600, "landscape": 800, "concept_art": 800,
        "cartoon": 800, "western_comics_(style)": 400,
        # 2026-09-26: v3-Fehlalarme sitzen auf polierter Profi-Kunst (Karten, Sheets, Promo)
        "card_(medium)": 1500, "character_sheet": 1200, "promotional_art": 1000}
PRO_KUENSTLER = 30


def main():
    os.makedirs(ZIEL, exist_ok=True)
    test_k = {json.loads(z).get("artist") for z in open(TEST_MANIFEST, encoding="utf-8")}
    test_k.discard("")
    test_k.discard(None)
    gesehen, je_k, zahl = set(), {}, {}
    if os.path.exists(MANIFEST):
        for z in open(MANIFEST, encoding="utf-8"):
            e = json.loads(z)
            if not str(e.get("grund", "")).startswith("fehler"):
                gesehen.add(e["id"])     # Fehlschlaege (meist 429) duerfen erneut versucht werden
            if e.get("datei"):
                zahl[e["fenster"]] = zahl.get(e["fenster"], 0) + 1
                je_k[e.get("artist")] = je_k.get(e.get("artist"), 0) + 1
    print(f"{len(JAHRE)} Jahre x {len(TAGS)} Tags, {len(test_k)} Test-Kuenstler gesperrt, schon {sum(zahl.values())}", flush=True)

    from imgutils.tagging import get_wd14_tags
    from imgutils.validate import anime_real
    api, dl = requests.Session(), requests.Session()
    api.headers.update(H)
    dl.headers.update(H)
    sperre = threading.Lock()
    mf = open(MANIFEST, "a", encoding="utf-8")

    def arbeite(p, fenster):
        artist = p.get("tag_string_artist") or ""
        k = artist or f"unbekannt:{p['id']}"
        r_ = __import__("zlib").crc32(k.encode()) % 20
        e = {"id": p["id"], "fenster": fenster, "artist": artist, "rating": p.get("rating"),
             "split": "test" if r_ < 3 else "val" if r_ == 3 else "train",
             "danbooru_tags": " ".join(t for t in p.get("tag_string_general", "").split() if t in TAGS),
             "created": p.get("created_at"), "w": p.get("image_width"), "h": p.get("image_height"),
             "quelle_url": f"https://danbooru.donmai.us/posts/{p['id']}"}
        try:
            r = holen(p["file_url"])
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            if max(im.size) < 512:
                e["grund"] = "zu klein"
            else:
                lab, sc = anime_real(im)
                if (sc if lab == "real" else 1 - sc) >= REAL_MIN:
                    e["grund"] = "foto"
                else:
                    _, alle, _ = get_wd14_tags(im, model_name="EVA02_Large", general_threshold=SCHWELLE)
                    if gesperrt_wd(alle):
                        e = {"id": p["id"], "fenster": fenster, "grund": "gesperrt"}
                    else:
                        allg = {t for t, v in alle.items() if v >= 0.35}
                        e["tags"] = sorted(allg & STIL_MERKEN)
                        if "photo_(medium)" in allg:
                            e["grund"] = "stil: photo_(medium)"   # Fotos bleiben draussen
            if not e.get("grund"):
                daten, gr = hochladen_bytes(im)
                pfad = os.path.join(ZIEL, f"{p['id']}.jpg")
                with open(pfad, "wb") as f:
                    f.write(daten)
                e["datei"], e["gw"], e["gh"] = pfad, gr[0], gr[1]
        except Exception as ex:
            e["grund"] = f"fehler: {type(ex).__name__}"
        with sperre:
            mf.write(json.dumps(e, ensure_ascii=False) + "\n")
            mf.flush()
            if e.get("datei"):
                zahl[fenster] = zahl.get(fenster, 0) + 1

    takt = threading.Lock()
    letzte = [0.0]

    def holen(url):
        """Hoechstens 2 Downloads pro Sekunde; auf 429 (Cloudflare: zu schnell)
        mit wachsender Pause warten und erneut versuchen — nichts umgehen, nur
        langsamer werden (2026-09-25: 61 % der Downloads scheiterten mit 429)."""
        for versuch in range(6):
            with takt:
                warte = letzte[0] + 0.5 - time.time()
                if warte > 0:
                    time.sleep(warte)
                letzte[0] = time.time()
            r = dl.get(url, timeout=120)
            if r.status_code != 429:
                r.raise_for_status()
                return r
            time.sleep(30 * 2 ** versuch)
        r.raise_for_status()

    t0 = time.time()
    with ThreadPoolExecutor(3) as ex:
        for von, bis in JAHRE:
            for art, tag in ((t, t) for t in TAGS):
                fenster = f"{von[:7]}:{art}"
                vor = None
                while zahl.get(fenster, 0) < TAGS[art]:
                    par = {"limit": 200, "tags": f"{tag} date:{von}..{bis}"}
                    if vor:
                        par["page"] = f"b{vor}"
                    seite = None
                    for versuch in range(8):
                        try:
                            r = api.get(API, params=par, timeout=90)
                            if r.status_code == 200:
                                seite = r.json()
                                break
                        except (requests.RequestException, ValueError):
                            pass
                        time.sleep(10 * (versuch + 1))
                    if not seite:
                        break
                    vor = min(p["id"] for p in seite)
                    wahl = []
                    for p in seite:
                        if p["id"] in gesehen or not p.get("file_url") or p.get("is_deleted") or p.get("is_banned"):
                            continue
                        if p.get("file_ext") not in ("jpg", "jpeg", "png", "webp"):
                            continue
                        tags_p = set(p.get("tag_string", "").split())
                        a = p.get("tag_string_artist") or ""
                        if tags_p & KI_TAGS or (a and a in test_k):
                            continue
                        gesehen.add(p["id"])
                        if gesperrt_tags(tags_p) or (p.get("rating") in ("s", "q", "e") and gesperrt_text(
                            p.get("tag_string_character", "") + " " + p.get("tag_string_copyright", ""))):
                            mf.write(json.dumps({"id": p["id"], "fenster": fenster, "grund": "gesperrt"}) + "\n")
                            continue
                        if a and je_k.get(a, 0) >= PRO_KUENSTLER:
                            continue
                        je_k[a] = je_k.get(a, 0) + 1
                        wahl.append(p)
                    list(ex.map(lambda p: arbeite(p, fenster), wahl))
                    time.sleep(0.5)
                print(f"  {fenster}: {zahl.get(fenster, 0)}  (gesamt {sum(zahl.values())}, "
                      f"{(time.time()-t0)/60:.0f} min)", flush=True)
    mf.close()
    print("FERTIG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
