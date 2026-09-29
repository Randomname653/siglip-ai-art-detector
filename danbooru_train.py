r"""danbooru_train.py — current human art (2022 to 03/2026) for TRAINING.

Finding: all human training images were from before 2022, all AI images after — the head had partly learned
"new drawing style = AI" and flagged 6-7 % instead of 1 % of human art from 2026. Remedy: human art from the
same period as the AI images. Separate from the test (danbooru.py): period 2022-01 to 2026-03 (test/val:
2026-04 to 2026-06) and no artist that appears in test/val. Two queries per quarter (official_art, score>=5).
Danbooru bans AI images; posts tagged ai-generated/ai-assisted are skipped anyway. All ratings, schutz.py on
booru and WD tags, photo/style filter, upload simulation. Polite rate limit with backoff on HTTP 429.

    python danbooru_train.py
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

ZIEL = f"{DATASETS}/danbooru_train"
MANIFEST = os.path.join(ZIEL, "manifest.jsonl")
QUARTALE = [(f"{j}-{m:02d}-01", f"{j + (m + 3 > 12)}-{(m + 3 - 1) % 12 + 1:02d}-01")
            for j in range(2022, 2027) for m in (1, 4, 7, 10)
            if (j, m) <= (2026, 1)]
SOLL_JE = 2000             # je Quartal und Anfrage -> ~68k
PRO_KUENSTLER = 20


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
    print(f"{len(QUARTALE)} Quartale, {len(test_k)} Test-Kuenstler gesperrt, schon {sum(zahl.values())}", flush=True)

    from imgutils.tagging import get_wd14_tags
    from imgutils.validate import anime_real
    api, dl = requests.Session(), requests.Session()
    api.headers.update(H)
    dl.headers.update(H)
    sperre = threading.Lock()
    mf = open(MANIFEST, "a", encoding="utf-8")

    def arbeite(p, fenster):
        artist = p.get("tag_string_artist") or ""
        e = {"id": p["id"], "fenster": fenster, "artist": artist, "rating": p.get("rating"),
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
                        stil = sorted(STIL_RAUS & allg)
                        if stil:
                            e["grund"] = "stil: " + ",".join(stil)
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
        for von, bis in QUARTALE:
            for art, tag in (("official", "official_art"), ("score", "score:>=5")):
                fenster = f"{von[:7]}:{art}"
                vor = None
                while zahl.get(fenster, 0) < SOLL_JE:
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
