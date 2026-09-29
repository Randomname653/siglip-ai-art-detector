r"""danbooru.py — human anime art AFTER 2022 for testing and calibration (never trained on).

Danbooru bans and deletes AI images; what remains is human with high confidence. Additionally: only posts up to
a cut-off date (moderation had months), no deleted/banned posts, no ai-generated/ai-assisted tags, all ratings,
schutz.py on booru and WD tags, photo/style filter, upload simulation. Two queries: official_art and score>=10.
At most PRO_KUENSTLER images per artist; the artist decides whether an image goes to val (threshold) or test.

    python danbooru.py
"""
import argparse
import io
import json
import os
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from civitai import REAL_MIN, STIL_MERKEN, STIL_RAUS, hochladen_bytes   # noqa: E402
from schutz import SCHWELLE, gesperrt_tags, gesperrt_text, gesperrt_wd                # noqa: E402

ZIEL = f"{DATASETS}/danbooru_ab2023"
MANIFEST = os.path.join(ZIEL, "manifest.jsonl")
API = "https://danbooru.donmai.us/posts.json"
STICHTAG = "2023-01-01..2026-06-30"
ANFRAGEN = {"official": f"official_art date:{STICHTAG}", "score": f"score:>=10 date:{STICHTAG}"}
SOLL_JE = 6000
PRO_KUENSTLER = 15
KI_TAGS = {"ai-generated", "ai-assisted", "ai-generated_background", "ai_generated",
           "ai-generated_art_(topic)", "stable_diffusion", "novelai", "midjourney"}
H = {"User-Agent": "ai-art-detector-lab/0.1 (research)"}


def split_von(kuenstler, pid):
    k = kuenstler or f"unbekannt:{pid}"
    return "val" if zlib.crc32(k.encode()) % 10 < 3 else "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(ZIEL, exist_ok=True)
    gesehen, je_k, gespeichert = set(), {}, {a: 0 for a in ANFRAGEN}
    if os.path.exists(MANIFEST):
        for z in open(MANIFEST, encoding="utf-8"):
            e = json.loads(z)
            gesehen.add(e["id"])
            if e.get("datei"):
                gespeichert[e["anfrage"]] += 1
                je_k[e.get("artist")] = je_k.get(e.get("artist"), 0) + 1

    from imgutils.tagging import get_wd14_tags
    from imgutils.validate import anime_real
    api, dl = requests.Session(), requests.Session()
    api.headers.update(H)
    dl.headers.update(H)
    sperre = threading.Lock()
    mf = open(MANIFEST, "a", encoding="utf-8")

    def arbeite(p, anfrage):
        artist = p.get("tag_string_artist") or ""
        e = {"id": p["id"], "anfrage": anfrage, "artist": artist, "rating": p.get("rating"),
             "created": p.get("created_at"), "split": split_von(artist, p["id"]),
             "w": p.get("image_width"), "h": p.get("image_height")}
        try:
            r = dl.get(p["file_url"], timeout=120)
            r.raise_for_status()
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
                        e = {"id": p["id"], "anfrage": anfrage, "grund": "gesperrt"}
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
                gespeichert[anfrage] += 1

    with ThreadPoolExecutor(args.worker) as ex:
        for anfrage, tags in ANFRAGEN.items():
            vor, t0 = None, time.time()
            while gespeichert[anfrage] < SOLL_JE:
                par = {"limit": 200, "tags": tags}
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
                    if tags_p & KI_TAGS:
                        continue
                    gesehen.add(p["id"])
                    if gesperrt_tags(tags_p) or (p.get("rating") in ("s", "q", "e") and gesperrt_text(
                            p.get("tag_string_character", "") + " " + p.get("tag_string_copyright", ""))):
                        mf.write(json.dumps({"id": p["id"], "anfrage": anfrage, "grund": "gesperrt"}) + "\n")
                        continue
                    a = p.get("tag_string_artist") or ""
                    if a and je_k.get(a, 0) >= PRO_KUENSTLER:
                        continue
                    je_k[a] = je_k.get(a, 0) + 1
                    wahl.append(p)
                list(ex.map(lambda p: arbeite(p, anfrage), wahl))
                print(f"  {anfrage:<9} bis ID {vor}: {gespeichert[anfrage]}/{SOLL_JE} "
                      f"({(time.time()-t0)/60:.0f} min)", flush=True)
                time.sleep(0.5)
    mf.close()
    print("FERTIG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
