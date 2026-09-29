r"""aibooru.py — AI anime images from aibooru.online (official API).

AIBooru is a Danbooru-like site for AI images only, mostly anime, with model tags.
  - ONLY ratings general/sensitive, plus schutz.py on booru tags AND WD tags (large parts of the top-rated
    explicit posts depict minors; they never touch the disk)
  - only post IDs below GRENZE_ID: a Telegram channel reposts newer AIBooru posts and is the independent test
  - no animations/videos; photo and style filter as civitai.py; only the upload simulation is stored
  - at most PRO_NUTZER images per uploader; the uploader is the split unit

    python aibooru.py
    python aibooru.py --tag nijijourney     (rating still limited to g/s)
"""
import argparse
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
from schutz import SCHWELLE, gesperrt_tags, gesperrt_wd                # noqa: E402

ZIEL = f"{DATASETS}/aibooru"
MANIFEST = os.path.join(ZIEL, "manifest.jsonl")
API = "https://aibooru.online/posts.json"
GRENZE_ID = 150000
SOLL = 35000
PRO_NUTZER = 150
H = {"User-Agent": "ai-art-detector-lab/0.1 (research)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=6)
    ap.add_argument("--soll", type=int, default=SOLL)
    ap.add_argument("--tag", default=None,
                    help="gezielt ein Modell-Tag holen (z. B. nijijourney), ohne Uploader-Deckel")
    args = ap.parse_args()
    os.makedirs(ZIEL, exist_ok=True)

    gesehen, je_nutzer, gespeichert = set(), {}, 0
    if os.path.exists(MANIFEST):
        for z in open(MANIFEST, encoding="utf-8"):
            e = json.loads(z)
            gesehen.add(e["id"])
            gespeichert += bool(e.get("datei"))
            je_nutzer[e.get("user")] = je_nutzer.get(e.get("user"), 0) + 1
    print(f"schon entschieden {len(gesehen)}, gespeichert {gespeichert}", flush=True)

    from imgutils.tagging import get_wd14_tags
    from imgutils.validate import anime_real
    api, dl = requests.Session(), requests.Session()
    api.headers.update(H)
    dl.headers.update(H)
    sperre = threading.Lock()
    mf = open(MANIFEST, "a", encoding="utf-8")
    stand = {"n": gespeichert}
    if args.tag:
        args.soll = gespeichert + 10 ** 6       # im Tag-Modus alles, was es gibt

    def arbeite(p):
        e = {"id": p["id"], "user": p.get("uploader_id"), "rating": p.get("rating"),
             "model": p.get("tag_string_model", ""), "meta": p.get("tag_string_meta", ""),
             "w": p.get("image_width"), "h": p.get("image_height"), "created": p.get("created_at")}
        try:
            r = dl.get(p["file_url"], timeout=120)
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content))
            e["fmt"] = im.format
            im = im.convert("RGB")
            if max(im.size) < 512:
                e["grund"] = "zu klein"
            else:
                lab, sc = anime_real(im)
                e["real"] = round(float(sc if lab == "real" else 1 - sc), 3)
                if e["real"] >= REAL_MIN:
                    e["grund"] = "foto"
                else:
                    _, alle, _ = get_wd14_tags(im, model_name="EVA02_Large", general_threshold=SCHWELLE)
                    if gesperrt_wd(alle):
                        e = {"id": p["id"], "user": p.get("uploader_id"), "grund": "gesperrt"}
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
            stand["n"] += bool(e.get("datei"))

    # rueckwaerts ab der Grenze: page=b<ID> liefert Posts mit kleinerer ID
    vor, t0, n0 = GRENZE_ID, time.time(), stand["n"]
    with ThreadPoolExecutor(args.worker) as ex:
        while stand["n"] < args.soll and vor > 1:
            for versuch in range(8):
                try:
                    r = api.get(API, params={"limit": 200, "page": f"b{vor}",
                                             "tags": args.tag or "rating:g,s"},
                                timeout=90)
                    if r.status_code == 200:
                        seite = r.json()
                        break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(10 * (versuch + 1))
            else:
                print("API antwortet nicht, Abbruch", flush=True)
                break
            if not seite:
                break
            vor = min(p["id"] for p in seite)
            wahl = []
            for p in seite:
                if p["id"] in gesehen or p["id"] >= GRENZE_ID or not p.get("file_url"):
                    continue
                if p.get("rating") not in ("g", "s") or p.get("file_ext") not in ("jpg", "jpeg", "png", "webp"):
                    continue
                if gesperrt_tags(p.get("tag_string", "")):
                    gesehen.add(p["id"])
                    mf.write(json.dumps({"id": p["id"], "grund": "gesperrt"}) + "\n")
                    continue
                u = p.get("uploader_id")
                if je_nutzer.get(u, 0) >= PRO_NUTZER and not args.tag:
                    continue
                je_nutzer[u] = je_nutzer.get(u, 0) + 1
                gesehen.add(p["id"])
                wahl.append(p)
            list(ex.map(arbeite, wahl))
            el = time.time() - t0
            print(f"  bis ID {vor}: gespeichert {stand['n']}/{args.soll} "
                  f"({(stand['n']-n0)/max(el,1):.1f}/s)", flush=True)
            time.sleep(0.5)
    mf.close()
    print("FERTIG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
