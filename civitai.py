r"""civitai.py — real AI images from CivitAI (official API) for training and testing.

Why: a head trained on 14 locally run generators caught 97 % of them but only about a third of a real Telegram
AI channel. Missing was the variety of real pipelines: merges, LoRAs, hires fix, upscalers, and models that
cannot run locally (OpenAI, Nano Banana, Seedream, Grok, ...).

Per image:
  1. download the original (images only, no video)
  2. photo filter (anime_real "real" >= 0.75), style filter via WD tags (realistic/3d/photorealistic ->
     civitai_stil.py later fetched those separately) and child protection (schutz.py, never stored)
  3. upload simulation (longer side <= 1280, JPEG q87 4:2:0); ONLY this version is stored, so every source
     has the same history
At most PRO_NUTZER images per creator and base model; the creator is the split unit later.
Resumable: decided IDs are in manifest.jsonl. API key: env CIVITAI_API_KEY or <DETEKTOR_SCHLUESSEL>/civitai_key.txt.

    python civitai.py --plan
    python civitai.py
"""
import argparse
import collections
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS, SCHLUESSEL           # noqa: E402
from schutz import SCHWELLE, gesperrt_wd         # noqa: E402

ZIEL = f"{DATASETS}/civitai"
MANIFEST = os.path.join(ZIEL, "manifest.jsonl")
API = "https://civitai.com/api/v1/images"
PRO_NUTZER = 40
REAL_MIN = 0.75
STIL_RAUS = {"realistic", "3d", "photorealistic", "photo_(medium)"}
STIL_MERKEN = STIL_RAUS | {"monochrome", "greyscale", "comic", "sketch", "furry", "chibi",
                           "pixel_art", "traditional_media", "watercolor_(medium)",
                           "western_comics_(style)", "cartoonized", "english_text"}

# Soll je Basismodell (gespeicherte Bilder). Geschlossene Modelle liefern viel
# Fotorealistisches, das der Filter verwirft — dort wird laenger gesucht.
SOLL = {
    "Illustrious": 22000, "Pony": 10000, "NoobAI": 7000, "SDXL 1.0": 7000, "SD 1.5": 7000,
    "Anima": 6000, "Krea 2": 7000, "ZImageTurbo": 5000, "ZImageBase": 2000,
    "Flux.1 D": 6000, "Flux.1 S": 1500, "Flux.1 Krea": 1500, "Flux.2 Klein 9B": 2000,
    "Flux.2 Klein 4B": 1000, "Flux.2 D": 1500, "Chroma": 3000, "Qwen": 4000, "Qwen 2": 2000,
    "Qwen 2.1": 2000, "OpenAI": 6000, "Nano Banana": 6000, "Seedream": 4000, "Grok": 3000,
    "MiniMax H3": 4000, "Imagen4": 1500, "Pony V7": 2000, "HiDream": 2000, "Kolors": 1500,
    "SD 3.5": 1500, "Hunyuan 1": 1000, "Lumina": 1000, "Muse Image": 1000, "Reve": 1000,
}
# 2026-09-26: v3 erkennt Lumina (ungesehene Ersteller) nur zu 72 % — duenne
# Basismodelle aufstocken (Neta Lumina laeuft auf CivitAI als "Lumina")
SOLL.update({"Lumina": 8000, "SD 3.5": 4000, "Kolors": 4000, "HiDream": 4000, "Imagen4": 4000,
             "Flux.2 Klein 9B": 4000, "Flux.2 Klein 4B": 3000, "Flux.1 Krea": 3000, "Pony V7": 4000,
             "ZImageBase": 4000, "Qwen 2": 3000, "Hunyuan 1": 2000, "MiniMax H3": 4000,
             "Grok": 5000, "Chroma": 5000})

SORTEN = (("Newest", "AllTime"), ("Most Reactions", "AllTime"), ("Most Reactions", "Year"))


def schluessel():
    # Schluessel nie im Code: Umgebungsvariable CIVITAI_API_KEY oder Datei <DETEKTOR_SCHLUESSEL>/civitai_key.txt
    return os.environ.get("CIVITAI_API_KEY") or open(os.path.join(SCHLUESSEL, "civitai_key.txt"),
                                                     encoding="utf-8").read().strip()


def ordner(basis):
    return os.path.join(ZIEL, re.sub(r"[^A-Za-z0-9._-]+", "_", basis))


def hochladen_bytes(im):
    """aufbau2.hochladen, aber die JPEG-Bytes selbst — kein zweites Kodieren."""
    w, h = im.size
    if max(w, h) > 1280:
        f = 1280 / max(w, h)
        im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=87, subsampling=2)
    return buf.getvalue(), im.size


class Api:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {schluessel()}"

    def seite(self, params):
        for versuch in range(8):
            try:
                r = self.s.get(API, params=params, timeout=90)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(10 * (versuch + 1))
                    continue
                return r.json()
            except (requests.RequestException, ValueError):
                time.sleep(10 * (versuch + 1))
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="nur zaehlen, was schon da ist")
    ap.add_argument("--worker", type=int, default=12)
    ap.add_argument("--nur", action="append", help="nur diese Basismodelle")
    ap.add_argument("--hoechstens", type=int, default=0, help="Soll je Basismodell deckeln (Probe)")
    args = ap.parse_args()
    if args.hoechstens:
        for b in SOLL:
            SOLL[b] = min(SOLL[b], args.hoechstens)
    os.makedirs(ZIEL, exist_ok=True)

    gesehen, gespeichert, je_nutzer = set(), collections.Counter(), collections.Counter()
    if os.path.exists(MANIFEST):
        for z in open(MANIFEST, encoding="utf-8"):
            e = json.loads(z)
            gesehen.add(e["id"])
            if e.get("datei"):
                gespeichert[e["basis"]] += 1
                je_nutzer[(e["basis"], e.get("user"))] += 1
    print(f"schon entschieden: {len(gesehen)}, gespeichert: {sum(gespeichert.values())}")
    for b in SOLL:
        print(f"  {b:<16} {gespeichert[b]:>6} / {SOLL[b]}")
    if args.plan:
        return 0

    from imgutils.tagging import get_wd14_tags
    from imgutils.validate import anime_real
    api = Api()
    dl = requests.Session()
    sperre = threading.Lock()
    mf = open(MANIFEST, "a", encoding="utf-8")

    def arbeite(item, basis):
        e = {"id": item["id"], "basis": basis, "user": item.get("username"),
             "nsfw": item.get("nsfwLevel"), "w": item.get("width"), "h": item.get("height"),
             "created": item.get("createdAt"), "url": item.get("url")}
        try:
            r = dl.get(item["url"], timeout=120)
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
                    # anime_real haelt 3D-Renderings und Halbrealistisches fuer
                    # "anime" (Probe Nano Banana, 25.09.). Die menschliche Seite
                    # ist Anime — sonst lernt der Kopf "3D-Look = KI".
                    rating, alle, _ = get_wd14_tags(im, model_name="EVA02_Large",
                                                    general_threshold=SCHWELLE)
                    allg = {t: p for t, p in alle.items() if p >= 0.35}
                    stil = sorted(STIL_RAUS & set(allg))
                    e["rating"] = max(rating, key=rating.get)
                    e["tags"] = [t for t in allg if t in STIL_MERKEN]
                    minderj = gesperrt_wd(alle)
                    if minderj:
                        # schutz.py: nie speichern, auch keine Tags im Manifest
                        e = {k: e[k] for k in ("id", "basis", "user")}
                        e["grund"] = "gesperrt"
                    elif stil:
                        e["grund"] = "stil: " + ",".join(stil)
                if not e.get("grund"):
                    daten, gr = hochladen_bytes(im)
                    pfad = os.path.join(ordner(basis), f"{item['id']}.jpg")
                    with open(pfad, "wb") as f:
                        f.write(daten)
                    e["datei"] = pfad
                    e["gw"], e["gh"] = gr
        except Exception as ex:
            e["grund"] = f"fehler: {type(ex).__name__}"
        with sperre:
            mf.write(json.dumps(e, ensure_ascii=False) + "\n")
            mf.flush()
            if e.get("datei"):
                gespeichert[basis] += 1
        return e

    t0, n0 = time.time(), sum(gespeichert.values())
    with ThreadPoolExecutor(args.worker) as ex:
        for basis in (args.nur or SOLL):
            os.makedirs(ordner(basis), exist_ok=True)
            for sort, period in SORTEN:
                cur, leer = None, 0
                while gespeichert[basis] < SOLL[basis]:
                    p = {"limit": 200, "sort": sort, "period": period, "nsfw": "X",
                         "baseModels": basis}
                    if cur:
                        p["cursor"] = cur
                    d = api.seite(p)
                    items = [i for i in d.get("items", [])
                             if i.get("type", "image") == "image" and i.get("baseModel") == basis
                             and i["id"] not in gesehen
                             and je_nutzer[(basis, i.get("username"))] < PRO_NUTZER]
                    for i in items:
                        gesehen.add(i["id"])
                        je_nutzer[(basis, i.get("username"))] += 1
                    # Stapel abarbeiten, bevor die naechste Seite kommt — so
                    # schiesst das Soll nicht um Tausende ueber
                    list(ex.map(lambda i: arbeite(i, basis), items))
                    cur = d.get("metadata", {}).get("nextCursor")
                    leer = leer + 1 if not items else 0
                    if not cur or leer >= 25:
                        break
                    el = time.time() - t0
                    n = sum(gespeichert.values()) - n0
                    print(f"  {basis:<16} {sort[:6]}/{period:<7} {gespeichert[basis]:>6}/{SOLL[basis]}"
                          f"  gesamt +{n} ({n/max(el,1):.1f}/s)", flush=True)
            print(f"{basis}: {gespeichert[basis]} gespeichert", flush=True)
    mf.close()
    print("FERTIG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
