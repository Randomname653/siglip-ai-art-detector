r"""civitai_stil.py — die vom Stil-Filter verworfenen CivitAI-Bilder fuer Phase 2 nachholen.

civitai.py hat fuer den Anime-Detektor alles verworfen, was der WD-Tagger als
realistic/3d/photorealistic einstufte (~30.000 Bilder; im Manifest mit grund
"stil: ..."). Fuer Phase 2 (digitale Kunst allgemein: Konzeptkunst, 3D, halbrealistisch)
sind genau diese Bilder die KI-Seite. Echte Fotos bleiben weiter draussen:
  - Foto-Filter (anime_real) und WD-Tag photo_(medium)
  - schutz.py wie ueberall
Eigener Ordner <DATASETS>\civitai_stil, damit die Anime-Quelle unveraendert bleibt.

    python lab/detektor/civitai_stil.py
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
from civitai import MANIFEST as QUELL_MANIFEST, REAL_MIN, STIL_MERKEN, hochladen_bytes, ordner  # noqa: E402
from schutz import SCHWELLE, gesperrt_wd                                                          # noqa: E402

ZIEL = f"{DATASETS}/civitai_stil"
MANIFEST = os.path.join(ZIEL, "manifest.jsonl")


def main():
    os.makedirs(ZIEL, exist_ok=True)
    gesehen = set()
    if os.path.exists(MANIFEST):
        gesehen = {json.loads(z)["id"] for z in open(MANIFEST, encoding="utf-8")}
    kand = [e for e in (json.loads(z) for z in open(QUELL_MANIFEST, encoding="utf-8"))
            if str(e.get("grund", "")).startswith("stil") and e.get("url") and e["id"] not in gesehen]
    print(f"{len(kand)} stil-verworfene Bilder nachzuholen", flush=True)

    from imgutils.tagging import get_wd14_tags
    from imgutils.validate import anime_real
    dl = requests.Session()
    sperre = threading.Lock()
    mf = open(MANIFEST, "a", encoding="utf-8")
    stand = {"n": 0, "ok": 0}
    t0 = time.time()

    def arbeite(q):
        e = {k: q.get(k) for k in ("id", "basis", "user", "nsfw", "w", "h", "created", "url")}
        try:
            r = dl.get(q["url"], timeout=120)
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            lab, sc = anime_real(im)
            e["real"] = round(float(sc if lab == "real" else 1 - sc), 3)
            if e["real"] >= REAL_MIN:
                e["grund"] = "foto"
            else:
                rating, alle, _ = get_wd14_tags(im, model_name="EVA02_Large", general_threshold=SCHWELLE)
                allg = {t for t, v in alle.items() if v >= 0.35}
                if gesperrt_wd(alle):
                    e = {"id": q["id"], "basis": q.get("basis"), "user": q.get("user"), "grund": "gesperrt"}
                elif "photo_(medium)" in allg:
                    e["grund"] = "stil: photo_(medium)"
                else:
                    e["rating"] = max(rating, key=rating.get)
                    e["tags"] = sorted(allg & (STIL_MERKEN | {"realistic", "3d", "photorealistic"}))
                    daten, gr = hochladen_bytes(im)
                    pfad = os.path.join(ZIEL, os.path.basename(ordner(q["basis"])), f"{q['id']}.jpg")
                    os.makedirs(os.path.dirname(pfad), exist_ok=True)
                    with open(pfad, "wb") as f:
                        f.write(daten)
                    e["datei"], e["gw"], e["gh"] = pfad, gr[0], gr[1]
        except Exception as ex:
            e["grund"] = f"fehler: {type(ex).__name__}"
        with sperre:
            mf.write(json.dumps(e, ensure_ascii=False) + "\n")
            mf.flush()
            stand["n"] += 1
            stand["ok"] += bool(e.get("datei"))
            if stand["n"] % 2000 == 0:
                print(f"  {stand['n']}/{len(kand)}  behalten {stand['ok']}  "
                      f"{stand['n']/(time.time()-t0):.1f}/s", flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(arbeite, kand))
    mf.close()
    print(f"FERTIG: {stand['ok']} von {stand['n']} behalten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
