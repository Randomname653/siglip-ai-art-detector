r"""wallhaven_general.py — human non-anime art from BEFORE 2022 (local wallpaper archive).

Wallhaven "general" wallpapers from the owner's archive: digital painting, concept art, game art, 3D renders,
landscapes — and photos. Files dated before 2022 predate usable image generators. Per image: photo filter
(anime_real, WD photo_(medium)), schutz.py, upload simulation. Split by file name (no artist info): 10 % test,
5 % val. Read-only on the archive.

    python wallhaven_general.py
"""
import datetime
import json
import os
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS, NAS                  # noqa: E402
from civitai import REAL_MIN, STIL_MERKEN, hochladen_bytes   # noqa: E402
from schutz import SCHWELLE, gesperrt_wd                   # noqa: E402

QUELLE = f"{NAS}/wallhaven/general"
ZIEL = f"{DATASETS}/wallhaven_general"
MANIFEST = os.path.join(ZIEL, "manifest.jsonl")
GRENZE = datetime.datetime(2022, 1, 1).timestamp()
ENDUNGEN = (".jpg", ".jpeg", ".png", ".webp")


def main():
    os.makedirs(ZIEL, exist_ok=True)
    gesehen = set()
    if os.path.exists(MANIFEST):
        gesehen = {json.loads(z)["name"] for z in open(MANIFEST, encoding="utf-8")}
    pfade = []
    for dp, dn, fn in os.walk(QUELLE):
        for f in fn:
            if f.lower().endswith(ENDUNGEN) and f not in gesehen:
                p = os.path.join(dp, f)
                try:
                    if os.path.getmtime(p) < GRENZE:
                        pfade.append(p)
                except OSError:
                    pass
    print(f"{len(pfade)} Dateien vor 2022 zu pruefen (schon: {len(gesehen)})", flush=True)

    from imgutils.tagging import get_wd14_tags
    from imgutils.validate import anime_real
    sperre = threading.Lock()
    mf = open(MANIFEST, "a", encoding="utf-8")
    stand = {"n": 0, "ok": 0}
    t0 = time.time()

    def arbeite(p):
        name = os.path.basename(p)
        r_ = zlib.crc32(name.encode()) % 20
        e = {"name": name, "quelle": p, "ordner": os.path.basename(os.path.dirname(p)),
             "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat()[:10],
             "split": "test" if r_ < 2 else "val" if r_ == 2 else "train"}
        try:
            im = Image.open(p).convert("RGB")
            if max(im.size) < 512:
                e["grund"] = "zu klein"
            else:
                lab, sc = anime_real(im)
                if (sc if lab == "real" else 1 - sc) >= REAL_MIN:
                    e["grund"] = "foto"
                else:
                    rating, alle, _ = get_wd14_tags(im, model_name="EVA02_Large", general_threshold=SCHWELLE)
                    allg = {t for t, v in alle.items() if v >= 0.35}
                    e["rating"] = max(rating, key=rating.get)
                    e["tags"] = sorted(allg & (STIL_MERKEN | {"realistic", "3d", "no_humans", "scenery"}))
                    if gesperrt_wd(alle):
                        e = {"name": name, "grund": "gesperrt"}
                    elif "photo_(medium)" in allg:
                        e["grund"] = "stil: photo_(medium)"
            if not e.get("grund"):
                daten, gr = hochladen_bytes(im)
                ziel = os.path.join(ZIEL, os.path.splitext(name)[0] + ".jpg")
                with open(ziel, "wb") as f:
                    f.write(daten)
                e["datei"], e["gw"], e["gh"] = ziel, gr[0], gr[1]
        except Exception as ex:
            e["grund"] = f"fehler: {type(ex).__name__}"
        with sperre:
            mf.write(json.dumps(e, ensure_ascii=False) + "\n")
            mf.flush()
            stand["n"] += 1
            stand["ok"] += bool(e.get("datei"))
            if stand["n"] % 5000 == 0:
                el = time.time() - t0
                print(f"  {stand['n']}/{len(pfade)}  behalten {stand['ok']}  {stand['n']/el:.0f}/s", flush=True)

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(arbeite, pfade))
    mf.close()
    print(f"FERTIG: {stand['ok']} von {stand['n']} behalten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
