r"""diffusiondb.py — early Stable Diffusion images from 2022 (DiffusionDB 2M, CC0) as an AI source.

Why: 2022 posts of a test channel were detected at 67 % vs. 90 % for 2023 — early AI (SD 1.x, Aug 2022:
landscapes, sci-fi, concept art with a render look) was missing from training.
Source: huggingface.co/datasets/poloclub/diffusiondb, license CC0-1.0. Every 50th of 2,000 parts (40 parts);
at most PRO_NUTZER images per (hashed) Discord user. Filters as civitai_stil.py (child-protection words in the
prompt, WD tags, character list for non-general images, photo filter); stored as upload simulation.

    python diffusiondb.py [--teile 40] [--pro-nutzer 25]
"""
import argparse
import collections
import io
import json
import os
import re
import sys
import time
import zipfile

import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from civitai import REAL_MIN, STIL_MERKEN, hochladen_bytes                  # noqa: E402
from schutz import SCHWELLE, SPERRE, figurenregel_gilt, gesperrt_text, gesperrt_wd  # noqa: E402

ZIEL = f"{DATASETS}/diffusiondb"
MANIFEST = os.path.join(ZIEL, "manifest.jsonl")
URL = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/"
SAMPLER = {1: "ddim", 2: "plms", 3: "k_euler", 4: "k_euler_ancestral", 5: "k_heun", 6: "k_dpm_2",
           7: "k_dpm_2_ancestral", 8: "k_lms", 9: "others"}
WORT = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def sperrwort(prompt):
    """SPERRE-Begriffe als ganze Woerter im Prompt (Unterstrich-Tags auch als Wortfolge)."""
    woerter = WORT.findall((prompt or "").lower().replace("_", " "))
    s = " " + " ".join(woerter) + " "
    return sorted(t for t in SPERRE if " " + t.replace("_", " ") + " " in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teile", type=int, default=40)
    ap.add_argument("--pro-nutzer", type=int, default=25)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    meta = pq.read_table(os.path.join(ZIEL, "metadata.parquet")).to_pydict()
    schritt = 2000 // a.teile
    teile = list(range(1, 2001, schritt))[:a.teile]
    zeilen = collections.defaultdict(dict)
    for i, part in enumerate(meta["part_id"]):
        if part in teile:
            zeilen[part][meta["image_name"][i]] = i

    gesehen, pro_nutzer = set(), collections.Counter()
    if os.path.exists(MANIFEST):
        for z in open(MANIFEST, encoding="utf-8"):
            e = json.loads(z)
            gesehen.add(e["id"])
            if e.get("datei"):
                pro_nutzer[e["user"]] += 1
    fertig_teile = {int(p) for p in os.listdir(ZIEL) if p.isdigit()} if os.path.isdir(ZIEL) else set()

    from imgutils.tagging import get_wd14_tags
    from imgutils.validate import anime_real
    mf = open(MANIFEST, "a", encoding="utf-8")
    t0, stand = time.time(), collections.Counter()
    for part in teile:
        if part in fertig_teile and os.path.exists(os.path.join(ZIEL, f"{part:06d}", ".fertig")):
            continue
        zp = os.path.join(ZIEL, f"part-{part:06d}.zip")
        for versuch in range(5):
            try:
                with requests.get(URL + f"images/part-{part:06d}.zip", stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(zp + ".tmp", "wb") as f:
                        for block in r.iter_content(1 << 20):
                            f.write(block)
                os.replace(zp + ".tmp", zp)
                break
            except Exception as ex:
                print(f"  Teil {part}: {type(ex).__name__}, neuer Versuch", flush=True)
                time.sleep(30 * (versuch + 1))
        else:
            print(f"  Teil {part} uebersprungen", flush=True)
            continue
        ordner = os.path.join(ZIEL, f"{part:06d}")
        os.makedirs(ordner, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".png")):
                i = zeilen[part].get(name)
                if i is None or name in gesehen:
                    continue
                user = meta["user_name"][i]
                ts = meta["timestamp"][i]
                e = {"id": name, "part": part, "user": user, "prompt": meta["prompt"][i],
                     "w": meta["width"][i], "h": meta["height"][i], "steps": meta["step"][i],
                     "cfg": meta["cfg"][i], "sampler": SAMPLER.get(meta["sampler"][i], "?"),
                     "zeit": ts.isoformat() if ts else None, "nsfw_bild": round(meta["image_nsfw"][i], 3),
                     "lizenz": "CC0-1.0", "quelle": URL + f"images/part-{part:06d}.zip"}
                try:
                    if pro_nutzer[user] >= a.pro_nutzer:
                        e = {"id": name, "user": user, "grund": "nutzer-limit"}
                    elif sperrwort(e["prompt"]):
                        e = {"id": name, "user": user, "grund": "gesperrt"}
                    else:
                        im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
                        lab, sc = anime_real(im)
                        e["real"] = round(float(sc if lab == "real" else 1 - sc), 3)
                        if e["real"] >= REAL_MIN:
                            e["grund"] = "foto"
                        else:
                            rating, alle, _ = get_wd14_tags(im, model_name="EVA02_Large",
                                                            general_threshold=SCHWELLE)
                            allg = {t for t, v in alle.items() if v >= 0.35}
                            e["rating"] = max(rating, key=rating.get)
                            if gesperrt_wd(alle) or (figurenregel_gilt(e["rating"]) and gesperrt_text(e["prompt"])):
                                e = {"id": name, "user": user, "grund": "gesperrt"}
                            elif "photo_(medium)" in allg:
                                e["grund"] = "stil: photo_(medium)"
                            else:
                                e["tags"] = sorted(allg & (STIL_MERKEN | {"realistic", "3d", "photorealistic"}))
                                daten, gr = hochladen_bytes(im)
                                pfad = os.path.join(ordner, name[:-4] + ".jpg")
                                with open(pfad, "wb") as f:
                                    f.write(daten)
                                e["datei"], e["gw"], e["gh"] = pfad, gr[0], gr[1]
                                pro_nutzer[user] += 1
                except Exception as ex:
                    e["grund"] = f"fehler: {type(ex).__name__}"
                mf.write(json.dumps(e, ensure_ascii=False) + "\n")
                mf.flush()
                stand["n"] += 1
                stand[e.get("grund", "ok").split(":")[0]] += 1
        os.remove(zp)
        open(os.path.join(ordner, ".fertig"), "w").close()
        print(f"Teil {part}: bisher {dict(stand)}  {(time.time()-t0)/60:.0f} min", flush=True)
    mf.close()
    print(f"FERTIG: {dict(stand)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
