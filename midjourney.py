r"""midjourney.py — Midjourney/Nijijourney from two CC0 collections of individual creators.

Early Midjourney (v3/v4, 2022) was missing from training; the Telegram Midjourney channels are test only.
Only collections whose creators release their OWN images: wafflefan/felix-midjourney-archive (CC0, 32,477
images, 2022-2025, v3/v4 to v7 and niji) and p1atdev/niji-v5 (CC0). Filters: schutz.py, photo filter, SHA-256
duplicates; upload simulation. Split later by job ID / prompt (one creator per source).

    python midjourney.py [--nur felix|niji]
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import zipfile

import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from civitai import REAL_MIN, STIL_MERKEN, hochladen_bytes                  # noqa: E402
from diffusiondb import sperrwort                                           # noqa: E402
from schutz import SCHWELLE, figurenregel_gilt, gesperrt_text, gesperrt_wd  # noqa: E402

HF = "https://huggingface.co/datasets/"
QUELLEN = {
    "felix": ("wafflefan/felix-midjourney-archive", f"{DATASETS}/midjourney_felix"),
    "niji": ("p1atdev/niji-v5", f"{DATASETS}/nijijourney_p1atdev"),
}
BILD = (".png", ".jpg", ".jpeg", ".webp")
Image.MAX_IMAGE_PIXELS = 200_000_000


def holen(url, ziel):
    for versuch in range(5):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(ziel + ".tmp", "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            os.replace(ziel + ".tmp", ziel)
            return True
        except Exception as ex:
            print(f"  {os.path.basename(ziel)}: {type(ex).__name__}, neuer Versuch", flush=True)
            time.sleep(30 * (versuch + 1))
    return False


class Pruefer:
    def __init__(self, ziel):
        from imgutils.tagging import get_wd14_tags
        from imgutils.validate import anime_real
        self.wd, self.ar = get_wd14_tags, anime_real
        self.ziel = ziel
        self.mf_pfad = os.path.join(ziel, "manifest.jsonl")
        self.gesehen = set()
        if os.path.exists(self.mf_pfad):
            self.gesehen = {json.loads(z)["sha256"] for z in open(self.mf_pfad, encoding="utf-8")}
        self.mf = open(self.mf_pfad, "a", encoding="utf-8")
        self.stand = collections.Counter()

    def bild(self, daten, e, unterordner):
        sha = hashlib.sha256(daten).hexdigest()
        if sha in self.gesehen:
            self.stand["doppelt"] += 1
            return
        self.gesehen.add(sha)
        e["sha256"] = sha
        try:
            if sperrwort(e.get("prompt")):
                e = {"sha256": sha, "grund": "gesperrt"}
            else:
                im = Image.open(io.BytesIO(daten)).convert("RGB")
                e["w"], e["h"] = im.size
                lab, sc = self.ar(im)
                e["real"] = round(float(sc if lab == "real" else 1 - sc), 3)
                if e["real"] >= REAL_MIN:
                    e["grund"] = "foto"
                else:
                    rating, alle, _ = self.wd(im, model_name="EVA02_Large", general_threshold=SCHWELLE)
                    allg = {t for t, v in alle.items() if v >= 0.35}
                    e["rating"] = max(rating, key=rating.get)
                    if gesperrt_wd(alle) or (figurenregel_gilt(e["rating"]) and gesperrt_text(e.get("prompt"))):
                        e = {"sha256": sha, "grund": "gesperrt"}
                    elif "photo_(medium)" in allg:
                        e["grund"] = "stil: photo_(medium)"
                    else:
                        e["tags"] = sorted(allg & (STIL_MERKEN | {"realistic", "3d", "photorealistic"}))
                        jpg, gr = hochladen_bytes(im)
                        pfad = os.path.join(self.ziel, unterordner, sha[:24] + ".jpg")
                        os.makedirs(os.path.dirname(pfad), exist_ok=True)
                        with open(pfad, "wb") as f:
                            f.write(jpg)
                        e["datei"], e["gw"], e["gh"] = pfad, gr[0], gr[1]
        except Exception as ex:
            e["grund"] = f"fehler: {type(ex).__name__}"
        self.mf.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.mf.flush()
        self.stand[e.get("grund", "ok").split(":")[0]] += 1


def felix():
    repo, ziel = QUELLEN["felix"]
    os.makedirs(ziel, exist_ok=True)
    kat = os.path.join(ziel, "catalog.csv")
    if not os.path.exists(kat):
        holen(HF + repo + "/resolve/main/catalog.csv", kat)
    csv.field_size_limit(10 ** 8)
    katalog = {r["\ufeffimage_id"] if "\ufeffimage_id" in r else r["image_id"]: r
               for r in csv.DictReader(open(kat, encoding="utf-8"))}
    baum = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main/data", timeout=60).json()
    shards = sorted(x["path"] for x in baum if x["path"].endswith(".tar"))
    p = Pruefer(ziel)
    t0 = time.time()
    for sh in shards:
        fertig = os.path.join(ziel, "." + os.path.basename(sh) + ".fertig")
        if os.path.exists(fertig):
            continue
        tp = os.path.join(ziel, os.path.basename(sh))
        if not holen(HF + repo + "/resolve/main/" + sh, tp):
            print(f"  {sh} uebersprungen", flush=True)
            continue
        with tarfile.open(tp) as tf:
            for m in tf:
                if not m.isfile() or not m.name.lower().endswith(BILD):
                    continue
                key = os.path.splitext(os.path.basename(m.name))[0]
                k = katalog.get(key, {})
                e = {"id": key, "quelle": "midjourney_felix", "lizenz": "CC0-1.0", "ersteller": "Felix / waffles13",
                     "prompt": k.get("prompt", ""), "version": k.get("model_version", ""),
                     "zeit": k.get("created_at", ""), "job": k.get("midjourney_job_id", ""),
                     "shard": sh, "url": HF + repo}
                ordner = (e["zeit"][:4] or "unbekannt")
                p.bild(tf.extractfile(m).read(), e, ordner)
        os.remove(tp)
        open(fertig, "w").close()
        print(f"{sh}: bisher {dict(p.stand)}  {(time.time()-t0)/60:.0f} min", flush=True)
    return p.stand


def niji():
    repo, ziel = QUELLEN["niji"]
    os.makedirs(ziel, exist_ok=True)
    baum = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true", timeout=60).json()
    zips = sorted(x["path"] for x in baum if x["path"].endswith(".zip"))
    p = Pruefer(ziel)
    for zpfad in zips:
        fertig = os.path.join(ziel, "." + zpfad.replace("/", "_") + ".fertig")
        if os.path.exists(fertig):
            continue
        lokal = os.path.join(ziel, zpfad.replace("/", "_"))
        if not holen(HF + repo + "/resolve/main/" + requests.utils.quote(zpfad), lokal):
            continue
        with zipfile.ZipFile(lokal) as z:
            namen = [n for n in z.namelist() if n.lower().endswith(BILD)]
            for n in sorted(namen):
                txt = os.path.splitext(n)[0] + ".txt"
                prompt = z.read(txt).decode("utf-8", "replace") if txt in z.namelist() else ""
                e = {"id": n, "quelle": "nijijourney_p1atdev", "lizenz": "CC0-1.0", "ersteller": "p1atdev",
                     "prompt": prompt, "version": "niji 5", "archiv": zpfad, "url": HF + repo}
                p.bild(z.read(n), e, os.path.splitext(zpfad.replace("/", "_"))[0])
        os.remove(lokal)
        open(fertig, "w").close()
        print(f"{zpfad}: bisher {dict(p.stand)}", flush=True)
    return p.stand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nur", choices=["felix", "niji"])
    a = ap.parse_args()
    erg = {}
    if a.nur in (None, "niji"):
        erg["niji"] = dict(niji())
    if a.nur in (None, "felix"):
        erg["felix"] = dict(felix())
    print(f"FERTIG: {erg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
