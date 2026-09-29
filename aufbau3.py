r"""aufbau3.py — build supplementary packs (web AI sources, current human art, later phases).

The base pack v2 stays unchanged; supplementary packs are added at training time (train3.py --zusatz), so a new
source never requires re-packing and re-embedding everything.

Split by CREATOR (CivitAI user, AIBooru uploader, Danbooru artist, DiffusionDB user, Midjourney job/prompt):
a creator has a style, favourite LoRAs, series — if the same creator were in training and test, the test would
measure the creator. Fixed crc32 split: 10 % test, 5 % val, rest train (Danbooru sets use their own splits).

Files already carry the upload simulation (downloaders), so hoch=False. schutz.py was applied at download; rows
whose manifest says "gesperrt" or whose file is missing are dropped here as well.

    python aufbau3.py --plan
    python aufbau3.py --p2 --stamm p2z     (also --mensch, --p3, --p4, --p5)
"""
import argparse
import collections
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from aufbau2 import _arbeit                      # noqa: E402
from lesen import DATEN                          # noqa: E402

QUELLEN = {"civitai": f"{DATASETS}/civitai/manifest.jsonl",
           "aibooru": f"{DATASETS}/aibooru/manifest.jsonl",
           # menschlich, 2022 bis 03/2026, Kuenstler disjunkt zum Danbooru-Test
           # (danbooru_train.py) — gegen "neuer Zeichenstil = KI"
           "danbooru_train": f"{DATASETS}/danbooru_train/manifest.jsonl"}
# Menschlich nach 2022, NUR Test und Schwellen-Kalibrierung (danbooru.py legt val/test
# je Kuenstler fest): eigenes Pack, damit es nie versehentlich ins Training geraet
QUELLEN_MENSCH = {"danbooru": f"{DATASETS}/danbooru_ab2023/manifest.jsonl"}
# Phase 2 (digitale Kunst allgemein): Splits kommen aus den Manifesten bzw. je Ersteller
QUELLEN_P2 = {"civitai_stil": f"{DATASETS}/civitai_stil/manifest.jsonl",
              "danbooru_nichtanime": f"{DATASETS}/danbooru_nichtanime/manifest.jsonl",
              "wallhaven_general": f"{DATASETS}/wallhaven_general/manifest.jsonl"}
# Phase 3: fruehe KI von 2022 (DiffusionDB, SD 1.x im Discord, CC0), Split je Discord-Nutzer
QUELLEN_P3 = {"diffusiondb": f"{DATASETS}/diffusiondb/manifest.jsonl"}
# Phase 4: Midjourney/Niji eigener Ersteller (CC0); Phase 5: MJ-v6-Stichprobe aus Scrape-Datensatz.
# Je Quelle nur ein Ersteller bzw. keine Nutzer-ID -> Split je Job-ID/Prompt, damit Varianten zusammenbleiben
QUELLEN_P4 = {"midjourney_felix": f"{DATASETS}/midjourney_felix/manifest.jsonl",
              "nijijourney_p1atdev": f"{DATASETS}/nijijourney_p1atdev/manifest.jsonl"}
QUELLEN_P5 = {"midjourney_v6_scrape": f"{DATASETS}/midjourney_v6_scrape/manifest.jsonl"}


def split_von(quelle, user):
    r = zlib.crc32(f"{quelle}:{user}".encode()) % 20
    return "test" if r < 2 else "val" if r == 2 else "train"


def zeilen_lesen(quellen=None):
    zeilen = []
    for q, mf in (quellen or QUELLEN).items():
        if not os.path.exists(mf):
            continue
        for z in open(mf, encoding="utf-8"):
            e = json.loads(z)
            if not e.get("datei") or e.get("grund") or not os.path.exists(e["datei"]):
                continue
            if q == "danbooru":
                zeilen.append(dict(wh_id="", model="mensch", label=0, split=e["split"],
                                   grp=f"danbooru:{e.get('artist') or e['id']}", src=e["datei"], quelle=q,
                                   hoch=False, nsfw=e.get("rating"), tags=e.get("tags", []),
                                   anfrage=e.get("anfrage")))
                continue
            if q == "danbooru_nichtanime":
                zeilen.append(dict(wh_id="", model="mensch", label=0, split=e["split"],
                                   grp=f"danbooru:{e.get('artist') or e['id']}", src=e["datei"], quelle=q,
                                   hoch=False, nsfw=e.get("rating"), tags=e.get("tags", []),
                                   stil=e.get("danbooru_tags", ""), fenster=e.get("fenster")))
                continue
            if q == "wallhaven_general":
                zeilen.append(dict(wh_id=e["name"], model="mensch", label=0, split=e["split"],
                                   grp=f"wallhaven:{e['name']}", src=e["datei"], quelle=q, hoch=False,
                                   nsfw=e.get("rating"), tags=e.get("tags", [])))
                continue
            if q == "civitai_stil":
                zeilen.append(dict(wh_id="", model=e.get("basis"), label=1, split=split_von("civitai", e.get("user")),
                                   grp=f"civitai:{e.get('user')}", src=e["datei"], quelle=q, hoch=False,
                                   nsfw=e.get("nsfw") or e.get("rating"), tags=e.get("tags", [])))
                continue
            if q in ("midjourney_felix", "nijijourney_p1atdev", "midjourney_v6_scrape"):
                gk = e.get("job") or (e.get("prompt") or "")[:200] or e["sha256"]
                zeilen.append(dict(wh_id="", model={"midjourney_felix": "Midjourney (Felix)",
                                                    "nijijourney_p1atdev": "Nijijourney v5",
                                                    "midjourney_v6_scrape": "Midjourney v6"}[q], label=1,
                                   split=split_von(q, gk), grp=f"{q}:{gk}", src=e["datei"], quelle=q, hoch=False,
                                   nsfw=e.get("rating"), tags=e.get("tags", []), ki_modell=e.get("version", "")))
                continue
            if q == "diffusiondb":
                zeilen.append(dict(wh_id="", model="SD 1.x 2022 (DiffusionDB)", label=1,
                                   split=split_von(q, e.get("user")), grp=f"{q}:{e.get('user')}", src=e["datei"],
                                   quelle=q, hoch=False, nsfw=e.get("rating"), tags=e.get("tags", [])))
                continue
            if q == "danbooru_train":
                k = e.get("artist") or f"unbekannt:{e['id']}"
                zeilen.append(dict(wh_id="", model="mensch", label=0,
                                   split="val" if zlib.crc32(k.encode()) % 20 == 0 else "train",
                                   grp=f"danbooru:{k}", src=e["datei"], quelle=q, hoch=False,
                                   nsfw=e.get("rating"), tags=e.get("tags", []), fenster=e.get("fenster")))
                continue
            basis = e.get("basis") if q == "civitai" else "aibooru"
            zeilen.append(dict(wh_id="", model=basis, label=1, split=split_von(q, e.get("user")),
                               grp=f"{q}:{e.get('user')}", src=e["datei"], quelle=q, hoch=False,
                               nsfw=e.get("nsfw") or e.get("rating"), tags=e.get("tags", []),
                               ki_modell=e.get("model", "")))
    return zeilen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--stamm", default="v3z")
    ap.add_argument("--worker", type=int, default=16)
    ap.add_argument("--mensch", action="store_true", help="Danbooru-Testpack statt Web-KI")
    ap.add_argument("--p2", action="store_true", help="Phase-2-Quellen (civitai_stil, danbooru_nichtanime, "
                                                      "wallhaven_general)")
    ap.add_argument("--p3", action="store_true", help="Phase-3-Quelle diffusiondb")
    ap.add_argument("--p4", action="store_true", help="Phase-4-Quellen Midjourney Felix + Niji p1atdev")
    ap.add_argument("--p5", action="store_true", help="Phase-5-Quelle Midjourney-v6-Stichprobe")
    ap.add_argument("--ohne", default="", help="Packs (mit Komma), deren Bilder schon gepackt sind — "
                                               "fuer kleine Aufstockungs-Packs")
    args = ap.parse_args()

    auswahl = (QUELLEN_MENSCH if args.mensch else QUELLEN_P2 if args.p2 else QUELLEN_P3 if args.p3
               else QUELLEN_P4 if args.p4 else QUELLEN_P5 if args.p5 else QUELLEN)
    zeilen = zeilen_lesen(auswahl)
    if args.ohne:
        schon = set()
        for n in args.ohne.split(","):
            schon |= {z["src"] for z in json.load(open(os.path.join(DATEN, n + "_manifest.json"), encoding="utf-8"))}
        vorher = len(zeilen)
        zeilen = [z for z in zeilen if z["src"] not in schon]
        print(f"ohne {args.ohne}: {vorher - len(zeilen)} schon gepackt, {len(zeilen)} neu")
    tab = collections.Counter((z["quelle"], z["split"]) for z in zeilen)
    print(f"{'Quelle':<10}{'train':>9}{'val':>8}{'test':>8}   Ersteller")
    for q in auswahl:
        n_u = len({z["grp"] for z in zeilen if z["quelle"] == q})
        print(f"{q:<10}" + "".join(f"{tab[(q, s)]:>{9 if s == 'train' else 8}}"
                                   for s in ("train", "val", "test")) + f"   {n_u}")
    print("Basismodelle:", collections.Counter(z["model"] for z in zeilen).most_common())
    if args.plan:
        return 0

    zeilen.sort(key=lambda z: z["src"].lower())
    n = len(zeilen)
    off, lng = np.zeros(n, np.int64), np.zeros(n, np.int32)
    groesse = np.zeros((n, 4), np.int32)
    fehler = collections.Counter()
    t0, pos = time.time(), 0
    with open(os.path.join(DATEN, args.stamm + ".bin"), "wb") as f, \
            ProcessPoolExecutor(max_workers=args.worker) as ex:
        for i, erg, err in ex.map(_arbeit, ((i, z["src"], False) for i, z in enumerate(zeilen)),
                                  chunksize=32):
            if erg is None:
                fehler[err.split(":")[0]] += 1
                off[i] = -1
            else:
                b, w0, h0, w, h = erg
                f.write(b)
                off[i], lng[i] = pos, len(b)
                groesse[i] = (w0, h0, w, h)
                pos += len(b)
            if (i + 1) % 10000 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{n}  {(i+1)/el:.0f}/s  {pos/2**30:.1f} GB", flush=True)
    np.save(os.path.join(DATEN, args.stamm + "_off.npy"), off)
    np.save(os.path.join(DATEN, args.stamm + "_len.npy"), lng)
    np.save(os.path.join(DATEN, args.stamm + "_size.npy"), groesse)
    for i, z in enumerate(zeilen):
        z["i"] = i
    with open(os.path.join(DATEN, args.stamm + "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(zeilen, f)
    print(f"\n{n - sum(fehler.values())} von {n} gepackt, {pos/2**30:.1f} GB, "
          f"{(time.time()-t0)/60:.0f} min. Verworfen: {dict(fehler)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
