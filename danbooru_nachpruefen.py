r"""danbooru_nachpruefen.py — are "human" Danbooru images that the head flags actually AI?

Danbooru bans AI images, but some slip through and are tagged or deleted later. All Danbooru human images
(training, calibration, test) above the head 1 % threshold are looked up via the Danbooru API (100 IDs per
request, polite pacing). Posts WITH an AI tag go to daten/falsch_menschlich.txt (--anwenden), which lesen.Pack
excludes everywhere. Result 2026-09-28 (p8sx): 298 posts above the threshold, none tagged AI or deleted.

    python danbooru_nachpruefen.py [--kopf p8sx] [--anwenden]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train3                                     # noqa: E402
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from lesen import DATEN, Pack                    # noqa: E402
from p2_test import merkmale                     # noqa: E402

H = {"User-Agent": "ai-art-detector-lab/0.1 (research)"}
KI_TAGS = {"ai-generated", "ai-assisted", "ai_generated", "ai_assisted"}


def verdaechtige(kopf):
    """[(post_id, quelle, split, score)] ueber der 1-%-Schwelle."""
    k = koepfe_von(kopf)
    s1 = float(k[0]["schwelle_1"])
    train3.RG = tuple(str(r) for r in k[0]["rueckgrate"])
    out = []
    for st, q in (("v3z2", "danbooru_train"), ("p2z", "danbooru_nichtanime"), ("v3d", "danbooru")):
        p = Pack(st)
        for split in ("train", "val", "test"):
            m = p.ok & (p.quelle == q) & (p.split == split) & (p.label == 0)
            if not m.any():
                continue
            if st == "v3d":
                X, ok = merkmale("v3d")
            elif split == "test":
                X, ok = merkmale(st, "test")
            else:                                  # Trainings-/val-Merkmale gibt es nur gestoert (aug)
                teile, ok = train3.lade(st, True)
                from p2_test import Spalten
                X = Spalten(teile)
            idx = np.flatnonzero(m & ok)
            z = np.concatenate([bewerte(np.asarray(X[idx[i:i + 4000]]), k) for i in range(0, len(idx), 4000)])
            for j in np.flatnonzero(z > s1):
                pid = int(os.path.splitext(os.path.basename(p.m[idx[j]]["src"]))[0])
                out.append((pid, q, split, float(z[j]), p.m[idx[j]]["src"]))
            print(f"  {q:<20} {split:<5} {len(idx):>6} Bilder, {int((z > s1).sum()):>4} ueber der Schwelle", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kopf", default="p8sx")
    ap.add_argument("--anwenden", action="store_true")
    a = ap.parse_args()
    v = verdaechtige(a.kopf)
    ids = sorted({x[0] for x in v})
    print(f"{len(ids)} Posts nachschlagen", flush=True)
    s = requests.Session()
    s.headers.update(H)
    info = {}
    for i in range(0, len(ids), 100):
        teil = ids[i:i + 100]
        for versuch in range(5):
            r = s.get("https://danbooru.donmai.us/posts.json",
                      params={"tags": "id:" + ",".join(map(str, teil)) + " status:any", "limit": 200}, timeout=60)
            if r.status_code == 429:
                time.sleep(10 * (versuch + 1))
                continue
            r.raise_for_status()
            break
        for post in r.json():
            tags = set(post.get("tag_string", "").split())
            info[post["id"]] = {"ki": sorted(tags & KI_TAGS), "geloescht": bool(post.get("is_deleted")),
                                "gebannt": bool(post.get("is_banned")), "rating": post.get("rating")}
        time.sleep(1.5)
    ki = [x for x in v if info.get(x[0], {}).get("ki")]
    geloescht = [x for x in v if info.get(x[0], {}).get("geloescht") and not info.get(x[0], {}).get("ki")]
    fehlt = [x for x in v if x[0] not in info]
    print(f"\nueber der Schwelle: {len(v)} Bilder / {len(ids)} Posts")
    print(f"  inzwischen als KI getaggt: {len(ki)}")
    print(f"  geloescht (ohne KI-Tag):   {len(geloescht)}")
    print(f"  nicht gefunden:            {len(fehlt)}")
    for q in ("danbooru_train", "danbooru_nichtanime", "danbooru"):
        print(f"    {q:<20} KI getaggt {sum(1 for x in ki if x[1] == q):>4} von {sum(1 for x in v if x[1] == q)}")
    json.dump({"verdaechtig": v, "info": {str(k): w for k, w in info.items()}},
              open(os.path.join(DATEN, "danbooru_nachpruefen.json"), "w", encoding="utf-8"), indent=1)
    if a.anwenden and ki:
        liste = os.path.join(DATEN, "falsch_menschlich.txt")
        alt = set(open(liste, encoding="utf-8").read().split("\n")) if os.path.exists(liste) else set()
        neu = sorted({x[4] for x in ki} - alt)
        with open(liste, "a", encoding="utf-8") as f:
            for pfad in neu:
                f.write(pfad + "\n")
        print(f"{len(neu)} Pfade -> {liste}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
