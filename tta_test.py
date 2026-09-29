r"""tta_test.py — does test-time augmentation reduce false positives at equal hit rate?

Each image is scored in several views (original, horizontal flip, 90 % centre crop, 85 % downscale); the ensemble
logit per image is the mean (or the minimum, "conservative") over views. Compared at EQUAL false-positive rate on
human test images (four groups incl. the in-domain Telegram humans before 2022, equally weighted): hit rate on the
Telegram AI channels, plus AUC. Views are normalized like the scanner (512 px, JPEG q95 4:4:4).

    python tta_test.py [--kopf p8sx] [--n 3000]
"""
import argparse
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from lesen import DATEN, Pack, kanal_X           # noqa: E402
from norm import normiere                        # noqa: E402
from pruefe import stapelrechner                 # noqa: E402
from robust_test import auc, mitte, scanner, skaliert   # noqa: E402

ANSICHTEN = {"original": lambda b: b, "flip": ImageOps.mirror, "crop90": lambda b: mitte(b, 0.9),
             "scale85": lambda b: skaliert(b, 0.85)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kopf", default="p8sx")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    k = koepfe_von(a.kopf)
    rg = [str(r) for r in k[0]["rueckgrate"]]
    rechner = stapelrechner(rg)
    rng = np.random.default_rng(11)
    gruppen = {}
    pd, p2, pz = Pack("v3d"), Pack("v2"), Pack("p2z")
    i = rng.choice(pd.wo(split="test"), min(a.n, len(pd.wo(split="test"))), replace=False)
    gruppen["M Danbooru-Anime 2026"] = [lambda j=j: pd.bild(j) for j in i]
    tg = np.flatnonzero(p2.ok & (p2.split == "test") & (p2.quelle == "telegram"))
    gruppen["M Telegram <2022"] = [lambda j=j: p2.bild(j) for j in rng.choice(tg, a.n, replace=False)]
    for q in ("wallhaven_general", "danbooru_nichtanime"):
        m = np.flatnonzero(pz.ok & (pz.split == "test") & (pz.quelle == q))
        gruppen[f"M {q}"] = [lambda j=j: pz.bild(j) for j in rng.choice(m, a.n, replace=False)]
    kan = sorted(f[6:-8] for f in os.listdir(DATEN) if f.startswith("kanal_") and f.endswith("_emb.npz"))
    for name in kan:
        _, pfade = kanal_X(name)
        if len(pfade) >= 50:
            for j in rng.choice(len(pfade), min(150, len(pfade)), replace=False):
                gruppen.setdefault("K Kanaele", []).append(
                    lambda p=str(pfade[j]): Image.open(io.BytesIO(normiere(p)[0])).convert("RGB"))
    Z = {}
    with ThreadPoolExecutor(8) as ex:
        for g, laden in gruppen.items():
            bilder = list(ex.map(lambda f: f(), laden))
            Z[g] = {}
            for v, fn in ANSICHTEN.items():
                neu = list(ex.map(lambda b: scanner(fn(b)), bilder))
                X = np.concatenate([np.concatenate([rechner[r](neu[s:s + 32], ex) for r in rg], 1)
                                    for s in range(0, len(neu), 32)])
                Z[g][v] = bewerte(X, k)
            print(f"  {g}: {len(bilder)} Bilder", flush=True)
    np.savez(os.path.join(DATEN, f"tta_{a.kopf}.npz"), **{f"{g}|{v}": z for g, d in Z.items() for v, z in d.items()})
    menschen = [g for g in Z if g.startswith("M ")]
    varianten = {"nur original": lambda d: d["original"],
                 "mittel orig+flip": lambda d: (d["original"] + d["flip"]) / 2,
                 "mittel alle 4": lambda d: np.mean([d[v] for v in ANSICHTEN], 0),
                 "minimum alle 4": lambda d: np.min([d[v] for v in ANSICHTEN], 0),
                 "median alle 4": lambda d: np.median([d[v] for v in ANSICHTEN], 0)}
    erg = {}
    print(f"\n{'Variante':<20}{'AUC':>8}" + "".join(f"{'Treffer@' + str(z) + '%':>14}" for z in (1, 0.5, 0.25, 0.1)))
    for name, f in varianten.items():
        m = {g: f(Z[g]) for g in menschen}
        kk = f(Z["K Kanaele"])
        zeile = {"auc": float(np.mean([auc(m[g], kk) for g in menschen]))}
        for ziel in (0.01, 0.005, 0.0025, 0.001):
            lo, hi = -100.0, 100.0
            for _ in range(60):
                t = (lo + hi) / 2
                if np.mean([np.mean(m[g] > t) for g in menschen]) > ziel:
                    lo = t
                else:
                    hi = t
            zeile[str(ziel)] = float(np.mean(kk > hi))
        erg[name] = zeile
        print(f"{name:<20}{zeile['auc']:8.4f}" + "".join(f"{zeile[str(z)]:14.1%}" for z in (0.01, 0.005, 0.0025, 0.001)))
    json.dump(erg, open(os.path.join(DATEN, f"tta_{a.kopf}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
