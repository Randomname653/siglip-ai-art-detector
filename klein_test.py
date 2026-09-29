r"""klein_test.py — how does a head behave on images BELOW 512 px (never seen in training)?

Same images (human test groups, Telegram AI channels) once normal and once first downscaled to 480/384/256 px
(JPEG q90, like a small web image), then upscaled to 512 as the scanner would. Measured at the head threshold.
Result for p8sx: hit rate ~97 % down to 256 px; false positives unchanged down to 384 px, 2-3x at 256 px.

    python klein_test.py [--kopf p8sx] [--groessen 480,384,256]
"""
import argparse
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from lesen import DATEN, Pack, kanal_X           # noqa: E402
from norm import LANG, QUALITAET                 # noqa: E402
from pruefe import stapelrechner                 # noqa: E402

TELEGRAM_KANAELE = None


def verkleinern(bild, lang):
    """wie ein kleines Web-Bild: lange Seite = lang, JPEG q90; dann Scanner-Normierung (auf 512, q95 4:4:4)"""
    w, h = bild.size
    f = lang / max(w, h)
    klein = bild.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)
    buf = io.BytesIO()
    klein.save(buf, "JPEG", quality=90)
    klein = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    f = LANG / max(klein.size)
    gross = klein.resize((max(1, round(klein.size[0] * f)), max(1, round(klein.size[1] * f))), Image.LANCZOS)
    buf = io.BytesIO()
    gross.save(buf, "JPEG", quality=QUALITAET, subsampling=0)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kopf", default="p8sx")
    ap.add_argument("--groessen", default="480,384,256")
    a = ap.parse_args()
    k = koepfe_von(a.kopf)
    rg = [str(r) for r in k[0]["rueckgrate"]]
    schw = float(k[0]["schwelle_1"])
    rechner = stapelrechner(rg)
    rng = np.random.default_rng(8)
    gruppen = {}
    pd, pp = Pack("v3d"), Pack("p2z")
    i = rng.choice(pd.wo(split="test"), 600, replace=False)
    gruppen["M Danbooru-Anime 2026"] = [lambda j=j: pd.bild(j) for j in i]
    for q in ("wallhaven_general", "danbooru_nichtanime"):
        i = rng.choice(np.flatnonzero(pp.ok & (pp.split == "test") & (pp.quelle == q)), 600, replace=False)
        gruppen[f"M {q}"] = [lambda j=j: pp.bild(j) for j in i]
    from norm import normiere
    kan = sorted(f[6:-8] for f in os.listdir(DATEN) if f.startswith("kanal_") and f.endswith("_emb.npz"))
    for name in kan:
        _, pfade = kanal_X(name)
        if len(pfade) < 50:
            continue
        for j in rng.choice(len(pfade), min(120, len(pfade)), replace=False):
            gruppen.setdefault("K Telegram-Kanaele", []).append(
                lambda p=str(pfade[j]): Image.open(io.BytesIO(normiere(p)[0])).convert("RGB"))
    groessen = [0] + [int(g) for g in a.groessen.split(",")]
    erg = {}
    with ThreadPoolExecutor(8) as ex:
        for g, laden in gruppen.items():
            bilder = list(ex.map(lambda f: f(), laden))
            erg[g] = {}
            for gr in groessen:
                bs = bilder if gr == 0 else list(ex.map(lambda b: verkleinern(b, gr), bilder))
                X = np.concatenate([np.concatenate([rechner[r](bs[s:s + 32], ex) for r in rg], 1)
                                    for s in range(0, len(bs), 32)])
                erg[g][gr] = float(np.mean(bewerte(X, k) > schw))
            print(f"  {g:<28} n={len(bilder):>5}  " + "  ".join(
                f"{'original' if gr == 0 else f'{gr}px'}: {erg[g][gr]:6.1%}" for gr in groessen), flush=True)
    json.dump(erg, open(os.path.join(DATEN, f"klein_test_{a.kopf}.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
