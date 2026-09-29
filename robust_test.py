r"""robust_test.py — robustness matrix: how do everyday edits change hit rate and false positives?

Same images as klein_test.py (600 each from three human test groups, up to 120 per Telegram AI channel), each
edited once per condition and then normalized as the scanner does (512 px, JPEG q95 4:4:4). Measured at the
head's 1 % threshold, plus AUC (humans vs. channels) per condition. Conditions:

  jpeg50 / jpeg30      heavy recompression (re-uploads, messengers)
  social               downscale to 0.6x + JPEG q75 (typical social-media pipeline)
  crop50               centre crop to half width/height
  text                 caption band + semi-transparent watermark text
  noise                Gaussian noise sigma 6 (a common "detector evasion" trick)
  blur / sharpen       Gaussian blur r=1.2 / unsharp mask
  colour               saturation +35 %, brightness +10 %
  grey                 greyscale
  flip                 horizontal mirror (sanity check, should change nothing)
  upscale_cdc          4x anime upscaler (imgutils CDC HGSR), then back to 512 — AI images are often upscaled

    python robust_test.py [--kopf p8sx] [--nur jpeg50,noise]
"""
import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from lesen import DATEN, Pack, kanal_X           # noqa: E402
from norm import LANG, QUALITAET, normiere       # noqa: E402
from pruefe import stapelrechner                 # noqa: E402


def jpeg(b, q, sub=2):
    buf = io.BytesIO()
    b.save(buf, "JPEG", quality=q, subsampling=sub)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def skaliert(b, f):
    return b.resize((max(1, round(b.size[0] * f)), max(1, round(b.size[1] * f))), Image.LANCZOS)


def scanner(b):
    """Normierung wie norm.normiere: lange Seite LANG, JPEG QUALITAET 4:4:4"""
    return jpeg(skaliert(b, LANG / max(b.size)), QUALITAET, 0)


def text(b, seed):
    b = b.copy()
    w, h = b.size
    d = ImageDraw.Draw(b, "RGBA")
    try:
        f = ImageFont.truetype("arial.ttf", max(12, h // 18))
        f2 = ImageFont.truetype("arial.ttf", max(14, h // 9))
    except OSError:
        f = f2 = ImageFont.load_default()
    d.rectangle((0, int(h * 0.88), w, h), fill=(0, 0, 0, 170))
    d.text((int(w * 0.03), int(h * 0.9)), "when you finally finish the drawing lol", font=f, fill=(255, 255, 255, 255))
    d.text((int(w * 0.25), int(h * 0.4)), f"@artist_{seed % 997}", font=f2, fill=(255, 255, 255, 90))
    return b


def rauschen(b, seed):
    a = np.asarray(b, np.float32)
    a += np.random.default_rng(seed).normal(0, 6, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def mitte(b, f):
    w, h = b.size
    return b.crop((int(w * (1 - f) / 2), int(h * (1 - f) / 2), int(w * (1 + f) / 2), int(h * (1 + f) / 2)))


BEDINGUNGEN = {
    "original": lambda b, s: b,
    "jpeg50": lambda b, s: jpeg(b, 50),
    "jpeg30": lambda b, s: jpeg(b, 30),
    "social": lambda b, s: jpeg(skaliert(b, 0.6), 75),
    "crop50": lambda b, s: mitte(b, 0.5),
    "text": lambda b, s: text(b, s),
    "noise": lambda b, s: rauschen(b, s),
    "blur": lambda b, s: b.filter(ImageFilter.GaussianBlur(1.2)),
    "sharpen": lambda b, s: b.filter(ImageFilter.UnsharpMask(2, 120, 2)),
    "colour": lambda b, s: ImageEnhance.Brightness(ImageEnhance.Color(b).enhance(1.35)).enhance(1.1),
    "grey": lambda b, s: ImageOps.grayscale(b).convert("RGB"),
    "flip": lambda b, s: ImageOps.mirror(b),
    "upscale_cdc": None,   # GPU, im Hauptthread
}


def auc(m, k):
    x = np.concatenate([m, k])
    r = x.argsort().argsort().astype(np.float64) + 1
    return float((r[len(m):].sum() - len(k) * (len(k) + 1) / 2) / (len(m) * len(k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kopf", default="p8sx")
    ap.add_argument("--nur", default="")
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
    kan = sorted(f[6:-8] for f in os.listdir(DATEN) if f.startswith("kanal_") and f.endswith("_emb.npz"))
    for name in kan:
        _, pfade = kanal_X(name)
        if len(pfade) < 50:
            continue
        for j in rng.choice(len(pfade), min(120, len(pfade)), replace=False):
            gruppen.setdefault("K Telegram-Kanaele", []).append(
                lambda p=str(pfade[j]): Image.open(io.BytesIO(normiere(p)[0])).convert("RGB"))
    namen = [n for n in BEDINGUNGEN if not a.nur or n in a.nur.split(",") or n == "original"]
    ziel = os.path.join(DATEN, f"robust_test_{a.kopf}.json")
    erg = json.load(open(ziel)) if os.path.exists(ziel) else {}
    cdc = None
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        bilder = {g: list(ex.map(lambda f: f(), laden)) for g, laden in gruppen.items()}
        print(f"{sum(map(len, bilder.values()))} Bilder geladen  {(time.time()-t0)/60:.0f} min", flush=True)
        for bed in namen:
            if bed in erg and bed != "original" and not a.nur:
                continue
            z = {}
            for g, bs in bilder.items():
                if bed == "upscale_cdc":
                    if cdc is None:
                        from imgutils.upscale import upscale_with_cdc
                        cdc = upscale_with_cdc
                    neu = [scanner(cdc(b, silent=True)) for b in bs]
                else:
                    fn = BEDINGUNGEN[bed]
                    neu = list(ex.map(lambda ib: scanner(fn(ib[1], ib[0])), enumerate(bs)))
                X = np.concatenate([np.concatenate([rechner[r](neu[s:s + 32], ex) for r in rg], 1)
                                    for s in range(0, len(neu), 32)])
                z[g] = bewerte(X, k).astype(np.float32)
            m = np.concatenate([v for g, v in z.items() if g.startswith("M ")])
            kk = z["K Telegram-Kanaele"]
            erg[bed] = {"treffer": float((kk > schw).mean()), "fa": float((m > schw).mean()), "auc": auc(m, kk),
                        "fa_gruppen": {g: float((v > schw).mean()) for g, v in z.items() if g.startswith("M ")}}
            json.dump(erg, open(ziel, "w"), indent=1)
            print(f"  {bed:<12} Treffer {erg[bed]['treffer']:6.1%}  FA {erg[bed]['fa']:5.2%}  AUC {erg[bed]['auc']:.4f}"
                  f"   {(time.time()-t0)/60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
