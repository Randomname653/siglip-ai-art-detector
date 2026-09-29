r"""profikunst_datum.py — split the pro-art false positives by provable age and remove duplicates.

The NAS copies carry no useful file dates (all 2025/26), but many originals embed a creation date (Photoshop XMP
CreateDate / EXIF DateTimeOriginal). Images dated before 2022 cannot be generator output, so their false-positive
rate is the clean number; images dated 2022+ or undated may contain undisclosed AI-assisted work.
Duplicates (the same wallpaper filed under two artist folders) are removed via 32x32 pixel thumbnails.
Uses the features cached by profikunst_test.py.

    python profikunst_datum.py [--koepfe p8sx,p8x]
"""
import argparse
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS, NAS                   # noqa: E402
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from lesen import DATEN                          # noqa: E402

MF = f"{DATASETS}/nas_profikunst/manifest.jsonl"
CACHE = os.path.join(DATEN, "profikunst_merkmale.npz")
DATUM = re.compile(rb'(?:CreateDate|DateTimeOriginal|DateCreated)(?:>|="|\x00+)\s*(\d{4})')
EXIF = re.compile(rb'((?:19|20)\d\d):[01]\d:[0-3]\d [0-2]\d:\d\d:\d\d')


def jahr(pfad):
    """Earliest plausible year embedded in the file header (XMP/EXIF), or None."""
    try:
        with open(pfad, "rb") as f:
            b = f.read(600_000)
    except OSError:
        return None
    js = [int(m.group(1)) for m in DATUM.finditer(b)] + [int(m.group(1)) for m in EXIF.finditer(b)]
    js = [j for j in js if 1995 <= j <= 2026]
    return min(js) if js else None


def daumen(pfad):
    im = Image.open(pfad).convert("L").resize((32, 32), Image.BILINEAR)
    v = np.asarray(im, np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--koepfe", default="p8sx,p8x")
    a = ap.parse_args()
    E = [json.loads(z) for z in open(MF, encoding="utf-8")]
    E = [e for e in E if e.get("datei") and not e.get("grund")]
    z = np.load(CACHE)
    assert list(z["pfade"]) == [e["datei"] for e in E]
    X = z["X"]
    with ThreadPoolExecutor(12) as ex:
        J = list(ex.map(lambda e: jahr(f"{NAS}/{e['datei_nas']}"), E))
        D = np.stack(list(ex.map(lambda e: daumen(e["datei"]), E)))
    # Dubletten: jeweils das erste Bild behalten
    behalten = np.ones(len(E), bool)
    for s in range(0, len(E), 2048):
        S = D[s:s + 2048] @ D.T
        for i in range(S.shape[0]):
            gi = s + i
            if behalten[gi] and (S[i, gi + 1:] >= 0.97).any():
                behalten[gi + 1:][S[i, gi + 1:] >= 0.97] = False
    print(f"{len(E)} Bilder, {int((~behalten).sum())} Dubletten entfernt", flush=True)
    quelle = np.array([e["quelle"] for e in E])
    alter = np.array(["vor 2022" if j and j < 2022 else ("ab 2022" if j else "undatiert") for j in J])
    aufl = np.array([re.search(r"_(\d{3,4})x(\d{3,4})", e["datei_nas"]) for e in E], object)
    breite = np.array([int(m.group(1)) if m else 0 for m in aufl])
    erg = {"dubletten": int((~behalten).sum()), "koepfe": {}}
    for tag in a.koepfe.split(","):
        k = koepfe_von(tag)
        f = bewerte(X, k) > float(k[0]["schwelle_1"])
        zeilen = {}
        for q in ("cgwallpapers", "gamewallpapers", "alle"):
            for al in ("vor 2022", "ab 2022", "undatiert", "alle"):
                m = behalten & ((quelle == q) if q != "alle" else True) & ((alter == al) if al != "alle" else True)
                if m.sum():
                    zeilen[f"{q}|{al}"] = {"n": int(m.sum()), "fa": float(f[m].mean()), "flags": int(f[m].sum())}
        for nm, m in (("<=2560 px", breite <= 2560), (">2560 px", breite > 2560)):
            m = m & behalten & (breite > 0)
            zeilen[f"alle|{nm}"] = {"n": int(m.sum()), "fa": float(f[m].mean()), "flags": int(f[m].sum())}
        erg["koepfe"][tag] = zeilen
        print(f"\n{tag}")
        for kk, v in zeilen.items():
            print(f"  {kk:<34} n={v['n']:>5}  FA {v['fa']:6.2%}  ({v['flags']})")
    json.dump(erg, open(os.path.join(DATEN, "profikunst_datum.json"), "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
