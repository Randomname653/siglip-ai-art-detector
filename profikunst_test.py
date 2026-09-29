r"""profikunst_test.py — false positives on professional concept and game art (nas_profikunst.py test group).

Human-made art by professional digital artists (CGWallpapers, with the software named in the footer) and official
game artwork (GameWallpapers) — polished work of exactly the kind that is hardest for a detector. Footers were
cropped before storing. Reported per source and for the ten artists/games with the most flags: false positives
of our heads at their scanner threshold, and of the seven production models at the threshold that gives 1 % on
the standard human validation groups (from bench.py). Also a contact sheet of flagged general-rated images.

    python profikunst_test.py [--koepfe p8sx,p8x]
"""
import argparse
import collections
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                        # noqa: E402
import messlatte                                  # noqa: E402
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from lesen import DATEN                          # noqa: E402
from norm import normiere                        # noqa: E402
from pruefe import stapelrechner                 # noqa: E402

MF = f"{DATASETS}/nas_profikunst/manifest.jsonl"
CACHE = os.path.join(DATEN, "profikunst_merkmale.npz")


class Dateien:
    """messlatte-kompatibel: bild(i) liefert das normierte Bild."""
    def __init__(self, pfade):
        self.p = pfade

    def bild(self, i, aug=False):
        return Image.open(io.BytesIO(normiere(self.p[i])[0])).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--koepfe", default="p8sx,p8x")
    a = ap.parse_args()
    E = [json.loads(z) for z in open(MF, encoding="utf-8")]
    E = [e for e in E if e.get("datei") and not e.get("grund")]
    pfade = [e["datei"] for e in E]
    print(f"{len(E)} Bilder: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(e["quelle"] for e in E).items()),
          flush=True)
    rg_alle = ("clip_mid", "dino_mid", "siglip_mid")
    if os.path.exists(CACHE) and len(np.load(CACHE)["pfade"]) == len(pfade):
        X = np.load(CACHE)["X"]
    else:
        rechner = stapelrechner(list(rg_alle))
        X = []
        with ThreadPoolExecutor(8) as ex:
            for s in range(0, len(pfade), 32):
                bs = list(ex.map(lambda p: Image.open(io.BytesIO(normiere(p)[0])).convert("RGB"), pfade[s:s + 32]))
                X.append(np.concatenate([rechner[r](bs, ex) for r in rg_alle], 1).astype(np.float16))
        X = np.concatenate(X)
        np.savez(CACHE, X=X, pfade=np.array(pfade))
    quelle = np.array([e["quelle"] for e in E])
    gruppe = np.array([e["gruppe"] for e in E])
    erg = {"n": {q: int((quelle == q).sum()) for q in set(quelle)}, "koepfe": {}, "produktion": {}}
    geflaggt = {}
    for tag in a.koepfe.split(","):
        k = koepfe_von(tag)
        z = bewerte(X, k)
        f = z > float(k[0]["schwelle_1"])
        geflaggt[tag] = f
        erg["koepfe"][tag] = {q: float(f[quelle == q].mean()) for q in set(quelle)}
        erg["koepfe"][tag]["alle"] = float(f.mean())
    # Produktion: Schwellen aus bench.json (1 % auf den val-Menschen) -> hier nur die Scores
    bench = json.load(open(os.path.join(DATEN, "bench_p8s.json"), encoding="utf-8"))["ergebnis"]
    q = Dateien(pfade)
    for key, backend, name in messlatte.MODELLE:
        ziel = os.path.join(DATEN, f"profikunst_{key.replace(':', '_')}.npy")
        if os.path.exists(ziel) and len(np.load(ziel)) == len(pfade):
            s = np.load(ziel)
        else:
            fn = messlatte.score_hf if backend == "hf" else messlatte.score_deepghs
            s = np.asarray(fn(q, np.arange(len(pfade)), name), np.float32)
            np.save(ziel, s)
        s = np.log(np.clip(s, 1e-7, 1 - 1e-7) / np.clip(1 - s, 1e-7, 1))
        t = bench[key]["schwelle"]
        f = s > t
        erg["produktion"][key] = {qq: float(f[quelle == qq].mean()) for qq in set(quelle)}
        erg["produktion"][key]["alle"] = float(f.mean())
    haupt = a.koepfe.split(",")[0]
    top = collections.Counter(gruppe[geflaggt[haupt]]).most_common(10)
    erg["meiste_flags"] = [(g, int(n), int((gruppe == g).sum())) for g, n in top]
    print(f"\n{'Detektor':<36}" + "".join(f"{qq:>16}" for qq in sorted(set(quelle))) + f"{'alle':>10}")
    for tag, v in list(erg["koepfe"].items()) + list(erg["produktion"].items()):
        print(f"{tag:<36}" + "".join(f"{v[qq]:>15.2%} " for qq in sorted(set(quelle))) + f"{v['alle']:>9.2%}")
    print("\nmeiste Flags (" + haupt + "):", erg["meiste_flags"])
    json.dump(erg, open(os.path.join(DATEN, "profikunst_test.json"), "w", encoding="utf-8"), indent=1)
    # Kontaktbogen: geflaggte, jugendfreie Bilder
    zeigen = [i for i in np.flatnonzero(geflaggt[haupt]) if E[i].get("rating") == "general"][:30]
    if zeigen:
        W = 240
        b = Image.new("RGB", (6 * W, ((len(zeigen) + 5) // 6) * (W + 16)), "white")
        d = ImageDraw.Draw(b)
        for n, i in enumerate(zeigen):
            im = Image.open(pfade[i]).convert("RGB")
            im.thumbnail((W - 4, W - 4))
            x, y = (n % 6) * W, (n // 6) * (W + 16)
            b.paste(im, (x + 2, y + 2))
            d.text((x + 3, y + W), f"{E[i]['gruppe'][:28]}", fill="black")
        b.save(os.path.join(DATEN, "profikunst_flags.jpg"), quality=85)
    return 0


if __name__ == "__main__":
    sys.exit(main())
