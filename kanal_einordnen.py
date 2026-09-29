r"""kanal_einordnen.py — new Telegram export: AI channel (-> lab) or human channel (-> production inbox)?

A random sample of photos (no thumbnails) goes through the scanner head; share above the 1 % and 5 %
thresholds. At ~1 % false positives, "almost all above" means an AI channel, "almost none" a human channel,
anything in between is mixed and decided by hand. No images are displayed.

    python kanal_einordnen.py "<export>" ["<export>" ...] [--n 300] [--kopf p8sx]
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kandidat import logit_stapel                 # noqa: E402
from pruefe import _kopf, stapelrechner, vorbereiten   # noqa: E402
from lesen import DATEN                           # noqa: E402

BILD = (".jpg", ".jpeg", ".png", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exporte", nargs="+")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--kopf", default="p6x")
    a = ap.parse_args()
    koepfe = []
    while os.path.exists(os.path.join(DATEN, f"kopf_{a.kopf}_s{len(koepfe)}.npz")):
        koepfe.append(_kopf(f"{a.kopf}_s{len(koepfe)}"))
    s1, s5 = float(koepfe[0]["schwelle_1"]), float(koepfe[0]["schwelle_5"])
    rg = [str(r) for r in koepfe[0]["rueckgrate"]]
    rechner = stapelrechner(rg)
    rng = np.random.default_rng(5)
    for exp in a.exporte:
        pfade = [os.path.join(p, f) for p, _, fs in os.walk(exp) for f in fs
                 if f.lower().endswith(BILD) and "_thumb" not in f and os.sep + "images" + os.sep not in p + os.sep
                 and os.sep + "css" not in p]
        wahl = [pfade[i] for i in rng.choice(len(pfade), min(a.n, len(pfade)), replace=False)] if pfade else []
        z, foto = [], 0
        with ThreadPoolExecutor(8) as ex:
            for s in range(0, len(wahl), 32):
                st = list(ex.map(vorbereiten, wahl[s:s + 32]))
                ok = [b for _, v, b in st if b is not None]
                foto += sum(1 for _, v, _ in st if v == "foto")
                if ok:
                    X = np.concatenate([rechner[r](ok, ex) for r in rg], 1)
                    z.append(np.mean([logit_stapel(X, k) for k in koepfe], 0))
        z = np.concatenate(z) if z else np.array([])
        print(f"{os.path.basename(exp)}: {len(pfade)} Bilder, Stichprobe {len(wahl)}, bewertet {len(z)}, Fotos {foto} | "
              f"KI bei 1 %: {np.mean(z > s1):.0%}  bei 5 %: {np.mean(z > s5):.0%}  Median-Logit {np.median(z):+.1f}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
