r"""kontrolle3.py — leak controls for a head ensemble (batched).

  Scale intervention  human test images are upscaled by the factor typical of the AI buckets, AI images are
                      downscaled accordingly. If the head relied on scale/resampling traces, humans would move
                      up and AI down; they must stay put.
  Robustness          both classes perturbed IDENTICALLY (JPEG, downscaling). If AUC drops sharply, the
                      performance rested on processing traces.
Humans come from the v2 test set and Danbooru 2026. The backbones are taken from the head itself.

    python kontrolle3.py p8sx
"""
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from danbooru_test import koepfe_von, bewerte   # noqa: E402
from lesen import Pack                            # noqa: E402
from train import auc                             # noqa: E402
from train2 import tpr_bei                        # noqa: E402

RG = ("clip_mid", "dino_mid")


def main():
    tag = sys.argv[1]
    koepfe = koepfe_von(tag)
    global RG
    RG = tuple(str(r) for r in koepfe[0]["rueckgrate"])    # die Rueckgrate, mit denen der Kopf trainiert wurde
    from norm import QUALITAET
    from pruefe import stapelrechner
    rechner = stapelrechner(list(RG))
    p2, pd = Pack("v2"), Pack("v3d")
    rng = np.random.default_rng(2)

    def einbetten(bilder_fn, n):
        out = []
        with ThreadPoolExecutor(8) as ex:
            for a in range(0, n, 32):
                bs = list(ex.map(bilder_fn, range(a, min(n, a + 32))))
                out.append(np.concatenate([rechner[r](bs, ex) for r in RG], 1))
        return np.concatenate(out)

    def jpeg(b, q, sub):
        buf = io.BytesIO()
        b.save(buf, "JPEG", quality=q, subsampling=sub)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    # --- Eingriff
    te = p2.wo(split="test")
    kurz = np.minimum(p2.size[te, 2], p2.size[te, 3])
    kp = 293 if (kurz[p2.quelle[te] == "generiert"] == 293).any() else 292
    neg = te[(p2.model[te] == "mensch") & (kurz == 288)]
    pos = te[(p2.quelle[te] == "generiert") & (kurz == kp)]
    neg = rng.choice(neg, min(1500, len(neg)), replace=False)
    pos = rng.choice(pos, min(1500, len(pos)), replace=False)
    print(f"Eingriff: {len(neg)} Menschen x {kp}/288, {len(pos)} KI x 288/{kp}", flush=True)

    def skaliert(idx, f):
        def fn(j):
            b = p2.bild(idx[j])
            w, h = b.size
            return jpeg(b.resize((round(w * f), round(h * f)), Image.LANCZOS), QUALITAET, 0)
        return fn
    for lab, idx, f in (("Menschen", neg, kp / 288), ("KI", pos, 288 / kp)):
        vor = bewerte(einbetten(skaliert(idx, 1.0), len(idx)), koepfe)
        nach = bewerte(einbetten(skaliert(idx, f), len(idx)), koepfe)
        dz = nach - vor
        print(f"  {lab:<9} Logit vorher {np.median(vor):+.2f}  nachher {np.median(nach):+.2f}  "
              f"Verschiebung Median {np.median(dz):+.3f}", flush=True)

    # --- Robustheit: Menschen aus v2-Telegram-Test + Danbooru 2026, KI aus v2-Test
    tm = p2.wo(split="test", label=0)
    tm = tm[p2.quelle[tm] == "telegram"]
    dm = np.flatnonzero(pd.ok & (pd.split == "test"))
    tk = p2.wo(split="test", label=1)
    n = 1000
    quellen = [(p2, rng.choice(tm, n, replace=False)), (pd, rng.choice(dm, n, replace=False)),
               (p2, rng.choice(tk, 2 * n, replace=False))]
    y = np.r_[np.zeros(2 * n), np.ones(2 * n)]
    print(f"\nRobustheit: {2*n} Menschen (Telegram + Danbooru 2026) gegen {2*n} KI, beide gleich gestoert")
    for lab, f, q in (("unveraendert", 1.0, 95), ("nur JPEG q90", 1.0, 90), ("x0,9 + q90", 0.9, 90),
                      ("x0,75 + q85", 0.75, 85), ("x0,5 + q80", 0.5, 80)):
        teile = []
        for p, idx in quellen:
            def fn(j, p=p, idx=idx):
                b = p.bild(idx[j])
                if f != 1.0:
                    w, h = b.size
                    b = b.resize((round(w * f), round(h * f)), Image.LANCZOS)
                return jpeg(b, q, 0 if q >= 95 else 2)
            teile.append(bewerte(einbetten(fn, len(idx)), koepfe))
        s = np.concatenate(teile)
        print(f"  {lab:<14} AUC {auc(s, y):.4f}   Treffer bei 1 % FA {tpr_bei(s[y == 0], s[y == 1]):.1%}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
