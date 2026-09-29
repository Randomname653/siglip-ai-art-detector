r"""kanal_siglip.py — SigLIP 2 features for all Telegram channel images, in the same order as the existing
CLIP+DINO channel caches (daten/kanal_<name>_emb.npz). Output daten/kanal_<name>_siglip.npz.

    python kanal_siglip.py
"""
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embed                                      # noqa: E402
from lesen import DATEN                           # noqa: E402
from norm import normiere                         # noqa: E402


def main():
    model, proc = embed.siglip_modell()
    ex = ThreadPoolExecutor(12)
    for f in sorted(os.listdir(DATEN)):
        if not (f.startswith("kanal_") and f.endswith("_emb.npz")):
            continue
        ziel = os.path.join(DATEN, f.replace("_emb.npz", "_siglip.npz"))
        if os.path.exists(ziel):
            continue
        pfade = [str(p) for p in np.load(os.path.join(DATEN, f))["pfade"]]
        X = np.zeros((len(pfade), 1152 * len(embed.SIGLIP_MID_SCHICHTEN)), np.float16)
        t0 = time.time()
        for s in range(0, len(pfade), 64):
            bilder = list(ex.map(lambda p: Image.open(io.BytesIO(normiere(p)[0])).convert("RGB"), pfade[s:s + 64]))
            X[s:s + len(bilder)] = embed.siglip_mid_merkmale(model, proc, bilder, ex)
        np.savez(ziel, X=X, pfade=np.array(pfade))
        print(f"{f[6:-8]}: {len(pfade)} in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
