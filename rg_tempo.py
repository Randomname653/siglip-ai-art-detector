r"""rg_tempo.py — scanner cost per backbone: images/s for feature extraction on the GPU (1,024 normalized images,
batches of 32, same path as pruefe.py). Output daten/rg_tempo.json.

    python rg_tempo.py
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import DATEN, RG_ALLE, Pack            # noqa: E402
from pruefe import stapelrechner                  # noqa: E402


def main():
    p = Pack("v3d")
    idx = p.wo(split="test")[:1024]
    bilder = [p.bild(i) for i in idx]
    erg = {}
    with ThreadPoolExecutor(8) as ex:
        for r in RG_ALLE:
            f = stapelrechner([r])[r]
            f(bilder[:32], ex)                         # Aufwaermen (CUDA, cuDNN)
            t0 = time.time()
            for s in range(0, len(bilder), 32):
                f(bilder[s:s + 32], ex)
            erg[r] = len(bilder) / (time.time() - t0)
            print(f"{r:<12} {erg[r]:6.0f} Bilder/s", flush=True)
    json.dump(erg, open(os.path.join(DATEN, "rg_tempo.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
