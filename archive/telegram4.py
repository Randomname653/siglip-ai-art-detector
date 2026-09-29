r"""telegram4.py — die Produktion auf dem Telegram-Test von v2.

Dieselben Bilder wie Teil B von train2.py: vier nie gesehene Menschen-Kanaele
(Posts von vor 2022) gegen den nie gesehenen KI-Kanal aus Ai Only. Die
Produktionsdetektoren bekommen die ROHDATEI, so wie im Betrieb; der neue Kopf
sieht in train2.py die ungestoerte Zurichtung derselben Datei. Beide Seiten
sind echte Telegram-Posts — keine Simulation noetig.

Das Ziel des Users ist, dass der neue Detektor die Produktion hier ALLEIN
schlaegt. Diese Datei liefert die Latte.

    python lab/detektor/telegram4.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from lesen import DATEN, Pack                    # noqa: E402


class Roh:
    def __init__(self, pfade):
        self.p = pfade

    def bild(self, i, aug=False):
        from PIL import Image
        with Image.open(self.p[i]) as im:
            return im.convert("RGB")


def main():
    import messlatte
    pack = Pack("v2")
    tb = np.flatnonzero(pack.ok & (pack.split == "test")
                        & np.isin(pack.quelle, ["telegram", "ai_only"]))
    pfade = [pack.m[i]["src"] for i in tb]
    print(f"{len(tb)} Telegram-Testbilder "
          f"({int((pack.label[tb]==0).sum())} Mensch, {int((pack.label[tb]==1).sum())} KI)", flush=True)
    np.save(os.path.join(DATEN, "t4_idx.npy"), tb)
    roh, idx = Roh(pfade), np.arange(len(pfade))
    t0 = time.time()
    for key, backend, name in messlatte.MODELLE:
        ziel = os.path.join(DATEN, f"t4_{key.replace(':', '_')}.npy")
        if os.path.exists(ziel):
            continue
        f = messlatte.score_hf if backend == "hf" else messlatte.score_deepghs
        np.save(ziel, f(roh, idx, name))
        print(f"  {key}: {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
