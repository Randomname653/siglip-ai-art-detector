r"""ordner_merkmale.py — SigLIP-2 features for every image of a folder (recursive), for evaluating any head on it.

Images are normalized as in production (norm.normiere: 512 px, JPEG q95 4:4:4; smaller ones are upscaled like
imagesort does). Saves daten/ordner_<name>.npz with X (float16, siglip_mid), pfade, groesse (longer side).
Read-only on the folder; resumable is not needed (one pass).

    python ordner_merkmale.py "<folder>" --name sorted
"""
import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import DATEN                          # noqa: E402
from norm import normiere                        # noqa: E402
from pruefe import stapelrechner                 # noqa: E402

ENDUNGEN = (".jpg", ".jpeg", ".png", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ordner")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    pfade = sorted(os.path.join(dp, f) for dp, dn, fn in os.walk(a.ordner) for f in fn
                   if f.lower().endswith(ENDUNGEN) and "_ai_quarantine" not in dp and "_thumb" not in f)
    print(f"{len(pfade)} Bilder", flush=True)
    rechner = stapelrechner(["siglip_mid"])
    X, ok, gr = [], [], []
    t0 = time.time()

    def laden(p):
        try:
            daten, w0, h0, _, _ = normiere(p)
            return Image.open(io.BytesIO(daten)).convert("RGB"), max(w0, h0)
        except Exception:
            return None, 0
    with ThreadPoolExecutor(8) as ex:
        for s in range(0, len(pfade), 64):
            geladen = list(ex.map(laden, pfade[s:s + 64]))
            bs = [b for b, _ in geladen if b is not None]
            if bs:
                X.append(rechner["siglip_mid"](bs, ex).astype(np.float16))
            ok += [b is not None for b, _ in geladen]
            gr += [g for _, g in geladen]
            if (s // 64) % 50 == 0:
                print(f"  {s + len(geladen)}/{len(pfade)}  {(s + len(geladen)) / (time.time() - t0):.0f}/s", flush=True)
    ok = np.array(ok)
    np.savez(os.path.join(DATEN, f"ordner_{a.name}.npz"), X=np.concatenate(X), pfade=np.array(pfade)[ok],
             groesse=np.array(gr)[ok])
    print(f"FERTIG {int(ok.sum())} Merkmale -> ordner_{a.name}.npz")


if __name__ == "__main__":
    main()
