r"""mj_v6.py — Midjourney v6 sample from a ready-made scrape dataset (many creators).

Source: huggingface.co/datasets/brivangl/midjourney-v6-llava (dataset license MIT; images originally collected
from the Midjourney Discord by its authors). Counterpart to the single-creator Felix archive: thousands of users,
so the head learns "Midjourney", not the style of one person. 8 of 124 parquet files, 2,500 random images each;
same filters as midjourney.py.

    python mj_v6.py [--dateien 8] [--je 2500]
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from midjourney import HF, Pruefer, holen      # noqa: E402

REPO = "brivangl/midjourney-v6-llava"
ZIEL = f"{DATASETS}/midjourney_v6_scrape"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dateien", type=int, default=8)
    ap.add_argument("--je", type=int, default=2500)
    a = ap.parse_args()
    os.makedirs(ZIEL, exist_ok=True)
    nummern = np.linspace(0, 123, a.dateien).round().astype(int)
    p = Pruefer(ZIEL)
    t0 = time.time()
    for n in nummern:
        name = f"train_{n:03d}.parquet"
        fertig = os.path.join(ZIEL, "." + name + ".fertig")
        if os.path.exists(fertig):
            continue
        lokal = os.path.join(ZIEL, name)
        if not holen(HF + REPO + "/resolve/main/data/" + name, lokal):
            print(f"  {name} uebersprungen", flush=True)
            continue
        tab = pq.read_table(lokal, columns=["id", "prompt", "image"])
        rng = np.random.default_rng(int(n))
        for i in sorted(rng.choice(tab.num_rows, min(a.je, tab.num_rows), replace=False)):
            zeile = {k: tab.column(k)[int(i)].as_py() for k in ("id", "prompt", "image")}
            daten = zeile["image"]["bytes"] if isinstance(zeile["image"], dict) else zeile["image"]
            e = {"id": zeile["id"], "quelle": "midjourney_v6_scrape", "lizenz": "MIT (Datensatz-Angabe); Discord-Scrape",
                 "prompt": zeile["prompt"] or "", "version": "6", "datei_hf": name, "url": HF + REPO}
            p.bild(daten, e, name[:-8])
        del tab
        os.remove(lokal)
        open(fertig, "w").close()
        print(f"{name}: bisher {dict(p.stand)}  {(time.time()-t0)/60:.0f} min", flush=True)
    print(f"FERTIG: {dict(p.stand)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
