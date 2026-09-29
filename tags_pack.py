r"""tags_pack.py — WD content rating and child-protection flag for every image of a pack.

  1. Applies schutz.py to the older sources too (Telegram channels, own generators, Wallhaven), which never went
     through the check at download time. Blocked images are excluded from training.
  2. Rating per image for rating balancing in training (--rating-ausgleich): one AI source is 90 % explicit, the
     human side hardly — otherwise a head could learn "explicit = AI".

Batched through the imgutils ONNX session (same preprocessing as get_wd14_tags). Output daten/<pack>_wd.npz with
rating (0 general, 1 sensitive, 2 questionable, 3 explicit, -1 not rated) and gesperrt.

    python tags_pack.py <pack>
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import DATEN, Pack                    # noqa: E402
from schutz import SCHWELLE, SPERRE              # noqa: E402


def main():
    from imgutils.tagging import wd14
    stamm = sys.argv[1]
    pack = Pack(stamm)
    sess = wd14._get_wd14_model("EVA02_Large")
    namen, rat_idx, gen_idx, _ = wd14._get_wd14_labels("EVA02_Large")
    sperr_idx = np.array([i for i, n in enumerate(namen) if n in SPERRE])
    print(f"{stamm}: {int(pack.ok.sum())} Bilder, {len(sperr_idx)} Sperr-Tags im Vokabular", flush=True)

    n = len(pack)
    rating = np.full(n, -1, np.int8)
    gesperrt = np.zeros(n, bool)
    idx = np.flatnonzero(pack.off >= 0)          # auch Vorschaubilder: die Sperre gilt fuer alles
    vor = lambda i: wd14._prepare_image_for_tagging(pack.bild(i), 448)
    t0 = time.time()
    with ThreadPoolExecutor(12) as ex:
        for s in range(0, len(idx), 8):
            b = idx[s:s + 8]
            x = np.concatenate(list(ex.map(vor, b)))
            p = sess.run(["output"], {"input": x})[0]
            rating[b] = np.argmax(p[:, rat_idx], 1)
            gesperrt[b] = (p[:, sperr_idx] >= SCHWELLE).any(1)
            if (s // 8) % 1000 == 0 and s:
                el = time.time() - t0
                print(f"  {s}/{len(idx)}  {s/el:.0f}/s  Rest {(len(idx)-s)/(s/el)/60:.0f} min", flush=True)
    np.savez(os.path.join(DATEN, f"{stamm}_wd.npz"), rating=rating, gesperrt=gesperrt)
    for q in np.unique(pack.quelle):
        m = (pack.quelle == q) & (rating >= 0)
        verteilung = " ".join(f"{k} {np.mean(rating[m] == j):.1%}" for j, k in
                              enumerate(("general", "sensitive", "questionable", "explicit")))
        print(f"  {q:<10} n={m.sum():>6}  gesperrt {int(gesperrt[m].sum()):>5}  {verteilung}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
