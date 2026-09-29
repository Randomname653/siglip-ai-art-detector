r"""rating_mix.py — wie verteilt sich das WD-Rating je Quelle im Training?

Ist die menschliche Seite fast nur jugendfrei und die KI-Seite sehr explizit,
kann der Kopf "explizit = KI" lernen, und die Testkanaele (fast nur SFW)
wuerden das nicht zeigen. Stichprobe je Quelle, WD-EVA02-Rating.

    python lab/detektor/rating_mix.py
"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from lesen import Pack                           # noqa: E402

N = 2500


def main():
    from imgutils.tagging import get_wd14_tags
    rng = np.random.default_rng(3)
    for stamm, quellen in (("v2", ("wallhaven", "telegram", "generiert", "ai_only")),
                           ("v3z", ("civitai", "aibooru"))):
        p = Pack(stamm)
        for q in quellen:
            i = np.flatnonzero(p.ok & (p.split == "train") & (p.quelle == q))
            wahl = rng.choice(i, min(N, len(i)), replace=False)
            z = collections.Counter()
            for j in wahl:
                r, _, _ = get_wd14_tags(p.bild(j), model_name="EVA02_Large")
                z[max(r, key=r.get)] += 1
            n = sum(z.values())
            print(f"{q:<10} n={n:>5}  " + "  ".join(f"{k} {z[k]/n:5.1%}" for k in
                                                   ("general", "sensitive", "questionable", "explicit")),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
