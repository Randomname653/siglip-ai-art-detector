r"""kontrolle_inhalt.py — hat der Kopf Inhalt statt Herkunft gelernt?

Die Web-Quellen (vor allem CivitAI) haben einen anderen Inhaltsmix als die
menschliche Seite: viel NSFW, eigene Stile. Lernt der Kopf "explizit = KI"
oder "Stil X = KI", steigt der Fehlalarm genau bei diesen menschlichen Bildern.

Gemessen auf den menschlichen Telegram-Testbildern (4 Kanaele): Fehlalarm je
WD-Rating und je Stil-Tag bei einer gemeinsamen Schwelle (1 % Fehlalarm ueber
alle Menschen), fuer zwei Koepfe im Vergleich.

    python lab/detektor/kontrolle_inhalt.py v3a v3a_ohne_web
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from lesen import DATEN, Pack                    # noqa: E402

STILE = ("monochrome", "greyscale", "comic", "sketch", "furry", "chibi", "3d", "realistic",
         "english_text", "traditional_media", "pixel_art", "1boy", "multiple_boys", "nude", "sex")


def main():
    koepfe = sys.argv[1:]
    pack = Pack("v2")
    s = {k: np.load(os.path.join(DATEN, f"score_{k}_B.npy")) for k in koepfe}
    idx = s[koepfe[0]][:, 0].astype(int)
    mensch = pack.label[idx] == 0
    hi = idx[mensch]

    cache = os.path.join(DATEN, "inhalt_mensch_test.json")
    if os.path.exists(cache):
        tags = {int(k): v for k, v in json.load(open(cache, encoding="utf-8")).items()}
    else:
        from imgutils.tagging import get_wd14_tags
        tags = {}
        for j, i in enumerate(hi):
            rating, allg, _ = get_wd14_tags(pack.bild(i), model_name="EVA02_Large")
            tags[int(i)] = {"rating": max(rating, key=rating.get),
                            "stil": [t for t in STILE if t in allg]}
            if j % 2000 == 0:
                print(f"  {j}/{len(hi)}", flush=True)
        json.dump(tags, open(cache, "w", encoding="utf-8"))

    gruppen = collections.defaultdict(list)
    for j, i in enumerate(hi):
        gruppen["rating:" + tags[int(i)]["rating"]].append(j)
        for t in tags[int(i)]["stil"]:
            gruppen[t].append(j)
    print(f"\nFehlalarm je Gruppe (Schwelle: 1 % ueber alle {len(hi)} Menschen)")
    print(f"{'Gruppe':<22}{'n':>7}" + "".join(f"{k:>16}" for k in koepfe))
    for g in sorted(gruppen, key=lambda g: (not g.startswith("rating"), g)):
        j = np.array(gruppen[g])
        if len(j) < 50:
            continue
        z = f"{g:<22}{len(j):>7}"
        for k in koepfe:
            sm = s[k][mensch, 1]
            thr = np.quantile(sm, 0.99)
            z += f"{(sm[j] > thr).mean():>15.2%} "
        print(z)
    return 0


if __name__ == "__main__":
    sys.exit(main())
