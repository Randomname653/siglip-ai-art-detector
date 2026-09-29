r"""ordner_bewerten.py — compare heads on a real folder (features from ordner_merkmale.py).

Human = Telegram post date before 2022-08-01 (era alibi). Reports per head: false positives at its own 1 %
threshold (posts and distinct images — duplicates grouped by 32x32 pixel thumbnails), flag rate on posts from
2025/26 (mixed, rough proxy for hits), and both at an EQUAL false-positive rate on the human posts.

    python ordner_bewerten.py --name sorted --koepfe p8sx,p10ax
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from fp_kategorien import daumen, gruppiere      # noqa: E402
from lesen import DATEN                          # noqa: E402
import imagesort as I                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sorted")
    ap.add_argument("--koepfe", default="p8sx")
    a = ap.parse_args()
    z = np.load(os.path.join(DATEN, f"ordner_{a.name}.npz"))
    X, pfade = z["X"], [str(p) for p in z["pfade"]]
    datum = [I.telegram_post_date(p) for p in pfade]
    mensch = np.array([d is not None and d < I.AI_ERA_CUTOFF for d in datum])
    neu = np.array([d is not None and d >= (2025, 1, 1) for d in datum])
    dubl = os.path.join(DATEN, f"ordner_{a.name}_dubletten.json")
    if os.path.exists(dubl):
        gruppe = json.load(open(dubl, encoding="utf-8"))
    else:
        hp = [p for p, m in zip(pfade, mensch) if m]
        with ThreadPoolExecutor(12) as ex:
            gruppe = gruppiere(hp, list(ex.map(daumen, hp)))
        json.dump(gruppe, open(dubl, "w", encoding="utf-8"))
    hp = [p for p, m in zip(pfade, mensch) if m]
    n_unik = len({gruppe[p] for p in hp})
    print(f"{int(mensch.sum())} menschliche Posts ({n_unik} verschiedene Bilder), {int(neu.sum())} Posts 2025/26")
    erg = {}
    print(f"\n{'Kopf':<10}{'FA Posts':>10}{'FA Bilder':>11}{'2025/26 markiert':>18} | bei gleichem FA:"
          + "".join(f"{f'{z_:.2%}':>9}" for z_ in (0.005, 0.0025, 0.001)))
    for tag in a.koepfe.split(","):
        k = koepfe_von(tag)
        s = bewerte(X, k)
        s1 = float(k[0]["schwelle_1"])
        fl = s > s1
        fa_bilder = len({gruppe[p] for p, f in zip(np.array(pfade)[mensch], fl[mensch]) if f}) / n_unik
        zeile = {"fa_posts": float(fl[mensch].mean()), "fa_bilder": fa_bilder, "neu_markiert": float(fl[neu].mean())}
        sh = np.sort(s[mensch])[::-1]
        for ziel in (0.005, 0.0025, 0.001):
            t = sh[max(0, int(round(ziel * len(sh))) - 1)]
            zeile[f"neu_bei_{ziel}"] = float(np.mean(s[neu] > t))
        erg[tag] = zeile
        print(f"{tag:<10}{zeile['fa_posts']:10.2%}{fa_bilder:11.2%}{zeile['neu_markiert']:18.1%} |                "
              + "".join(f"{zeile[f'neu_bei_{z_}']:9.1%}" for z_ in (0.005, 0.0025, 0.001)))
    json.dump(erg, open(os.path.join(DATEN, f"ordner_{a.name}_bewertung.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
