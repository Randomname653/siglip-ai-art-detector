r"""gleich_fa.py — compare heads at an EQUAL false-positive rate.

Each head has its own stored threshold, calibrated on different human data; comparing hit rates at each head's
own threshold therefore also compares calibration, not only separating power. Here every head gets the
threshold at which it flags x % of the three human TEST groups on average (Danbooru anime 2026, Wallhaven
general <2022, Danbooru non-anime 2022-26; each group weighted equally). For comparison only — the threshold is
set on the test data. Also reports per-channel AUC (threshold-free). Channels: every Telegram AI channel with a
feature cache.

    python gleich_fa.py v3 p8x p8sx [--fa 1] [--json out.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from danbooru_test import bewerte, koepfe_von     # noqa: E402
from lesen import DATEN, Pack, kanal_X             # noqa: E402
from p2_test import merkmale                       # noqa: E402

# alle Telegram-KI-Kanaele mit Merkmals-Cache (anfangs 15; neue kommen dazu, sobald eingebettet)
KANAELE = tuple(sorted(f[6:-8] for f in os.listdir(DATEN) if f.startswith("kanal_") and f.endswith("_emb.npz")
                       and os.path.isdir(os.path.join(f"{DATASETS}/telegram_ki", f[6:-8]))))


def schwelle(gruppen, fa):
    """Schwelle t, bei der der Mittelwert der Gruppen-FA = fa (Bisektion)."""
    lo, hi = -40.0, 40.0
    for _ in range(60):
        t = (lo + hi) / 2
        if np.mean([np.mean(s > t) for s in gruppen]) > fa:
            lo = t
        else:
            hi = t
    return hi


def auc(neg, pos):
    alle = np.concatenate([neg, pos])
    r = alle.argsort().argsort().astype(np.float64) + 1
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--fa", type=float, default=1.0)
    ap.add_argument("--json", default="gleich_fa.json")
    a = ap.parse_args()

    pd, pp = Pack("v3d"), Pack("p2z")
    Xd, okd = merkmale("v3d")
    Xp, okp = merkmale("p2z", "test")
    wd = np.load(os.path.join(DATEN, "p2z_wd.npz"))
    okp &= pp.ok & ~wd["gesperrt"]
    mensch = {"Danbooru Anime 2026": np.asarray(Xd[np.flatnonzero(okd & pd.ok & (pd.split == "test"))])}
    for q, n in (("wallhaven_general", "Wallhaven general <2022"), ("danbooru_nichtanime", "Danbooru Nicht-Anime")):
        mensch[n] = np.asarray(Xp[np.flatnonzero(okp & (pp.split == "test") & (pp.quelle == q))])
    ki = {n: kanal_X(n)[0] for n in KANAELE}

    erg = {}
    for tag in a.tags:
        k = koepfe_von(tag)
        sm = {g: bewerte(X, k) for g, X in mensch.items()}
        sk = {n: bewerte(X, k) for n, X in ki.items()}
        t = schwelle(list(sm.values()), a.fa / 100)
        mh = np.concatenate(list(sm.values()))
        erg[tag] = {"schwelle": t,
                    "fa": {g: float(np.mean(s > t)) for g, s in sm.items()},
                    "treffer": {n: float(np.mean(s > t)) for n, s in sk.items()},
                    "auc": {n: float(auc(mh, s)) for n, s in sk.items()}}

    print(f"Gleicher Fehlalarm: im Mittel {a.fa:g} % auf den drei menschlichen Testgruppen\n")
    print(f"{'':<26}" + "".join(f"{t:>9}" for t in a.tags))
    print(f"{'Schwelle':<26}" + "".join(f"{erg[t]['schwelle']:>+9.2f}" for t in a.tags))
    for g in mensch:
        print(f"FA {g:<23}" + "".join(f"{erg[t]['fa'][g]:>9.1%}" for t in a.tags))
    print()
    for n in KANAELE:
        print(f"{n:<26}" + "".join(f"{erg[t]['treffer'][n]:>9.1%}" for t in a.tags)
              + "   AUC " + " ".join(f"{erg[t]['auc'][n]:.4f}" for t in a.tags))
    mt = {t: np.mean(list(erg[t]["treffer"].values())) for t in a.tags}
    print(f"{'Mittel der Kanaele':<26}" + "".join(f"{mt[t]:>9.1%}" for t in a.tags))
    with open(os.path.join(DATEN, a.json), "w", encoding="utf-8") as f:
        json.dump({"fa": a.fa, "ergebnis": erg}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
