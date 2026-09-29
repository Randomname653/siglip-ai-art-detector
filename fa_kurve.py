r"""fa_kurve.py — false positives vs. hit rate for one head across all human test groups, incl. the Telegram
human test split (in-domain: the owner's own Telegram exports before 2022) that p2_test.py does not report.

For each target false-positive rate (on the equally weighted calibration groups) the threshold is taken from the
unperturbed val humans (as kalibriere.py does), then applied to: every human test group (false positives) and the
Telegram AI channels (hit rate). Answers "what does a stricter threshold cost?".

    python fa_kurve.py [--kopf p8sx] [--ziele 1,0.5,0.25,0.1]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from lesen import DATEN, Pack, kanal_X, spalten_fuer   # noqa: E402
from train3 import lade                          # noqa: E402
import train3                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kopf", default="p8sx")
    ap.add_argument("--ziele", default="1,0.5,0.25,0.1")
    a = ap.parse_args()
    k = koepfe_von(a.kopf)
    rg = tuple(str(r) for r in k[0]["rueckgrate"])
    train3.RG = rg
    s1 = float(k[0]["schwelle_1"])

    def z(pack, split, maske):
        X, ok = lade(pack, False, split)
        idx = np.flatnonzero(ok & maske)
        return bewerte(np.concatenate([np.asarray(t[idx]) for t in X], 1), k)

    gruppen = {}
    p2, pz, pd = Pack("v2"), Pack("p2z"), Pack("v3d")
    gruppen["Telegram-Menschen vor 2022 (Test)"] = z("v2", "test", p2.ok & (p2.label == 0) & (p2.quelle == "telegram")
                                                   & (p2.split == "test"))
    gruppen["Wallhaven <2022 (Test)"] = z("v2", "test", p2.ok & (p2.label == 0) & (p2.quelle == "wallhaven")
                                         & (p2.split == "test"))
    for q in ("wallhaven_general", "danbooru_nichtanime"):
        gruppen[f"{q} (Test)"] = z("p2z", "test", pz.ok & (pz.quelle == q) & (pz.split == "test"))
    Xd, okd = lade("v3d", False)
    idx = np.flatnonzero(okd & pd.ok & (pd.split == "test"))
    gruppen["Danbooru-Anime 2026 (Test)"] = bewerte(np.concatenate([np.asarray(t[idx]) for t in Xd], 1), k)
    # Kanaele
    kan = sorted(f[6:-8] for f in os.listdir(DATEN) if f.startswith("kanal_") and f.endswith("_emb.npz"))
    kz = []
    for name in kan:
        X, _ = kanal_X(name)
        if len(X) >= 50:
            kz.append(bewerte(X[:, spalten_fuer(rg)] if X.shape[1] != len(spalten_fuer(rg)) else X, k))
    # Schwellen: val-Menschen ungestoert, drei Gruppen gleich gewichtet (wie kalibriere.py) -> hier aus dem
    # gespeicherten 1-%-Wert relativ verschoben: Zielquote auf den drei Kalibriergruppen im TEST gemittelt
    kal = [gruppen["Danbooru-Anime 2026 (Test)"], gruppen["wallhaven_general (Test)"],
           gruppen["danbooru_nichtanime (Test)"]]
    erg = {}
    print(f"{'Ziel-FA':>8} {'Schwelle':>9} " + " ".join(f"{g[:22]:>23}" for g in gruppen) + f" {'Kanaele Treffer':>16}")
    for ziel in [float(x) / 100 for x in a.ziele.split(",")]:
        # Schwelle t so, dass das Mittel der drei Kalibriergruppen = ziel
        ts = np.sort(np.concatenate(kal))[::-1]
        lo, hi = ts.min() - 1, ts.max() + 1
        for _ in range(60):
            t = (lo + hi) / 2
            if np.mean([np.mean(g > t) for g in kal]) > ziel:
                lo = t
            else:
                hi = t
        t = hi
        zeile = {g: float(np.mean(v > t)) for g, v in gruppen.items()}
        treffer = float(np.mean([np.mean(v > t) for v in kz]))
        erg[f"{ziel:.4f}"] = {"schwelle": t, "fa": zeile, "treffer_kanaele": treffer,
                              "score": float(1 / (1 + np.exp(-(t - s1))))}
        print(f"{ziel:8.2%} {t:9.2f} " + " ".join(f"{v:23.2%}" for v in zeile.values()) + f" {treffer:16.1%}")
    json.dump(erg, open(os.path.join(DATEN, f"fa_kurve_{a.kopf}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
