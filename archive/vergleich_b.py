r"""vergleich_b.py — neue Koepfe gegen die Produktion auf dem Telegram-Test von v2.

Gleiche Bilder fuer alle: vier nie gesehene Menschen-Kanaele (vor 2022) gegen
den nie gesehenen KI-Kanal plus die 300 losen Ai-Only-Dateien. Die Produktion
sah die Rohdateien (telegram4.py), die Koepfe die ungestoerte Zurichtung
(train2.py, Teil B).

Die Trefferquote bei festem Fehlalarm wird ueber die ROC-Kurve bestimmt — die
Schwelle sitzt also fuer jeden Detektor auf den menschlichen Testbildern
selbst. Das ist fuer alle gleich und damit fair; fuer den Einsatz zaehlt beim
neuen Kopf zusaetzlich die Schwelle aus val (steht in ergebnis_v2_*.json).

    python lab/detektor/vergleich_b.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from lesen import DATEN, Pack                    # noqa: E402


def kenn(y, s):
    from sklearn.metrics import roc_auc_score, roc_curve
    f, t, _ = roc_curve(y, s)
    return roc_auc_score(y, s), np.interp(0.01, f, t), np.interp(0.05, f, t)


def main():
    pack = Pack("v2")
    tb = np.load(os.path.join(DATEN, "t4_idx.npy"))
    keep = pack.ok[tb]                           # ohne Telegram-Vorschaubilder
    tb = tb[keep]
    y = pack.label[tb]
    grp = pack.grp[tb]
    kanal = (y == 0) | (grp == "ai:ChatExport_2026-05-31 (1)")
    lose = (y == 0) | (grp == "ai:.")
    zeilen = []
    for p in sorted(glob.glob(os.path.join(DATEN, "score_v2_*_B.npy"))):
        d = np.load(p)
        m = dict(zip(d[:, 0].astype(int), d[:, 1]))
        if not all(i in m for i in tb):
            continue
        s = np.array([m[i] for i in tb])
        zeilen.append(("neu: " + os.path.basename(p)[9:-6], s))
    for p in sorted(glob.glob(os.path.join(DATEN, "t4_*.npy"))):
        if p.endswith("_idx.npy"):
            continue
        zeilen.append(("Prod: " + os.path.basename(p)[3:-4].replace("deepghs_", "")
                       .replace("hf_", ""), np.load(p)[keep]))
    erg = {}
    print(f"{len(tb)} Bilder: {int((y==0).sum())} Mensch (4 Kanaele), {int((y==1).sum())} KI\n")
    print(f"{'':<44}{'gesamt':>22}{'KI-Kanal (1)':>22}{'300 lose':>12}")
    print(f"{'':<44}{'AUC   @1%   @5%':>22}{'AUC   @1%   @5%':>22}{'AUC':>12}")
    for name, s in sorted(zeilen, key=lambda z: -kenn(y, z[1])[0]):
        g, k, l = kenn(y, s), kenn(y[kanal], s[kanal]), kenn(y[lose], s[lose])
        erg[name] = {"gesamt": g, "kanal": k, "lose": l}
        print(f"{name[:43]:<44}{g[0]:>7.3f}{g[1]:>7.1%}{g[2]:>7.1%}"
              f"{k[0]:>8.3f}{k[1]:>7.1%}{k[2]:>7.1%}{l[0]:>12.3f}")
    with open(os.path.join(DATEN, "vergleich_b.json"), "w", encoding="utf-8") as f:
        json.dump(erg, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
