r"""geschichtet.py — die Tests ohne Inhalts-Verzerrung: nur jugendfrei gegen jugendfrei.

KI-Kanal 1 ist zu 90 % explizit, die menschlichen Testkanaele fast gar nicht
(rating_mix.py, 2026-09-25). Ein Detektor, der "explizit = KI" gelernt hat,
saehe dort gut aus, ohne Herkunft zu erkennen. Hier werden beide Seiten auf
WD-Rating general + sensitive beschraenkt und neu gemessen.

    python lab/detektor/geschichtet.py v3a v3a_ohne_web
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from lesen import DATEN, Pack                    # noqa: E402
from train import auc                            # noqa: E402
from train2 import tpr_bei                       # noqa: E402

SFW = ("general", "sensitive")


def ratings(pack, idx, cache):
    p = os.path.join(DATEN, cache)
    alt = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    fehlt = [int(i) for i in idx if str(int(i)) not in alt]
    if fehlt:
        from imgutils.tagging import get_wd14_tags
        for i in fehlt:
            r, _, _ = get_wd14_tags(pack.bild(i), model_name="EVA02_Large")
            alt[str(i)] = max(r, key=r.get)
        json.dump(alt, open(p, "w", encoding="utf-8"))
    return np.array([alt[str(int(i))] for i in idx])


def main():
    koepfe = sys.argv[1:]
    pack = Pack("v2")
    mensch_cache = json.load(open(os.path.join(DATEN, "inhalt_mensch_test.json"), encoding="utf-8"))
    for k in koepfe:
        d = np.load(os.path.join(DATEN, f"score_{k}_B.npy"))
        idx, s = d[:, 0].astype(int), d[:, 1]
        y = pack.label[idx]
        mensch = y == 0
        r_m = np.array([mensch_cache[str(i)]["rating"] for i in idx[mensch]])
        k1 = (y == 1) & (pack.grp[idx] == "ai:ChatExport_2026-05-31 (1)")
        r_k1 = ratings(pack, idx[k1], "rating_kanal1_test.json")
        # Kanal 2: Reihenfolge wie kanal2_emb.npz / score_{k}_K2.npy
        ids2 = np.load(os.path.join(DATEN, "kanal2_emb.npz"))["ids"]
        Z = {z["id"]: z for z in json.load(open(os.path.join(DATEN, "kanal2.json"), encoding="utf-8"))}
        s2 = np.load(os.path.join(DATEN, f"score_{k}_K2.npy"))
        r_k2 = _ratings_pfade([Z[i]["foto"] for i in ids2], "rating_kanal2.json")
        print(f"\n{k}")
        for name, sp, rp in (("KI-Kanal 1", s[k1], r_k1), ("KI-Kanal 2", s2, r_k2)):
            for lab, mm, mp in (("alle", np.ones(len(r_m), bool), np.ones(len(rp), bool)),
                                ("nur g+s", np.isin(r_m, SFW), np.isin(rp, SFW)),
                                ("nur q+e", ~np.isin(r_m, SFW), ~np.isin(rp, SFW))):
                neg, pos = s[mensch][mm], sp[mp]
                if len(pos) < 30 or len(neg) < 30:
                    print(f"  {name:<11} {lab:<8} zu wenig (Mensch {len(neg)}, KI {len(pos)})")
                    continue
                yy = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
                print(f"  {name:<11} {lab:<8} Mensch {len(neg):>6}  KI {len(pos):>5}  "
                      f"AUC {auc(np.r_[neg, pos], yy):.4f}  @1% {tpr_bei(neg, pos):.1%}")
    return 0


def _ratings_pfade(pfade, cache):
    p = os.path.join(DATEN, cache)
    alt = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    fehlt = [q for q in pfade if q not in alt]
    if fehlt:
        import io
        from PIL import Image
        from imgutils.tagging import get_wd14_tags
        from norm import normiere
        for q in fehlt:
            b = Image.open(io.BytesIO(normiere(q)[0])).convert("RGB")
            r, _, _ = get_wd14_tags(b, model_name="EVA02_Large")
            alt[q] = max(r, key=r.get)
        json.dump(alt, open(p, "w", encoding="utf-8"))
    return np.array([alt[q] for q in pfade])


if __name__ == "__main__":
    sys.exit(main())
