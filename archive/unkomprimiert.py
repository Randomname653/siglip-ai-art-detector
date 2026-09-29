r"""unkomprimiert.py — erkennt der Kopf auch Bilder, die nie hochgeladen wurden?

Trainiert wird nur auf Bildern mit (simulierter) Telegram-Kompression. Frage
des Users (2026-09-25): taugt er auch fuer Originale, etwa die "files" eines
Exports? Gemessen mit beiden Seiten in beiden Fassungen:
  KI      Kanal-2-Posts: Telegram-Foto (komprimiert) / AIBooru-Original (roh)
  Mensch  Wallhaven-Test: v2-Pack (simuliert hochgeladen) / v1-Pack (roh)
Dann AUC und @1 % fuer jede Kombination. Faellt roh-gegen-roh deutlich ab,
braucht das Training rohe Fassungen auf BEIDEN Seiten.

    python lab/detektor/unkomprimiert.py v3a
"""
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from kandidat import logit, logit_stapel                     # noqa: E402
from lesen import DATEN, Pack                    # noqa: E402
from train import auc                            # noqa: E402
from train2 import tpr_bei                       # noqa: E402

RG = ("clip_mid", "dino_mid")


def main():
    tag = sys.argv[1]
    koepfe = []
    s = 0
    while os.path.exists(os.path.join(DATEN, f"kopf_{tag}_s{s}.npz")):
        k = np.load(os.path.join(DATEN, f"kopf_{tag}_s{s}.npz"))
        koepfe.append({x: k[x] for x in k.files})
        s += 1
    print(f"{len(koepfe)} Koepfe {tag}", flush=True)

    def bewerte(X):
        return np.mean([[logit(x, k) for x in X] for k in koepfe], 0)

    from PIL import Image
    from norm import normiere
    from pruefe import stapelrechner
    rechner = stapelrechner(list(RG))

    def einbetten(bilder_fn, n):
        out = []
        with ThreadPoolExecutor(8) as ex:
            for a in range(0, n, 32):
                bs = list(ex.map(bilder_fn, range(a, min(n, a + 32))))
                out.append(np.concatenate([rechner[r](bs, ex) for r in RG], 1))
        return np.concatenate(out)

    # KI: Kanal 2, dieselben Posts wie im Test (nach beiden Sperren)
    ids = np.load(os.path.join(DATEN, "kanal2_emb.npz"))["ids"]
    Z = {z["id"]: z for z in json.load(open(os.path.join(DATEN, "kanal2.json"), encoding="utf-8"))}
    Z = [Z[i] for i in ids if Z[i].get("datei")]
    ki_foto = einbetten(lambda j: Image.open(io.BytesIO(normiere(Z[j]["foto"])[0])).convert("RGB"), len(Z))
    ki_roh = einbetten(lambda j: Image.open(io.BytesIO(normiere(Z[j]["datei"])[0])).convert("RGB"), len(Z))

    # Mensch: Wallhaven-Test, roh (v1-Pack) und simuliert hochgeladen (v2-Pack)
    p1, p2 = Pack("norm512"), Pack("v2")
    wid1 = {w: i for i, w in enumerate(p1.wh_id) if p1.label[i] == 0 and p1.split[i] == "test" and p1.ok[i]}
    paare = [(wid1[w], i) for i, w in enumerate(p2.wh_id)
             if p2.quelle[i] == "wallhaven" and p2.split[i] == "test" and p2.ok[i] and w in wid1]
    rng = np.random.default_rng(5)
    paare = [paare[j] for j in rng.choice(len(paare), min(4000, len(paare)), replace=False)]
    m_roh = einbetten(lambda j: p1.bild(paare[j][0]), len(paare))
    m_sim = einbetten(lambda j: p2.bild(paare[j][1]), len(paare))

    S = {"KI Telegram-Foto": bewerte(ki_foto), "KI Original": bewerte(ki_roh),
         "Mensch hochgeladen": bewerte(m_sim), "Mensch Original": bewerte(m_roh)}
    print(f"\nKI {len(Z)} Posts, Mensch {len(paare)} Wallhaven-Testbilder")
    erg = {}
    for mn in ("Mensch hochgeladen", "Mensch Original"):
        for kn in ("KI Telegram-Foto", "KI Original"):
            neg, pos = S[mn], S[kn]
            y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
            a, t = auc(np.r_[neg, pos], y), tpr_bei(neg, pos)
            erg[f"{mn} | {kn}"] = {"auc": a, "tpr1": t}
            print(f"  {mn:<20} gegen {kn:<18} AUC {a:.4f}  @1% {t:.1%}")
    for n, v in S.items():
        print(f"  Median-Logit {n:<20} {np.median(v):+.2f}")
    with open(os.path.join(DATEN, f"unkomprimiert_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(erg, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
