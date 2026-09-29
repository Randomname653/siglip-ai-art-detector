r"""familien.py — welche KI-Familien erkennt der Kopf schlecht?

AIBooru-Testbilder (Uploader nie im Training) nach Modell-Tag gruppiert, gegen
die vier menschlichen Telegram-Testkanaele. Geschlossene Modelle (Midjourney,
Niji, DALL-E, ChatGPT) laufen lokal nicht und fehlen auf CivitAI als
Basismodell — sind sie die Luecke, die KI-Kanal 1 so schwer macht?

    python lab/detektor/familien.py v3a
"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from kandidat import logit, logit_stapel                     # noqa: E402
from lesen import DATEN, Pack                    # noqa: E402
from train2 import tpr_bei                       # noqa: E402
from train import auc                            # noqa: E402

FAMILIE = {"nijijourney": "Midjourney/Niji", "midjourney": "Midjourney/Niji", "dall-e": "DALL-E/ChatGPT",
           "chatgpt": "DALL-E/ChatGPT", "novelai": "NovelAI", "flux": "Flux"}


def main():
    tag = sys.argv[1]
    koepfe = []
    while os.path.exists(os.path.join(DATEN, f"kopf_{tag}_s{len(koepfe)}.npz")):
        k = np.load(os.path.join(DATEN, f"kopf_{tag}_s{len(koepfe)}.npz"))
        koepfe.append({x: k[x] for x in k.files})
    pz = Pack("v3z")
    T = [np.load(os.path.join(DATEN, f"emb_{r}_v3z_test.npy"), mmap_mode="r") for r in ("clip_mid", "dino_mid")]
    ok = np.load(os.path.join(DATEN, "emb_clip_mid_v3z_test_ok.npy")) & pz.ok
    wd = np.load(os.path.join(DATEN, "v3z_wd.npz"))
    tw = np.flatnonzero(ok & ~wd["gesperrt"] & (pz.split == "test") & (pz.quelle == "aibooru"))
    X = np.concatenate([np.asarray(t[tw]) for t in T], 1).astype(np.float32)
    s = np.mean([[logit(x, k) for x in X] for k in koepfe], 0)
    mensch = np.load(os.path.join(DATEN, f"score_{tag}_B.npy"))
    p2 = Pack("v2")
    neg = mensch[p2.label[mensch[:, 0].astype(int)] == 0, 1]

    gruppe = collections.defaultdict(list)
    for j, i in enumerate(tw):
        tags = (pz.m[i].get("ki_modell") or "").split()
        fams = {FAMILIE[t] for t in tags if t in FAMILIE} or ({"ohne Modell-Tag"} if not tags else {"andere SD-Modelle"})
        for f in fams:
            gruppe[f].append(j)
    print(f"AIBooru-Test: {len(tw)} Bilder, Mensch {len(neg)}")
    for f, j in sorted(gruppe.items(), key=lambda x: -len(x[1])):
        pos = s[j]
        y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        print(f"  {f:<20} n={len(pos):>5}  AUC {auc(np.r_[neg, pos], y):.4f}  @1% {tpr_bei(neg, pos):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
