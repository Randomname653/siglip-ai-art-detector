r"""familien2.py — geschlossene Modelle (Midjourney/Niji, DALL-E, ChatGPT), die der
Kopf noch nie gesehen hat: die per aibooru.py --tag nachgeladenen Bilder, die in
keinem Pack stecken. Gegen die vier menschlichen Telegram-Testkanaele.

    python lab/detektor/familien2.py v3a
"""
import collections
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from pfade import DATASETS                       # noqa: E402
from familien import FAMILIE                     # noqa: E402
from kandidat import logit, logit_stapel                     # noqa: E402
from lesen import DATEN, Pack                    # noqa: E402
from train import auc                            # noqa: E402
from train2 import tpr_bei                       # noqa: E402


def main():
    tag = sys.argv[1]
    koepfe = []
    while os.path.exists(os.path.join(DATEN, f"kopf_{tag}_s{len(koepfe)}.npz")):
        k = np.load(os.path.join(DATEN, f"kopf_{tag}_s{len(koepfe)}.npz"))
        koepfe.append({x: k[x] for x in k.files})
    im_pack = {z["src"] for z in Pack("v3z").m}
    E = [json.loads(z) for z in open(f"{DATASETS}/aibooru/manifest.jsonl", encoding="utf-8")]
    neu = [e for e in E if e.get("datei") and e["datei"] not in im_pack
           and any(t in FAMILIE for t in (e.get("model") or "").split())]
    print(f"{len(neu)} neue Bilder geschlossener/benannter Familien", flush=True)

    from PIL import Image
    from norm import normiere
    from pruefe import stapelrechner
    rechner = stapelrechner(["clip_mid", "dino_mid"])
    X = []
    with ThreadPoolExecutor(8) as ex:
        for a in range(0, len(neu), 32):
            bs = list(ex.map(lambda e: Image.open(io.BytesIO(normiere(e["datei"])[0])).convert("RGB"),
                             neu[a:a + 32]))
            X.append(np.concatenate([rechner[r](bs, ex) for r in ("clip_mid", "dino_mid")], 1))
    X = np.concatenate(X)
    s = np.mean([[logit(x, k) for x in X] for k in koepfe], 0)

    m = np.load(os.path.join(DATEN, f"score_{tag}_B.npy"))
    neg = m[Pack("v2").label[m[:, 0].astype(int)] == 0, 1]
    g = collections.defaultdict(list)
    for j, e in enumerate(neu):
        for f in {FAMILIE[t] for t in e["model"].split() if t in FAMILIE}:
            g[f].append(j)
    for f, j in sorted(g.items(), key=lambda x: -len(x[1])):
        pos = s[j]
        y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        print(f"  {f:<18} n={len(pos):>5}  AUC {auc(np.r_[neg, pos], y):.4f}  @1% {tpr_bei(neg, pos):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
