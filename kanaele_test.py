r"""kanaele_test.py — measure every real Telegram AI channel and cache its features.

Per channel the Telegram photos (as they arrive in practice) go through the head, against the human Telegram
test channels of v2. CLIP+DINO features per channel are cached (daten/kanal_<name>_emb.npz; SigLIP features:
kanal_siglip.py). Only channels never used for training.

    python kanaele_test.py p6
"""
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from kandidat import logit, logit_stapel                     # noqa: E402
from lesen import DATEN, Pack                    # noqa: E402
from train import auc                            # noqa: E402
from train2 import tpr_bei                       # noqa: E402

WURZEL = f"{DATASETS}/telegram_ki"
RG = ("clip_mid", "dino_mid")


def merkmale(name):
    ziel = os.path.join(DATEN, f"kanal_{name}_emb.npz")
    E = [json.loads(z) for z in open(os.path.join(WURZEL, name, "manifest.jsonl"), encoding="utf-8")]
    # gesperrte Zeilen (schutz_*: nur noch id + grund) haben keine Datei mehr
    pfade = sorted({e.get("telegram") or e.get("datei") for e in E
                    if (e.get("telegram") or e.get("datei")) and not e.get("grund")
                    and (e.get("art", "telegram") == "telegram")})
    if os.path.exists(ziel):
        d = np.load(ziel, allow_pickle=False)
        alt = list(d["pfade"])
        if alt == pfade:
            return d["X"], pfade
        # Cache enthaelt alle aktuellen Bilder (nur gesperrte fielen weg): Zeilen auswaehlen statt neu
        # einbetten — der Cache selbst bleibt unveraendert, damit SigLIP-Caches zeilengleich bleiben
        pos = {p: i for i, p in enumerate(alt)}
        if all(p in pos for p in pfade):
            return d["X"][[pos[p] for p in pfade]], pfade
    from PIL import Image
    from norm import normiere
    from pruefe import stapelrechner
    rechner = stapelrechner(list(RG))
    X = []
    with ThreadPoolExecutor(8) as ex:
        for a in range(0, len(pfade), 32):
            bs = list(ex.map(lambda p: Image.open(io.BytesIO(normiere(p)[0])).convert("RGB"), pfade[a:a + 32]))
            X.append(np.concatenate([rechner[r](bs, ex) for r in RG], 1).astype(np.float16))
    X = np.concatenate(X)
    np.savez(ziel, X=X, pfade=np.array(pfade))
    return X, pfade


def main():
    tag = sys.argv[1]
    koepfe = []
    while os.path.exists(os.path.join(DATEN, f"kopf_{tag}_s{len(koepfe)}.npz")):
        k = np.load(os.path.join(DATEN, f"kopf_{tag}_s{len(koepfe)}.npz"))
        koepfe.append({x: k[x] for x in k.files})
    m = np.load(os.path.join(DATEN, f"score_{tag}_B.npy"))
    p2 = Pack("v2")
    y = p2.label[m[:, 0].astype(int)]
    neg = m[y == 0, 1]
    kanal1 = m[(y == 1) & (p2.grp[m[:, 0].astype(int)] == "ai:ChatExport_2026-05-31 (1)"), 1]
    erg = {}

    def zeile(name, pos):
        yy = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        erg[name] = {"n": int(len(pos)), "auc": auc(np.r_[neg, pos], yy), "tpr1": tpr_bei(neg, pos),
                     "tpr5": tpr_bei(neg, pos, 0.05)}
        e = erg[name]
        print(f"  {name:<22} n={e['n']:>5}  AUC {e['auc']:.4f}  @1% {e['tpr1']:.1%}  @5% {e['tpr5']:.1%}", flush=True)

    print(f"Kopf {tag} ({len(koepfe)} Koepfe), Menschen: {len(neg)} aus 4 Testkanaelen")
    zeile("ai_only_kanal1", kanal1)
    for name in sorted(os.listdir(WURZEL)):
        if not os.path.exists(os.path.join(WURZEL, name, "manifest.jsonl")):
            continue
        X, _ = merkmale(name)
        s = np.mean([logit_stapel(X, k) for k in koepfe], 0)
        zeile(name, s)
    with open(os.path.join(DATEN, f"kanaele_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(erg, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
