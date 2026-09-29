r"""leak.py — verraet die Zurichtung die Klasse?

Bevor irgendeinem Trainingsergebnis zu trauen ist, muss klar sein, dass der
Detektor die HERKUNFT lernt und nicht die Behandlung der Dateien. Drei
Pruefungen, jeweils trainiert auf `train`, gemessen auf `test`:

1. **Nur die Bildgroesse**, roh und normiert. Auf den Rohdateien sollte das
   verheerend gut funktionieren — die Negative stehen alle auf 1024, sd15 und
   nai_v1 deutlich darunter, die zwoelf anderen darueber. Nach norm.py muss es
   auf Zufall fallen. Tut es das nicht, war die Normierung wirkungslos.

2. **Stumpfe Bildstatistiken, roh gegen normiert.** Dieselben acht Merkmale wie
   im Frames-Trennbarkeitstest (lab/frames/separability.py). Der Abstand
   zwischen roh und normiert ist das, was die Normierung an Verarbeitungsspur
   entfernt hat.

3. **Was danach uebrig bleibt**, ist nicht automatisch ein Leck: Generatoren
   haben echte Fingerabdruecke im Rauschen und in der Schaerfe, und genau die
   soll ein Detektor ja finden. Aber diese Zahl ist die Latte, die das
   gelernte Modell deutlich reissen muss — sonst hat es nichts gelernt, was
   ueber Pixelstatistik hinausgeht.

    python lab/detektor/leak.py
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from pfade import PROJEKT  # noqa: E402
sys.path.insert(0, os.path.join(PROJEKT, "lab", "frames"))
from lesen import DATEN, Pack                    # noqa: E402

RNG = np.random.default_rng(20260924)
N_TRAIN, N_TEST = 400, 200          # je Generator
N_MENSCH_TRAIN, N_MENSCH_TEST = 3000, 1500


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _feat_roh(pfad):
    import cv2
    from PIL import Image
    from separability import features
    try:
        rgb = np.array(Image.open(pfad).convert("RGB"))
        return features(rgb)
    except Exception:
        return None


def _feat_norm(b):
    import io
    from PIL import Image
    from separability import features
    return features(np.array(Image.open(io.BytesIO(b)).convert("RGB")))


def stichprobe(pack):
    """Je Generator N_TRAIN/N_TEST, Mensch groesser — er wird gegen jeden
    einzelnen Generator gemessen."""
    auswahl = []
    for split, nm, ng in (("train", N_MENSCH_TRAIN, N_TRAIN),
                          ("test", N_MENSCH_TEST, N_TEST)):
        for m in np.unique(pack.model):
            idx = pack.wo(split=split, model=m)
            n = nm if m == "mensch" else ng
            auswahl.extend(RNG.choice(idx, min(n, len(idx)), replace=False))
    return np.sort(np.array(auswahl))


def bewerte(X, idx, pack, name):
    """xgboost auf train, AUC auf test — gesamt und je Generator."""
    import xgboost as xgb
    tr = pack.split[idx] == "train"
    te = pack.split[idx] == "test"
    y = pack.label[idx]
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", verbosity=0)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    mod = pack.model[idx][te]
    yt = y[te]
    erg = {"gesamt": auc(p, yt)}
    mensch = mod == "mensch"
    for m in sorted(set(mod) - {"mensch"}):
        k = mensch | (mod == m)
        erg[m] = auc(p[k], yt[k])
    print(f"\n  {name}: gesamt AUC {erg['gesamt']:.3f}")
    return erg


def main():
    pack = Pack()
    ergebnis = {}

    # 1. nur die Groesse — ueber den GANZEN Datensatz, kostet nichts
    alle = np.flatnonzero(pack.ok & np.isin(pack.split, ["train", "test"]))
    w0, h0, w, h = (pack.size[alle, k].astype(float) for k in range(4))
    ergebnis["groesse_roh"] = bewerte(
        np.c_[w0, h0, w0 * h0, w0 / h0], alle, pack, "nur Groesse, roh")
    ergebnis["groesse_norm"] = bewerte(
        np.c_[w, h, w * h, w / h], alle, pack, "nur Groesse, normiert")

    # 2. stumpfe Merkmale auf einer Stichprobe, roh und normiert
    idx = stichprobe(pack)
    print(f"\nStichprobe: {len(idx)} Bilder", flush=True)
    namen = None
    t0 = time.time()
    with ProcessPoolExecutor(16) as ex:
        roh = list(ex.map(_feat_roh, [pack.m[i]["src"] for i in idx], chunksize=16))
    print(f"  roh: {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(16) as ex:
        norm = list(ex.map(_feat_norm, [bytes(pack.bytes(i)) for i in idx],
                           chunksize=16))
    print(f"  normiert: {time.time()-t0:.0f}s", flush=True)

    ok = np.array([r is not None for r in roh])
    idx = idx[ok]
    roh = [r for r in roh if r is not None]
    norm = [n for n, k in zip(norm, ok) if k]
    namen = list(roh[0])
    Xr = np.array([[r[n] for n in namen] for r in roh])
    Xn = np.array([[r[n] for n in namen] for r in norm])
    ergebnis["stumpf_roh"] = bewerte(Xr, idx, pack, "stumpfe Merkmale, roh")
    ergebnis["stumpf_norm"] = bewerte(Xn, idx, pack, "stumpfe Merkmale, normiert")

    # einzelne Merkmale: welches traegt, roh gegen normiert?
    te = pack.split[idx] == "test"
    y = pack.label[idx][te]
    ergebnis["einzeln"] = {}
    for j, n in enumerate(namen):
        ar, an = auc(Xr[te, j], y), auc(Xn[te, j], y)
        ergebnis["einzeln"][n] = {"roh": max(ar, 1 - ar), "norm": max(an, 1 - an)}

    ergebnis["stichprobe"] = int(len(idx))
    with open(os.path.join(DATEN, "leak.json"), "w", encoding="utf-8") as f:
        json.dump(ergebnis, f, indent=1)

    # Tabelle
    spalten = ["groesse_roh", "groesse_norm", "stumpf_roh", "stumpf_norm"]
    print(f"\n{'':<17}" + "".join(f"{s:>14}" for s in spalten))
    for m in ["gesamt"] + sorted(k for k in ergebnis["stumpf_norm"] if k != "gesamt"):
        print(f"{m:<17}" + "".join(f"{ergebnis[s].get(m, float('nan')):>14.3f}"
                                   for s in spalten))
    print(f"\n{'Merkmal':<14}{'roh':>8}{'norm':>8}")
    for n, v in ergebnis["einzeln"].items():
        print(f"{n:<14}{v['roh']:>8.3f}{v['norm']:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
