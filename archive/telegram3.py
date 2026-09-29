r"""telegram3.py — der faire Vergleich: niemand hat die Bilder vorsortiert.

telegram2.py hat gezeigt: auf deinen bestaetigten KI-Bildern liegt die
Produktion vorn — aber diese Bilder kamen nur ins Review, WEIL die Produktion
sie markiert hatte. Man misst sie an ihrer eigenen Auswahl.

Hier ist keine Seite vorsortiert:

    Mensch  dieselben 2.500 Telegram-Posts von vor 2022 wie in telegram2.py
    KI      2.500 zufaellige Bilder aus <DATASETS>\Ai Only (heruntergeladen,
            unabhaengig von der Produktion)

Die beiden Quellen unterscheiden sich an der Datei: Downloads liegen bei JPEG
q>=94, Telegram bei q<=92 — laut Notiz xgb-meta-training-data trennt das allein
mit AUC 1,0. Deshalb bekommt JEDES Bild dieselbe Telegram-Simulation, bevor ein
Detektor es sieht: LANCZOS auf lange Kante 640 (liegt unter fast allen Bildern
beider Seiten), JPEG q87 mit 4:2:0. Bilder mit langer Kante unter 640 fallen
heraus — sie muessten hochgerechnet werden. Alle acht Detektoren bekommen
dieselben Bytes.

Rest-Asymmetrie, benannt: die KI-Bilder werden weniger stark verkleinert
(Median 768 -> 640) als die Telegram-Bilder (meist 1280 -> 640).

    python lab/detektor/telegram3.py
"""
import io
import os
import random
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from pfade import PROJEKT  # noqa: E402
from pfade import DATASETS                       # noqa: E402
from lesen import DATEN                          # noqa: E402

KI_ORDNER = f"{DATASETS}/Ai Only"
LANG, Q = 640, 87


DOPPELT = False     # per --doppelt, siehe main


def _hochladen(im):
    """Was Telegram beim Hochladen macht: lange Kante hoechstens 1280, JPEG
    um q87. Die Telegram-Posts haben das hinter sich, die Downloads nicht."""
    w, h = im.size
    if max(w, h) > 1280:
        f = 1280 / max(w, h)
        im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=87, subsampling=2)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def tg_sim(pfad, erst_hochladen=False):
    with Image.open(pfad) as im:
        im = im.convert("RGB")
        if erst_hochladen:
            im = _hochladen(im)
        w, h = im.size
        if max(w, h) < LANG:
            return None
        f = LANG / max(w, h)
        im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=Q, subsampling=2)
        return buf.getvalue()


class Bytes:
    def __init__(self, b):
        self.b = b

    def bild(self, i, aug=False):
        return Image.open(io.BytesIO(self.b[i])).convert("RGB")


def main():
    """--doppelt: die KI-Bilder bekommen VOR der gemeinsamen Simulation eine
    Telegram-artige Hochlade-Kompression. Dann haben beide Seiten zwei
    Kompressions-Generationen statt Mensch zwei, KI eine. Faellt ein Detektor
    dadurch stark, hat er im einfachen Test die Kompressionsgeschichte gemessen.

    --stumpf: zusaetzlich die acht stumpfen Bildstatistiken aus leak.py auf den
    simulierten Bildern — trennen die schon ohne gelernten Detektor, ist der
    Test selbst nicht sauber."""
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.metrics import roc_auc_score, roc_curve
    global DOPPELT
    DOPPELT = "--doppelt" in sys.argv
    t0 = time.time()
    t2 = np.load(os.path.join(DATEN, "telegram2.npz"))
    mensch = [p for p, y in zip(t2["pfade"], t2["y"]) if y == 0]
    ki = [os.path.join(dp, f) for dp, _, fn in os.walk(KI_ORDNER) for f in fn
          if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    random.Random(20260924).shuffle(ki)

    def sim_liste(pfade, n, erst=False):
        out = []
        with ThreadPoolExecutor(12) as ex:
            for s in range(0, len(pfade), 500):
                for b in ex.map(lambda p: tg_sim(p, erst) if os.path.exists(p) else None,
                                pfade[s:s + 500]):
                    if b:
                        out.append(b)
                if len(out) >= n:
                    break
        return out[:n]

    bm = sim_liste(mensch, 2500)
    bk = sim_liste(ki, len(bm), erst=DOPPELT)
    n = min(len(bm), len(bk))
    bm, bk = bm[:n], bk[:n]
    alle = bm + bk
    y = np.r_[np.zeros(n), np.ones(n)]
    print(f"je {n} Bilder, beide durch Telegram-Simulation (lange Kante {LANG}, q{Q} 4:2:0), "
          f"{time.time()-t0:.0f}s", flush=True)

    if "--stumpf" in sys.argv:
        sys.path.insert(0, os.path.join(PROJEKT, "lab", "frames"))
        from separability import features
        from sklearn.model_selection import cross_val_predict
        from xgboost import XGBClassifier
        F = [features(np.array(Image.open(io.BytesIO(b)).convert("RGB"))) for b in alle]
        X = np.array([[f[k] for k in F[0]] for f in F])
        p = cross_val_predict(XGBClassifier(n_estimators=200, max_depth=4, verbosity=0),
                              X, y, cv=5, method="predict_proba")[:, 1]
        print(f"  stumpfe Merkmale, 5-fach kreuzvalidiert: AUC {roc_auc_score(y, p):.3f}")
        for j, k_ in enumerate(F[0]):
            a_ = roc_auc_score(y, X[:, j])
            print(f"     {k_:<14} {max(a_, 1 - a_):.3f}")

    from kandidat import score
    erg = {"neu": np.array([score(io.BytesIO(b), "wd_eva02_clip_l14_mlp_aug") for b in alle])}
    print(f"  neu: {time.time()-t0:.0f}s", flush=True)
    import messlatte
    pk, idx = Bytes(alle), np.arange(len(alle))
    for key, backend, name in messlatte.MODELLE:
        f = messlatte.score_hf if backend == "hf" else messlatte.score_deepghs
        erg[key] = f(pk, idx, name)
        print(f"  {key}: {time.time()-t0:.0f}s", flush=True)
    np.savez(os.path.join(DATEN, "telegram3" + ("_doppelt" if DOPPELT else "") + ".npz"), y=y,
             **{k.replace(":", "_"): v for k, v in erg.items()})

    k = np.load(os.path.join(DATEN, "kopf_wd_eva02_clip_l14_mlp_aug.npz"))
    z = np.log(np.clip(erg["neu"], 1e-12, 1 - 1e-12) / np.clip(1 - erg["neu"], 1e-12, 1))
    for lab, t in (("1 %", float(k["schwelle_1"])), ("5 %", float(k["schwelle_5"]))):
        print(f"  neue val-Schwelle {lab}: Fehlalarm Telegram-Mensch {(z[y==0] > t).mean():6.2%}, "
              f"Treffer KI {(z[y==1] > t).mean():6.1%}")
    print(f"\n  {'':<34}{'AUC':>7}{'@1% FA':>9}{'@5% FA':>9}")
    for key, sc in erg.items():
        fpr, tpr, _ = roc_curve(y, sc)
        print(f"  {key:<34}{roc_auc_score(y, sc):>7.3f}{np.interp(0.01, fpr, tpr):>9.1%}"
              f"{np.interp(0.05, fpr, tpr):>9.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
