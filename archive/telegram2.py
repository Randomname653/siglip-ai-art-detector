r"""telegram2.py — Fehlalarm auf echter, UNVORSORTIERTER Telegram-Kunst.

Die erste Telegram-Probe (telegram.py) ist nicht deutbar: 57 % der als
`kept_as_human` gelabelten Bilder haben exakt die Groesse eines 1,5-fach
hochskalierten SDXL-Buckets (1248x1824 = 1,5 x 832x1216), der beste
Produktionsdetektor gibt ihnen im Median 1,000 — und `kept_ai` ("KI, will ich
aber behalten") gibt es erst seit dem 26.07.2026. Vorher war `kept_as_human`
die einzige Art, ein Bild zu behalten. Ob diese Labels "menschlich" oder
"behalten" heissen, kann nur der Mensch sagen, der sie gesetzt hat.

Diese Probe braucht die Labels nicht:

    Mensch  Telegram-Posts von VOR 2022 aus lab/../unbearbeitet — da gab es die
            Generatoren noch nicht (Aera-Alibi; bewusst vor 2022 statt vor
            08/2022, weil Midjourney und DALL-E 2 ab Fruehjahr 2022 liefen).
            Zufaellig gezogen, nicht vorsortiert. Fotos entfernt mit genau dem
            Filter, den imagesort.py benutzt (anime_real, real >= 0,75).
    KI      die von dir selbst bestaetigten KI-Bilder (ai_confirmed, kept_ai).

Verzerrung, und zugunsten wessen: ins Review kam nur, was die Produktion
markiert hatte. Die KI-Seite ist damit auf Bilder vorsortiert, die die
Produktion erkennt — das beguenstigt die Produktion.

Die Produktionsdetektoren sehen die Rohdatei, wie im Betrieb; der neue Detektor
seine Zurichtung (norm.normiere). Alle auf denselben Bildern.

    python lab/detektor/telegram2.py --n 2500
"""
import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from pfade import PROJEKT, TELEGRAM_EINGANG  # noqa: E402
from lesen import DATEN                          # noqa: E402

BASIS = TELEGRAM_EINGANG
FEATURES = os.path.join(PROJEKT, "features.db")
REAL_MIN = 0.75          # wie REAL_PHOTO_MIN_SCORE in imagesort.py


def vor_2022():
    out = []
    for dp, dn, fn in os.walk(BASIS):
        if "_ai_quarantine" in dp:
            continue
        for f in fn:
            m = re.search(r"@\d\d-\d\d-(\d{4})", f)
            if m and int(m.group(1)) < 2022 and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                out.append(os.path.join(dp, f))
    return out


def ki_bestaetigt():
    con = sqlite3.connect(f"file:{FEATURES}?mode=ro", uri=True, timeout=30)
    rows = con.execute("SELECT path, decision FROM review_log WHERE decision IN "
                       "('ai_confirmed','kept_ai','kept_as_human')").fetchall()
    con.close()
    letzte = {}
    for p, d in rows:
        letzte[p] = d
    return [p for p, d in letzte.items() if d != "kept_as_human" and os.path.exists(p)]


class Pfade:
    """Minimaler Ersatz fuer lesen.Pack, damit messlatte.py die Rohdateien bewertet."""
    def __init__(self, pfade):
        self.p = pfade

    def bild(self, i, aug=False):
        from PIL import Image
        with Image.open(self.p[i]) as im:
            return im.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--kopf", default="wd_eva02_clip_l14_mlp_aug")
    args = ap.parse_args()
    from imgutils.validate import anime_real
    from sklearn.metrics import roc_auc_score, roc_curve

    t0 = time.time()
    kandidaten = vor_2022()
    random.Random(20260924).shuffle(kandidaten)
    print(f"{len(kandidaten)} Telegram-Bilder von vor 2022 ({time.time()-t0:.0f}s)", flush=True)

    def pruefe(p):
        try:
            lab, sc = anime_real(p)
            return p, not (lab == "real" and sc >= REAL_MIN)
        except Exception:
            return p, None
    mensch, fotos, fehler = [], 0, 0
    with ThreadPoolExecutor(8) as ex:
        for s in range(0, len(kandidaten), 400):
            for p, ok in ex.map(pruefe, kandidaten[s:s + 400]):
                if ok is None:
                    fehler += 1
                elif ok:
                    mensch.append(p)
                else:
                    fotos += 1
            if len(mensch) >= args.n:
                break
    mensch = mensch[:args.n]
    ki = ki_bestaetigt()
    print(f"Mensch: {len(mensch)} (dafuer {fotos} Fotos aussortiert, {fehler} unlesbar); "
          f"KI: {len(ki)}", flush=True)

    pfade = mensch + ki
    y = np.r_[np.zeros(len(mensch)), np.ones(len(ki))]
    ergebnis = {}

    from kandidat import score
    s = np.array([score(p, args.kopf) for p in pfade])
    ergebnis["neu"] = s
    print(f"  neu: {time.time()-t0:.0f}s", flush=True)

    import messlatte
    pk = Pfade(pfade)
    idx = np.arange(len(pfade))
    for key, backend, name in messlatte.MODELLE:
        f = messlatte.score_hf if backend == "hf" else messlatte.score_deepghs
        ergebnis[key] = f(pk, idx, name)
        print(f"  {key}: {time.time()-t0:.0f}s", flush=True)

    np.savez(os.path.join(DATEN, "telegram2.npz"), y=y, pfade=np.array(pfade),
             **{k.replace(":", "_"): v for k, v in ergebnis.items()})

    k = np.load(os.path.join(DATEN, f"kopf_{args.kopf}.npz"))
    z = np.log(np.clip(s, 1e-12, 1 - 1e-12) / np.clip(1 - s, 1e-12, 1))
    print(f"\nKalibrierung des neuen Kopfes (Schwellen aus val, eigener Datensatz):")
    for lab, t in (("1 %", float(k["schwelle_1"])), ("5 %", float(k["schwelle_5"]))):
        print(f"  Schwelle fuer {lab}: Fehlalarm auf Telegram-Mensch {(z[y == 0] > t).mean():6.2%}, "
              f"Treffer auf bestaetigter KI {(z[y == 1] > t).mean():6.1%}")

    print(f"\n  {'':<34}{'AUC':>7}{'@1% FA':>9}{'@5% FA':>9}")
    for key, sc in ergebnis.items():
        fpr, tpr, _ = roc_curve(y, sc)
        print(f"  {key:<34}{roc_auc_score(y, sc):>7.3f}{np.interp(0.01, fpr, tpr):>9.1%}"
              f"{np.interp(0.05, fpr, tpr):>9.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
