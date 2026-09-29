r"""telegram.py — der neue Kopf auf echten Telegram-Bildern mit DEINEN Labels.

Die zweite externe Probe neben V5Minor100, und die naeher am Einsatz:
Bilder aus den Telegram-Exporten, die du in der Review-UI selbst entschieden
hast. Genommen werden NUR menschliche Urteile:

    ai_confirmed, kept_ai   -> KI
    kept_as_human           -> Mensch

`auto_quar_def_ai` bleibt draussen: das hat die Produktion selbst entschieden,
es als Wahrheit zu nehmen hiesse, die Produktion an sich selbst zu messen.

Verzerrung, und in welche Richtung: in der review_log steht nur, was es ins
Review geschafft hat — also was die Produktion fuer KI-verdaechtig hielt. Die
menschlichen Bilder hier sind genau die, an denen sie sich geirrt hat: harte
Faelle. Ein Fehlalarm des neuen Kopfes auf ihnen ist eher zu pessimistisch.
Die Produktionsdetektoren selbst werden hier nur zur Orientierung gezeigt —
ihre Zahlen sind durch die Auswahl geschoent, die sie selbst getroffen haben
(siehe Notiz tier2-veto-senkt-recall).

    python lab/detektor/telegram.py --kopf clip_l14_mlp_aug
"""
import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from pfade import PROJEKT  # noqa: E402
from lesen import DATEN                          # noqa: E402

DB = os.path.join(PROJEKT, "features.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kopf", default="clip_l14_mlp_aug")
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score, roc_curve
    from kandidat import score

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)
    rows = con.execute(
        "SELECT path, decision, ml_scores_json FROM review_log WHERE decision IN "
        "('ai_confirmed','kept_ai','kept_as_human')").fetchall()
    con.close()
    # je Pfad die LETZTE Entscheidung
    letzte = {}
    for p, d, ms in rows:
        letzte[p] = (d, ms)
    daten = [(p, 0 if d == "kept_as_human" else 1, ms)
             for p, (d, ms) in letzte.items() if os.path.exists(p)]
    print(f"{len(daten)} Bilder mit eigenem Label und vorhandener Datei "
          f"({sum(1 for _, y, _ in daten if y == 0)} Mensch, "
          f"{sum(1 for _, y, _ in daten if y == 1)} KI)", flush=True)

    s, y, prod, fehler = [], [], [], 0
    t0 = time.time()
    for k, (p, lab, ms) in enumerate(daten, 1):
        try:
            s.append(score(p, args.kopf))
            y.append(lab)
            prod.append(json.loads(ms) if ms else {})
        except Exception as e:
            fehler += 1
            if fehler <= 3:
                print("  Fehler", os.path.basename(p), e)
        if k % 500 == 0:
            print(f"  {k}/{len(daten)}  {k/(time.time()-t0):.1f}/s", flush=True)
    s, y = np.array(s), np.array(y)

    def zeile(name, sc, yy):
        a = roc_auc_score(yy, sc)
        fpr, tpr, _ = roc_curve(yy, sc)
        return (f"  {name:<34}{a:>7.3f}{np.interp(0.01, fpr, tpr):>9.1%}"
                f"{np.interp(0.05, fpr, tpr):>9.1%}")

    print(f"\n{len(s)} bewertet, {fehler} Fehler, {time.time()-t0:.0f}s\n")
    print(f"  {'':<34}{'AUC':>7}{'@1% FA':>9}{'@5% FA':>9}")
    print(zeile(f"neu: {args.kopf}", s, y))
    # Produktion zur Orientierung — durch die eigene Auswahl geschoent
    keys = sorted({k for d in prod for k in d})
    for key in keys:
        m = np.array([key in d and d[key] is not None for d in prod])
        if m.sum() > 100 and len(set(y[m])) == 2:
            print(zeile(f"({key})", np.array([d[key] for d in np.array(prod)[m]]), y[m]))
    np.save(os.path.join(DATEN, f"telegram_{args.kopf}.npy"), np.c_[y, s])
    return 0


if __name__ == "__main__":
    sys.exit(main())
