r"""kombination.py — does any production detector add anything on top of our head?

On the bench.py image set: our head logit plus the scores of the seven production models. Logistic regression
on [head, detector], fitted on the human val groups + AI generator test splits, evaluated on the human test
groups + Telegram channels (unseen by every model). Result: no detector improves p8sx
(AUC 0.9976 alone, 0.9975-0.9977 combined).

    python kombination.py [--kopf p8sx]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench                                      # noqa: E402
import messlatte                                  # noqa: E402
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from gleich_fa import auc                        # noqa: E402
from lesen import DATEN                          # noqa: E402


def logit(s):
    s = np.clip(np.asarray(s, np.float64), 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kopf", default="p8sx")
    a = ap.parse_args()
    # Gruppen genau wie bench.py (gleicher Zufall -> gleiche Bilder -> Produktions-Scores aus dem Cache)
    gruppen = bench.gruppen_bauen()
    art = {g: v[0] for g, v in gruppen.items()}
    k = koepfe_von(a.kopf)
    s_kopf = {g: bewerte(np.asarray(v[2], np.float32), k) for g, v in gruppen.items()}
    gkey = {g: bench.schluessel(v[1]) for g, v in gruppen.items()}
    prod = {}
    for key, _, _ in messlatte.MODELLE:
        det = key.replace(":", "_")
        prod[key] = {g: logit(np.load(os.path.join(DATEN, "bench_cache", f"{det}_{gkey[g]}.npy"))) for g in gruppen}

    def sammeln(arten, quelle):
        X, y = [], []
        for g in gruppen:
            if art[g] in arten:
                X.append(quelle[g])
                y.append(np.full(len(quelle[g]), 0 if art[g].startswith("m_") else 1))
        return np.concatenate(X), np.concatenate(y)

    from sklearn.linear_model import LogisticRegression
    lern = ("m_val", "k_gen")
    test = ("m_test", "k_kanal")

    def bewerten(merkmale_lern, merkmale_test, y_lern, y_test, name):
        if merkmale_lern.ndim == 1:
            z_l, z_t = merkmale_lern, merkmale_test
        else:
            m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(merkmale_lern, y_lern)
            z_l, z_t = m.decision_function(merkmale_lern), m.decision_function(merkmale_test)
        t = np.quantile(z_l[y_lern == 0], 0.99)
        return {"name": name, "auc": auc(z_t[y_test == 0], z_t[y_test == 1]),
                "fa": float(np.mean(z_t[y_test == 0] > t)), "treffer": float(np.mean(z_t[y_test == 1] > t))}

    kl, yl = sammeln(lern, s_kopf)
    kt, yt = sammeln(test, s_kopf)
    zeilen = [bewerten(kl, kt, yl, yt, f"{a.kopf} allein")]
    for key in prod:
        pl, _ = sammeln(lern, prod[key])
        pt, _ = sammeln(test, prod[key])
        zeilen.append(bewerten(pl, pt, yl, yt, f"{key} allein"))
        zeilen.append(bewerten(np.c_[kl, pl], np.c_[kt, pt], yl, yt, f"{a.kopf} + {key}"))
    alle_l = np.column_stack([kl] + [sammeln(lern, prod[x])[0] for x in prod])
    alle_t = np.column_stack([kt] + [sammeln(test, prod[x])[0] for x in prod])
    zeilen.append(bewerten(alle_l, alle_t, yl, yt, f"{a.kopf} + alle sieben"))
    print(f"{'Kombination':<52}{'AUC':>9}{'Treffer':>10}{'Fehlalarm':>11}")
    for z in zeilen:
        print(f"{z['name']:<52}{z['auc']:>9.4f}{z['treffer']:>9.1%} {z['fa']:>9.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
