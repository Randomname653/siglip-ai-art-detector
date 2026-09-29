r"""train2.py — Kopf auf Datensatz v2 (Wallhaven + Generatoren + Telegram + Ai Only).

Ziel (Entscheidung des Users, 2026-09-24): der Detektor steht ALLEIN und muss
die Produktion auch auf echtem Telegram-Material schlagen.

Trainiert wird wie in v1 nur auf gestoerten Embeddings (norm.stoere). Neu ist,
WO die Schwelle herkommt: aus den Telegram-Validierungskanaelen, nicht mehr
aus Wallhaven. In v1 erzeugte die Wallhaven-Schwelle fuer 1 % Fehlalarm auf
Telegram 16,9 % — die Schwelle muss dort gesetzt werden, wo der Detektor
arbeitet.

Gemessen wird:
  A  eigener Test: Wallhaven gegen die 14 Generatoren, je Generator
  B  Telegram-Test: vier nie gesehene Menschen-Kanaele gegen den nie
     gesehenen KI-Kanal — auf UNGESTOERTEN Embeddings, so wie echte
     Telegram-Bilder im Einsatz ankommen
  C  Fehlalarm je Telegram-Testkanal bei der Schwelle aus val — streuen die
     vier stark, haengt das Ergebnis an einzelnen Kanaelen

    python lab/detektor/train2.py
    python lab/detektor/train2.py --gewichtung quelle
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import DATEN, Pack                    # noqa: E402
from train import kopf_mlp, auc                  # noqa: E402

RUECKGRATE = ["wd_eva02", "clip_l14"]


def lade(pack, aug, split=None):
    """In float16, wie auf der Platte — mit drei Rueckgraten sind es 12.288
    Dimensionen, in float32 oder gar float64 passt das nicht mehr."""
    teile = []
    for r in RUECKGRATE:
        name = f"emb_{r}{'_aug' if aug else ''}_v2{'_' + split if split else ''}"
        X = np.load(os.path.join(DATEN, name + ".npy"))
        ok = np.load(os.path.join(DATEN, name + "_ok.npy"))
        teile.append((X, ok))
    return np.concatenate([t[0] for t in teile], axis=1), np.all([t[1] for t in teile], axis=0)


class Skalierer:
    """Wie sklearns StandardScaler (mean_, scale_), aber in Portionen und ohne
    float64-Kopie des ganzen Satzes — die wollte am 2026-09-25 20,6 GB."""
    def fit(self, X, idx, stueck=20000):
        s = np.zeros(X.shape[1]); q = np.zeros(X.shape[1])
        for a in range(0, len(idx), stueck):
            x = X[idx[a:a + stueck]].astype(np.float64)
            s += x.sum(0); q += (x * x).sum(0)
        n = len(idx)
        self.mean_ = s / n
        self.scale_ = np.sqrt(np.maximum(q / n - self.mean_ ** 2, 1e-12))
        return self

    def transform(self, X, stueck=20000, inplace=False):
        # inplace: ein frisch zusammengesetztes float16-Array direkt ueberschreiben — spart bei drei
        # Rueckgraten (550k x 16k) rund 17 GB Spitzen-RAM
        out = X if inplace and isinstance(X, np.ndarray) and X.dtype == np.float16 else np.empty(X.shape, np.float16)
        m, sc = self.mean_.astype(np.float32), self.scale_.astype(np.float32)
        for a in range(0, len(X), stueck):
            out[a:a + stueck] = (X[a:a + stueck].astype(np.float32) - m) / sc
        return out


def gewichte(pack, idx, art):
    y = pack.label[idx]
    w = np.ones(len(idx), np.float32)
    if art == "quelle":
        # jede Quelle traegt gleich viel Gewicht, egal wie gross sie ist
        q = pack.quelle[idx]
        for s in np.unique(q):
            w[q == s] = 1.0 / (q == s).sum()
    else:
        # nur Mensch gegen Maschine ausgleichen
        for c in (0, 1):
            w[y == c] = 1.0 / (y == c).sum()
    return w / w.mean()


def tpr_bei(s_neg, s_pos, fpr=0.01):
    from sklearn.metrics import roc_curve
    y = np.r_[np.zeros(len(s_neg)), np.ones(len(s_pos))]
    f, t, _ = roc_curve(y, np.r_[s_neg, s_pos])
    return float(np.interp(fpr, f, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gewichtung", default="klasse", choices=["klasse", "quelle"])
    ap.add_argument("--rueckgrat", action="append", default=None,
                    help="Vorgabe: wd_eva02 und clip_l14")
    ap.add_argument("--ohne-generator", action="store_true",
                    help="nur leave-one-generator-out auf Teil A: je Generator ein Kopf "
                         "ohne ihn, gemessen nur auf ihm")
    args = ap.parse_args()
    global RUECKGRATE
    if args.rueckgrat:
        RUECKGRATE = args.rueckgrat
    tag = args.gewichtung + ("" if RUECKGRATE == ["wd_eva02", "clip_l14"] else "_" + "_".join(RUECKGRATE))

    pack = Pack("v2")
    Xa, oka = lade(pack, aug=True)
    oka = oka & pack.ok                          # ohne Telegram-Vorschaubilder
    tr = np.flatnonzero(oka & (pack.split == "train"))
    va = np.flatnonzero(oka & (pack.split == "val"))
    te = np.flatnonzero(oka & (pack.split == "test"))
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)

    sc = Skalierer().fit(Xa, tr)
    if args.ohne_generator:
        # Je Generator: Training und val ohne ihn, Test nur Wallhaven gegen ihn.
        # Weil alle Generatoren auf denselben Motiven liefen, misst das den
        # Generator und nicht das Motiv.
        te_a = te[pack.quelle[te] == "wallhaven"]
        Xn = sc.transform(Xa[te_a])
        erg = {}
        for g in sorted(set(pack.model[tr][pack.quelle[tr] == "generiert"])):
            t1 = time.time()
            tr_g, va_g = tr[pack.model[tr] != g], va[pack.model[va] != g]
            clf, _ = kopf_mlp(sc.transform(Xa[tr_g]), pack.label[tr_g], sc.transform(Xa[va_g]),
                              pack.label[va_g], gewichte(pack, tr_g, args.gewichtung))
            te_g = te[(pack.model[te] == g)]
            sn, sp = clf.decision_function(Xn), clf.decision_function(sc.transform(Xa[te_g]))
            erg[g] = {"auc": auc(np.r_[sn, sp], np.r_[np.zeros(len(sn)), np.ones(len(sp))]),
                      "tpr@0.01": tpr_bei(sn, sp)}
            print(f"  ohne {g:<17} AUC {erg[g]['auc']:.4f}  @1% {erg[g]['tpr@0.01']:.1%}  "
                  f"({time.time()-t1:.0f}s)", flush=True)
        with open(os.path.join(DATEN, f"ergebnis_v2_{tag}_ohne.json"), "w", encoding="utf-8") as f:
            json.dump(erg, f, indent=1)
        return 0
    t0 = time.time()
    clf, info = kopf_mlp(sc.transform(Xa[tr]), pack.label[tr], sc.transform(Xa[va]),
                         pack.label[va], gewichte(pack, tr, args.gewichtung))
    print(f"Kopf trainiert in {time.time()-t0:.0f}s, val AUC {info['val_auc']:.4f}", flush=True)

    s_va = clf.decision_function(sc.transform(Xa[va]))
    tg_va = pack.quelle[va] == "telegram"
    schwelle = {f: float(np.quantile(s_va[tg_va], 1 - f)) for f in (0.01, 0.05)}
    erg = {"info": {k: v for k, v in info.items()}, "schwellen": schwelle,
           "gewichtung": args.gewichtung}

    # A — eigener Test, gestoert
    s_te_aug = clf.decision_function(sc.transform(Xa[te]))
    q, m = pack.quelle[te], pack.model[te]
    neg_a = s_te_aug[q == "wallhaven"]
    erg["A"] = {"gesamt": {"auc": auc(np.r_[neg_a, s_te_aug[q == "generiert"]],
                                      np.r_[np.zeros(len(neg_a)), np.ones((q == "generiert").sum())]),
                           "tpr@0.01": tpr_bei(neg_a, s_te_aug[q == "generiert"])}}
    for g in sorted(set(m[q == "generiert"])):
        pos = s_te_aug[(q == "generiert") & (m == g)]
        erg["A"][g] = {"auc": auc(np.r_[neg_a, pos], np.r_[np.zeros(len(neg_a)), np.ones(len(pos))]),
                       "tpr@0.01": tpr_bei(neg_a, pos)}

    # B — Telegram-Test auf UNGESTOERTEN Embeddings
    Xc, okc = lade(pack, aug=False, split="test")
    tb = te[okc[te] & np.isin(pack.quelle[te], ["telegram", "ai_only"])]
    s_b = clf.decision_function(sc.transform(Xc[tb]))
    yb = pack.label[tb]
    erg["B"] = {"auc": auc(s_b, yb), "tpr@0.01": tpr_bei(s_b[yb == 0], s_b[yb == 1]),
                "tpr@0.05": tpr_bei(s_b[yb == 0], s_b[yb == 1], 0.05),
                "n_mensch": int((yb == 0).sum()), "n_ki": int((yb == 1).sum()),
                "fa_bei_val_1": float((s_b[yb == 0] > schwelle[0.01]).mean()),
                "tr_bei_val_1": float((s_b[yb == 1] > schwelle[0.01]).mean())}
    # C — je Testkanal
    erg["C"] = {}
    for k in sorted(set(pack.grp[tb][yb == 0])):
        mk = (pack.grp[tb] == k) & (yb == 0)
        erg["C"][k] = {"n": int(mk.sum()), "fa_bei_val_1": float((s_b[mk] > schwelle[0.01]).mean())}
    np.save(os.path.join(DATEN, f"score_v2_{tag}_B.npy"), np.c_[tb, s_b])
    np.save(os.path.join(DATEN, f"score_v2_{tag}_A.npy"), np.c_[te, s_te_aug])

    # Kopf speichern (wie train.speichern, MLP)
    g = clf.gewichte
    np.savez(os.path.join(DATEN, f"kopf_v2_{tag}.npz"), art="mlp",
             mean=sc.mean_, scale=sc.scale_, w1=g["0.weight"], b1=g["0.bias"],
             w2=g["3.weight"], b2=g["3.bias"], rueckgrate=np.array(RUECKGRATE),
             schwelle_1=schwelle[0.01], schwelle_5=schwelle[0.05])
    with open(os.path.join(DATEN, f"ergebnis_v2_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(erg, f, indent=1)

    print(f"\nA  eigener Test (gestoert): AUC {erg['A']['gesamt']['auc']:.4f}, "
          f"@1% {erg['A']['gesamt']['tpr@0.01']:.1%}")
    for gg in sorted(k for k in erg["A"] if k != "gesamt"):
        print(f"     {gg:<17} {erg['A'][gg]['tpr@0.01']:.1%}")
    b = erg["B"]
    print(f"\nB  Telegram-Test ({b['n_mensch']} Mensch aus 4 Kanaelen, {b['n_ki']} KI aus 1 Kanal):"
          f" AUC {b['auc']:.4f}, @1% {b['tpr@0.01']:.1%}, @5% {b['tpr@0.05']:.1%}")
    print(f"   Schwelle aus Telegram-val: Fehlalarm {b['fa_bei_val_1']:.2%}, Treffer {b['tr_bei_val_1']:.1%}")
    print("\nC  Fehlalarm je Testkanal:")
    for k, v in erg["C"].items():
        print(f"     {k[-40:]:<42} n={v['n']:>5}  {v['fa_bei_val_1']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
