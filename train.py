r"""train.py — ein kleiner Kopf auf eingefrorenen Embeddings, ehrlich gemessen.

Drei Fragen, jede auf dem Testsplit, den kein Kopf je gesehen hat:

1. **Wie gut, je Generator.** Die Gesamt-AUC allein waere irrefuehrend: sd15
   und hdm_xut sind trivial zu erkennen und machen zusammen 14 % der
   Positiven aus. Ein Detektor, der die beiden perfekt und qwen_image zu 60 %
   erkennt, saehe im Mittel hervorragend aus. Deshalb jede Zahl je Generator.

2. **An einem festen Fehlalarm.** Die Kennzahl fuer die Praxis ist nicht die
   AUC, sondern: wenn hoechstens 1 % (bzw. 5 %) der menschlichen Bilder
   faelschlich markiert werden duerfen — wie viel von jedem Generator wird
   dann erwischt? Die Schwelle wird auf `val` festgelegt, gemessen auf `test`.
   So entsteht eine Zahl, die man auch einsetzen koennte.

3. **Unbekannte Generatoren (leave one out).** Fuer jeden der vierzehn einmal
   ohne ihn trainieren und nur auf ihm testen. Neue Modelle erscheinen
   laufend; die Frage ist, ob der Detektor die GATTUNG lernt oder nur die
   Exemplare, die er kennt. Weil jeder Generator auf denselben Motiven lief,
   misst das wirklich den Generator und nicht das Motiv.

Die Messlatten (messlatte.py) werden auf denselben Testbildern ausgewertet.

    python lab/detektor/train.py --rueckgrat clip_l14
    python lab/detektor/train.py --rueckgrat wd_eva02 --rueckgrat clip_l14
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import DATEN, Pack                    # noqa: E402

FPR = (0.01, 0.05)


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def schwelle_bei_fpr(s_mensch, fpr):
    """Die Schwelle, ueber der genau `fpr` der menschlichen Bilder liegen."""
    return float(np.quantile(s_mensch, 1 - fpr))


def auswerten(s, idx, pack, schwellen=None):
    """AUC und Trefferquote je Generator. `schwellen` kommt aus val; ohne
    sie wird sie auf den Menschenbildern DIESES Satzes gesetzt (fuer die
    Messlatten, die nur auf test gerechnet wurden — das ist fuer sie eher
    guenstig und damit eine konservative Vergleichsbasis)."""
    mod = pack.model[idx]
    y = pack.label[idx]
    mensch = mod == "mensch"
    if schwellen is None:
        return _auswerten_roc(s, mod, y, mensch)
    erg = {"gesamt": {"auc": auc(s, y)},
           "fpr_ist": {str(f): float((s[mensch] > t).mean())
                       for f, t in schwellen.items()}}
    for f, t in schwellen.items():
        erg["gesamt"][f"tpr@{f}"] = float((s[~mensch] > t).mean())
    for m in sorted(set(mod) - {"mensch"}):
        k = mod == m
        erg[m] = {"auc": auc(np.r_[s[mensch], s[k]],
                             np.r_[np.zeros(mensch.sum()), np.ones(k.sum())])}
        for f, t in schwellen.items():
            erg[m][f"tpr@{f}"] = float((s[k] > t).mean())
    return erg


def _auswerten_roc(s, mod, y, mensch):
    """Fuer die Messlatten: Trefferquote bei festem Fehlalarm ueber die
    ROC-Kurve, interpoliert.

    NICHT ueber `s > quantile(mensch, 0.99)`: die deepghs-Modelle geben oft
    glatt 1,0 aus. Liegen mehr als 1 % der Menschenbilder auf genau 1,0, wird
    die Schwelle 1,0, und "groesser als 1,0" ist nie wahr — am 2026-09-24
    stand eine Messlatte deshalb bei JEDEM Generator auf 0,000 Treffern, bei
    AUC 0,72. Die ROC-Kurve verteilt Gleichstaende richtig."""
    from sklearn.metrics import roc_curve

    def tpr_bei(pos):
        yy = np.r_[np.zeros(mensch.sum()), np.ones(pos.sum())]
        fpr, tpr, _ = roc_curve(yy, np.r_[s[mensch], s[pos]])
        return {f"tpr@{f}": float(np.interp(f, fpr, tpr)) for f in FPR}
    erg = {"gesamt": {"auc": auc(s, y), **tpr_bei(~mensch)},
           "fpr_ist": {str(f): f for f in FPR}}
    for m in sorted(set(mod) - {"mensch"}):
        k = mod == m
        erg[m] = {"auc": auc(np.r_[s[mensch], s[k]],
                             np.r_[np.zeros(mensch.sum()), np.ones(k.sum())]),
                  **tpr_bei(k)}
    return erg


def lade(rueckgrat, pack, aug=False):
    e = "_aug" if aug else ""
    X = np.load(os.path.join(DATEN, f"emb_{rueckgrat}{e}.npy")).astype(np.float32)
    ok = np.load(os.path.join(DATEN, f"emb_{rueckgrat}{e}_ok.npy"))
    assert ok[pack.ok].all(), "Embeddings unvollstaendig"
    return X


def kopf_logreg(Xtr, ytr, Xva, yva, gew_tr):
    """Logistische Regression, C auf val gewaehlt."""
    from sklearn.linear_model import LogisticRegression
    best = None
    # Am 2026-09-24 lag das Optimum auf dem Rand (C=1.0) — deshalb bis 10.
    for C in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=2000, tol=1e-4)
        clf.fit(Xtr, ytr, sample_weight=gew_tr)
        a = auc(clf.decision_function(Xva), yva)
        if best is None or a > best[0]:
            best = (a, C, clf)
    return best[2], {"C": best[1], "val_auc": best[0]}


def kopf_mlp(Xtr, ytr, Xva, yva, gew_tr, epochen=40):
    """Eine versteckte Schicht, fruehes Stoppen auf val."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    d = Xtr.shape[1]
    net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.3),
                        nn.Linear(512, 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    # Die Merkmale bleiben im Arbeitsspeicher, nur die Stapel wandern auf die
    # Karte: bei CLIP- plus DINOv2-Zwischenschichten sind es 10.240 Dimensionen
    # x 225.000 Bilder = 9 GB, neben einer laufenden Extraktion zu viel.
    # float16 im Speicher halbiert noch einmal; gerechnet wird je Stapel in float32
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float16))
    yt = torch.tensor(ytr, dtype=torch.float32)
    wt = torch.tensor(gew_tr, dtype=torch.float32)

    def vorhersage(X):
        with torch.no_grad():
            return torch.cat([net(torch.from_numpy(np.ascontiguousarray(
                X[s:s + 4096], dtype=np.float16)).cuda().float()).squeeze(1).cpu()
                for s in range(0, len(X), 4096)]).numpy()
    best = (-1, None, 0)
    for ep in range(epochen):
        net.train()
        perm = torch.randperm(len(Xt))
        for s in range(0, len(perm), 1024):
            b = perm[s:s + 1024]
            loss = (nn.functional.binary_cross_entropy_with_logits(
                net(Xt[b].cuda(non_blocking=True).float()).squeeze(1), yt[b].cuda(),
                reduction="none") * wt[b].cuda()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        a = auc(vorhersage(Xva), yva)
        if a > best[0]:
            best = (a, {k: v.clone() for k, v in net.state_dict().items()}, ep)
        elif ep - best[2] >= 6:
            break
    net.load_state_dict(best[1])
    net.eval()

    class Kopf:
        gewichte = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}

        def decision_function(self, X):
            return vorhersage(X)
    return Kopf(), {"val_auc": best[0], "epoche": best[2]}


def gewichte(y):
    """Mensch und Maschine gleich schwer — es gibt 1,38-mal so viele Positive."""
    w = np.ones(len(y), np.float32)
    w[y == 0] = (y == 1).sum() / max((y == 0).sum(), 1)
    return w / w.mean()


def lauf(rueckgrat, pack, X, kopf, ohne=None):
    """Einen Kopf trainieren. `ohne` = Generator, der im Training fehlt."""
    from sklearn.preprocessing import StandardScaler
    tr = pack.wo(split="train")
    va = pack.wo(split="val")
    te = pack.wo(split="test")
    if ohne:
        tr = tr[pack.model[tr] != ohne]
        va = va[pack.model[va] != ohne]
    sc = StandardScaler().fit(X[tr])
    Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
    ytr, yva = pack.label[tr], pack.label[va]
    f = kopf_logreg if kopf == "logreg" else kopf_mlp
    clf, info = f(Xtr, ytr, Xva, yva, gewichte(ytr))
    s_va = clf.decision_function(Xva)
    schwellen = {fp: schwelle_bei_fpr(s_va[pack.model[va] == "mensch"], fp)
                 for fp in FPR}
    s_te = clf.decision_function(sc.transform(X[te]))
    info["schwellen"] = schwellen
    info["_kopf"] = (sc, clf)
    return s_te, te, schwellen, info


def speichern(ziel, sc, clf, schwellen, rueckgrate):
    """Den fertigen Kopf ablegen, damit kandidat.py ihn auf beliebigen
    Bildern anwenden kann (Pruefstand, Telegram-Bestand). Nur die logistische
    Regression — ein Mittelwert, eine Skala, ein Gewichtsvektor; das laesst
    sich ohne sklearn-Pickle und ohne Versionsaerger laden."""
    if hasattr(clf, "gewichte"):            # MLP: Linear -> GELU -> Linear
        g = clf.gewichte
        np.savez(ziel, art="mlp", mean=sc.mean_, scale=sc.scale_,
                 w1=g["0.weight"], b1=g["0.bias"], w2=g["3.weight"], b2=g["3.bias"],
                 rueckgrate=np.array(rueckgrate),
                 schwelle_1=schwellen[0.01], schwelle_5=schwellen[0.05])
        return
    np.savez(ziel, art="logreg", mean=sc.mean_, scale=sc.scale_,
             coef=clf.coef_.ravel(), intercept=clf.intercept_,
             rueckgrate=np.array(rueckgrate),
             schwelle_1=schwellen[0.01], schwelle_5=schwellen[0.05])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rueckgrat", action="append", required=True)
    ap.add_argument("--kopf", default="logreg", choices=["logreg", "mlp"])
    ap.add_argument("--ohne-generator", action="store_true",
                    help="zusaetzlich leave-one-generator-out")
    ap.add_argument("--aug", action="store_true",
                    help="auf GESTOERTEN Embeddings trainieren (norm.stoere); "
                         "gemessen wird gestoert UND sauber")
    args = ap.parse_args()

    pack = Pack()
    ergebnis = {}
    endung = "_aug" if args.aug else ""
    name = "+".join(args.rueckgrat) + ":" + args.kopf
    datei = name.replace(":", "_").replace("+", "_") + endung

    X = np.concatenate([lade(r, pack, args.aug) for r in args.rueckgrat], axis=1)
    t0 = time.time()
    s, te, schw, info = lauf(name, pack, X, args.kopf)
    sc, clf = info.pop("_kopf")
    info["schwellen"] = {str(k): v for k, v in info["schwellen"].items()}
    speichern(os.path.join(
        DATEN, f"kopf_{'_'.join(args.rueckgrat)}{'_mlp' if args.kopf == 'mlp' else ''}"
               f"{endung}.npz"), sc, clf, schw, args.rueckgrat)
    ergebnis["voll"] = auswerten(s, te, pack, schw)
    ergebnis["voll"]["info"] = info
    np.save(os.path.join(DATEN, f"score_{datei}.npy"), np.c_[te, s])
    print(f"{name}{endung}: test AUC {ergebnis['voll']['gesamt']['auc']:.4f}  "
          f"({time.time()-t0:.0f}s, {info})", flush=True)

    # Derselbe Kopf auf den UNGESTOERTEN Testbildern. Die Schwellen bleiben
    # die aus val (gestoert) — wie weit die Fehlalarmquote dabei wandert,
    # steht in fpr_ist.
    if args.aug and all(os.path.exists(os.path.join(DATEN, f"emb_{r}.npy"))
                        for r in args.rueckgrat):
        Xc = np.concatenate([lade(r, pack, False) for r in args.rueckgrat], axis=1)
        sc_ = clf.decision_function(sc.transform(Xc[te]))
        ergebnis["sauber"] = auswerten(sc_, te, pack, schw)
        np.save(os.path.join(DATEN, f"score_{datei}_sauber.npy"), np.c_[te, sc_])
        print(f"  derselbe Kopf auf ungestoerten Testbildern: AUC "
              f"{ergebnis['sauber']['gesamt']['auc']:.4f}, Fehlalarm bei "
              f"1-%-Schwelle {ergebnis['sauber']['fpr_ist']['0.01']:.3%}", flush=True)

    if args.ohne_generator:
        ergebnis["ohne"] = {}
        gens = sorted(set(pack.model) - {"mensch"})
        for g in gens:
            t0 = time.time()
            s, te, schw, info = lauf(name, pack, X, args.kopf, ohne=g)
            info.pop("_kopf")
            k = np.isin(pack.model[te], ["mensch", g])
            e = auswerten(s[k], te[k], pack, schw)
            ergebnis["ohne"][g] = e[g]
            print(f"  ohne {g:<16} AUC {e[g]['auc']:.4f}  "
                  f"tpr@1% {e[g]['tpr@0.01']:.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)

    ziel = os.path.join(DATEN, f"ergebnis_{datei}.json")
    with open(ziel, "w", encoding="utf-8") as f:
        json.dump(ergebnis, f, indent=1)

    # Messlatten auf denselben Testbildern — getrennt sauber und gestoert
    mess = {}
    for tag in ("test", "test_aug"):
        idx_m = os.path.join(DATEN, f"mess_{tag}_idx.npy")
        if not os.path.exists(idx_m):
            continue
        mi = np.load(idx_m)
        for p in sorted(glob.glob(os.path.join(DATEN, f"mess_{tag}_*.npy"))):
            key = os.path.basename(p)[len(f"mess_{tag}_"):-4]
            if key == "idx" or (tag == "test" and key.startswith("aug_")):
                continue
            mess[f"{key}|{tag}"] = auswerten(np.load(p), mi, pack)
    if mess:
        with open(os.path.join(DATEN, "ergebnis_messlatten.json"), "w",
                  encoding="utf-8") as f:
            json.dump(mess, f, indent=1)

    # Tabelle: je Generator TPR bei 1 % Fehlalarm
    gens = sorted(set(pack.model) - {"mensch"})
    bed = "test_aug" if args.aug else "test"
    spalten = [("Kopf" + (" gestoert" if args.aug else ""), ergebnis["voll"])]
    if "sauber" in ergebnis:
        spalten.append(("Kopf sauber", ergebnis["sauber"]))
    if "ohne" in ergebnis:
        spalten.append(("ohne ihn", ergebnis["ohne"]))
    spalten += [(k.split("|")[0].replace("deepghs_", "").replace("hf_", "")
                 .replace("mobilenetv3_", "mnv3_").replace("caformer_s36_", "caf_"), v)
                for k, v in mess.items() if k.endswith("|" + bed)]
    print(f"\n(Messlatten: {bed}; ihre Schwelle ist auf den Menschenbildern von "
          f"test selbst gesetzt — fuer sie eher guenstig)")
    print(f"\nTrefferquote bei 1 % Fehlalarm auf Menschenbildern (test):")
    print(f"{'':<17}" + "".join(f"{n[:13]:>14}" for n, _ in spalten))
    for g in ["gesamt"] + gens:
        z = f"{g:<17}"
        for _, e in spalten:
            v = e.get(g, {}).get("tpr@0.01")
            z += f"{v:>14.3f}" if v is not None else f"{'':>14}"
        print(z)
    print(f"{'AUC gesamt':<17}" + "".join(
        f"{e.get('gesamt', {}).get('auc', float('nan')):>14.4f}" for _, e in spalten))
    return 0


if __name__ == "__main__":
    sys.exit(main())
