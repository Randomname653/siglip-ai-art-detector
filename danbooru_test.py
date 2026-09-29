r"""danbooru_test.py — false positives on human art from AFTER 2022; ensemble scoring helpers.

Danbooru 2026 (bans AI), all ratings (pack v3d): false positives per rating (general/sensitive/questionable/
explicit) at two thresholds — did a head learn "explicit = AI" or "new drawing style = AI"? — plus hit rates on
every Telegram AI channel. Also home of `koepfe_von` (load an ensemble) and `bewerte` (mean logit; each head
picks the feature columns of its own backbones).

    python danbooru_test.py p8x p8sx
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kandidat import logit, logit_stapel                     # noqa: E402
from lesen import DATEN, Pack, kanal_X, spalten_fuer   # noqa: E402

RG = ("clip_mid", "dino_mid")


def koepfe_von(tag):
    k = []
    while os.path.exists(os.path.join(DATEN, f"kopf_{tag}_s{len(k)}.npz")):
        z = np.load(os.path.join(DATEN, f"kopf_{tag}_s{len(k)}.npz"))
        k.append({x: z[x] for x in z.files})
    return k


def bewerte(X, koepfe):
    """Ensemble-Mittel der Logits. X hat die Spalten in RG_ALLE-Reihenfolge (CLIP | DINO | SigLIP,
    soweit eingebettet); jeder Kopf greift sich die Spalten seiner Rueckgrate heraus — so laufen
    Koepfe mit einem, zwei oder drei Rueckgraten auf denselben Merkmalen."""
    rg = [str(r) for r in koepfe[0]["rueckgrate"]]
    sp = spalten_fuer(rg)
    if X.shape[1] == len(sp):
        # X enthaelt genau die Rueckgrate des Kopfes (z. B. kontrolle3/pruefe rechnen nur diese)
        return np.mean([logit_stapel(np.asarray(X, np.float32), k) for k in koepfe], 0)
    if X.shape[1] <= sp.max():
        raise ValueError(f"Kopf braucht {rg}, da sind nur {X.shape[1]} Merkmalsspalten — "
                         f"SigLIP fuer diese Bilder noch nicht eingebettet?")
    X = np.asarray(X[:, sp] if len(sp) < X.shape[1] or (sp != np.arange(X.shape[1])).any() else X, np.float32)
    return np.mean([logit_stapel(X, k) for k in koepfe], 0)


def main():
    pd = Pack("v3d")
    from p2_test import merkmale
    X, ok = merkmale("v3d")
    ok = ok & pd.ok
    p2 = Pack("v2")
    RAT = {"g": "general", "s": "sensitive", "q": "questionable", "e": "explicit"}
    for tag in sys.argv[1:]:
        koepfe = koepfe_von(tag)
        d = np.asarray(X[np.flatnonzero(ok)])
        s = np.zeros(len(pd))
        s[ok] = bewerte(d, koepfe)
        m = np.load(os.path.join(DATEN, f"score_{tag}_B.npy"))
        tg_mensch = m[p2.label[m[:, 0].astype(int)] == 0, 1]
        alt = np.quantile(tg_mensch, 0.99)
        va = ok & (pd.split == "val")
        te = ok & (pd.split == "test")
        neu = np.quantile(s[va], 0.99)
        print(f"\n{tag}: {len(koepfe)} Koepfe, Danbooru val {va.sum()} / test {te.sum()}")
        print(f"  Schwelle A (1 % auf Telegram-Menschen vor 2022): {alt:+.2f}   "
              f"Schwelle B (1 % auf Danbooru-val 2026): {neu:+.2f}")
        print(f"  {'Danbooru-test':<24}{'n':>6}{'FA bei A':>11}{'FA bei B':>11}")
        rating = np.array([str(z.get("nsfw")) for z in pd.m])
        for lab, mk in [("alle", te)] + [(RAT[r], te & (rating == r)) for r in "gsqe"] + \
                       [(a, te & (np.array([z.get("anfrage") for z in pd.m]) == a)) for a in ("official", "score")]:
            if mk.sum():
                print(f"  {lab:<24}{mk.sum():>6}{(s[mk] > alt).mean():>10.2%}{(s[mk] > neu).mean():>10.2%}")
        print(f"  {'Telegram-Menschen':<24}{len(tg_mensch):>6}{(tg_mensch > alt).mean():>10.2%}"
              f"{(tg_mensch > neu).mean():>10.2%}")
        print(f"  {'KI-Kanal':<24}{'n':>6}{'Treffer A':>11}{'Treffer B':>11}")
        for f in sorted(os.listdir(DATEN)):
            if f.startswith("kanal_") and f.endswith("_emb.npz"):
                sk = bewerte(kanal_X(f[6:-8])[0], koepfe)
                print(f"  {f[6:-8]:<24}{len(sk):>6}{(sk > alt).mean():>10.1%}{(sk > neu).mean():>10.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
