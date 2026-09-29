r"""p2_test.py — does a head work for digital art in general? Each head at ITS stored threshold.

  false positives  human, never in training: Danbooru 2026 (anime), Danbooru non-anime per style tag,
                   Wallhaven general (before 2022)
  hit rate         CivitAI semi-realistic/3D test split and every Telegram AI channel
Also home of `merkmale(pack, split)`: all cached backbone features of a pack, side by side.

    python p2_test.py v3 p8x p8sx
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from danbooru_test import bewerte, koepfe_von   # noqa: E402
from lesen import DATEN, Pack, kanal_X, rueckgrate_da   # noqa: E402

RG = ("clip_mid", "dino_mid")


class Spalten:
    """Mehrere Merkmals-Memmaps als ein Array nebeneinander (Zeilenzugriff liest nur die Zeilen)."""
    def __init__(self, arrays):
        self.a = arrays
        self.shape = (len(arrays[0]), sum(x.shape[1] for x in arrays))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return np.concatenate([np.asarray(x[idx]) for x in self.a], 1)


def merkmale(stamm, split=None):
    """Merkmale eines Packs in RG_ALLE-Reihenfolge, soweit eingebettet (CLIP+DINO, ggf. SigLIP).
    ok = Zeilen, die in ALLEN geladenen Rueckgraten eingebettet sind."""
    n = f"_{stamm}" + (f"_{split}" if split else "")
    rg = rueckgrate_da("emb_", n)
    X = Spalten([np.load(os.path.join(DATEN, f"emb_{r}{n}.npy"), mmap_mode="r") for r in rg])
    ok = np.all([np.load(os.path.join(DATEN, f"emb_{r}{n}_ok.npy")) for r in rg], 0)
    return X, ok


def main():
    tags = sys.argv[1:]
    pd, pp = Pack("v3d"), Pack("p2z")
    Xd, okd = merkmale("v3d")
    Xp, okp = merkmale("p2z", "test")
    wd = np.load(os.path.join(DATEN, "p2z_wd.npz"))
    okp &= pp.ok & ~wd["gesperrt"]
    gruppen = collections.OrderedDict()
    gruppen["M Danbooru 2026 (Anime)"] = ("d", np.flatnonzero(okd & pd.ok & (pd.split == "test")))
    tp = okp & (pp.split == "test")
    gruppen["M Wallhaven general <2022"] = ("p", np.flatnonzero(tp & (pp.quelle == "wallhaven_general")))
    dn = tp & (pp.quelle == "danbooru_nichtanime")
    gruppen["M Danbooru Nicht-Anime (alle)"] = ("p", np.flatnonzero(dn))
    stile = np.array([str(z.get("stil", "")) for z in pp.m])
    for t in sorted({x for s in stile[dn] for x in s.split()}):
        i = np.flatnonzero(dn & np.array([t in s.split() for s in stile]))
        if len(i) >= 40:
            gruppen[f"M   {t}"] = ("p", i)
    gruppen["K CivitAI halbrealistisch/3D"] = ("p", np.flatnonzero(tp & (pp.quelle == "civitai_stil")))
    kanaele = sorted(f[6:-8] for f in os.listdir(DATEN) if f.startswith("kanal_") and f.endswith("_emb.npz"))

    ergebnis = {}
    for tag in tags:
        k = koepfe_von(tag)
        schw = float(k[0]["schwelle_1"])
        z = {}
        for name, (q, idx) in gruppen.items():
            X = Xd if q == "d" else Xp
            s = bewerte(np.asarray(X[idx]), k)
            z[name] = (len(idx), float((s > schw).mean()))
        for kn in kanaele:
            s = bewerte(kanal_X(kn)[0], k)
            z["K " + kn] = (len(s), float((s > schw).mean()))
        ergebnis[tag] = z
    namen = list(next(iter(ergebnis.values())))
    print(f"{'':<34}{'n':>7}" + "".join(f"{t:>10}" for t in tags))
    print("M = menschlich (Fehlalarm, soll ~1 %), K = KI (Treffer, soll hoch)")
    for n in namen:
        print(f"{n:<34}{ergebnis[tags[0]][n][0]:>7}" + "".join(f"{ergebnis[t][n][1]:>9.1%} " for t in tags))
    with open(os.path.join(DATEN, "p2_test.json"), "w", encoding="utf-8") as f:
        json.dump(ergebnis, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
