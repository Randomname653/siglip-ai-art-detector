r"""kalibriere.py — set a head's thresholds on unperturbed human validation images.

Three human groups weighted equally (Danbooru anime 2026, Wallhaven general <2022, Danbooru non-anime); the
1 % (5 %) threshold is where the mean false-positive rate over the groups is 1 % (5 %). An earlier approach
pooled PERTURBED validation images of all sources; large, easy groups dominated and the "1 %" threshold gave
2.0 % / 3.4 % on the test groups. The old threshold is kept in the head as schwelle_1_alt. Test data is not used.

    python kalibriere.py p8sx
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from danbooru_test import bewerte, koepfe_von     # noqa: E402
from gleich_fa import schwelle                     # noqa: E402
from lesen import DATEN, Pack                      # noqa: E402
from p2_test import merkmale                       # noqa: E402


def main():
    tag = sys.argv[1]
    k = koepfe_von(tag)
    pd, pp = Pack("v3d"), Pack("p2z")
    Xd, okd = merkmale("v3d")
    Xp, okp = merkmale("p2z", "val")
    wd = np.load(os.path.join(DATEN, "p2z_wd.npz"))
    okp &= pp.ok & ~wd["gesperrt"]
    gr = {"Danbooru Anime 2026": np.flatnonzero(okd & pd.ok & (pd.split == "val"))}
    s = {"Danbooru Anime 2026": bewerte(np.asarray(Xd[gr["Danbooru Anime 2026"]]), k)}
    for q in ("wallhaven_general", "danbooru_nichtanime"):
        i = np.flatnonzero(okp & (pp.split == "val") & (pp.quelle == q))
        s[q] = bewerte(np.asarray(Xp[i]), k)
    t = {f: schwelle(list(s.values()), f) for f in (0.01, 0.05)}
    for g, v in s.items():
        print(f"  val {g:<22} n={len(v):>5}  FA bei 1-%-Schwelle {np.mean(v > t[0.01]):.2%}")
    alt = float(k[0]["schwelle_1"])
    print(f"{tag}: Schwelle 1 % {alt:+.2f} -> {t[0.01]:+.2f}, 5 % {float(k[0]['schwelle_5']):+.2f} -> {t[0.05]:+.2f}")
    for seed in range(len(k)):
        f = os.path.join(DATEN, f"kopf_{tag}_s{seed}.npz")
        d = dict(np.load(f, allow_pickle=True))
        d.setdefault("schwelle_1_alt", d["schwelle_1"])
        d.setdefault("schwelle_5_alt", d["schwelle_5"])
        d["schwelle_1"], d["schwelle_5"] = t[0.01], t[0.05]
        d["kalibrierung"] = "val ungestoert, Danbooru-Anime/Wallhaven/Danbooru-Nicht-Anime gleich gewichtet"
        np.savez(f, **d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
