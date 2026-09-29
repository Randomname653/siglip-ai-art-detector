r"""vorversuch_rg.py — pilot: is a third backbone (SigLIP 2) worth re-embedding everything?

Same subset (60,000 training / 8,000 val images, perturbed), same seeds, two heads: CLIP+DINO vs.
CLIP+DINO+SigLIP; evaluated on human groups, AI test splits and all Telegram channels. Answer: yes
(channel AUC 0.9920 -> 0.9963), which led to p7x/p8x.

    python vorversuch_rg.py
"""
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
import embed                                      # noqa: E402
from gleich_fa import auc, schwelle               # noqa: E402
from lesen import DATEN, Pack                     # noqa: E402
from norm import normiere                         # noqa: E402
from train2 import Skalierer                      # noqa: E402
from train3 import Verbund, gewichte3, kopf, lade, nimm, rating_ausgleich   # noqa: E402

ZUSATZ = ["v3z2", "v3z3", "p2z", "p3z", "p4z", "p5z"]
MJ = ["midjourney_felix", "nijijourney_p1atdev", "midjourney_v6_scrape"]
CACHE = os.path.join(DATEN, "vorversuch_siglip.npz")


def main():
    rng = np.random.default_rng(31)
    t0 = time.time()
    # --- Auswahl ------------------------------------------------------------------------------
    p2, pz = Pack("v2"), Verbund(ZUSATZ)
    A2, ok2 = lade("v2", True)
    Az, okz = pz.lade(True)
    wd2, wdz = np.load(os.path.join(DATEN, "v2_wd.npz")), pz.wd()
    ok2 &= p2.ok & ~wd2["gesperrt"]
    okz &= pz.ok & ~wdz["gesperrt"]
    alle_tr = [("v2", i) for i in np.flatnonzero(ok2 & (p2.split == "train"))] + \
              [("z", i) for i in np.flatnonzero(okz & (pz.split == "train"))]
    alle_va = [("v2", i) for i in np.flatnonzero(ok2 & (p2.split == "val"))] + \
              [("z", i) for i in np.flatnonzero(okz & (pz.split == "val"))]
    tr = [alle_tr[i] for i in np.sort(rng.choice(len(alle_tr), 60000, replace=False))]
    va = [alle_va[i] for i in np.sort(rng.choice(len(alle_va), 8000, replace=False))]

    def teil(liste, A_2, A_z):
        i2 = np.array([i for q, i in liste if q == "v2"])
        iz = np.array([i for q, i in liste if q == "z"])
        X = np.concatenate([nimm(A_2, i2), nimm(A_z, iz)])
        y = np.r_[p2.label[i2], pz.label[iz]]
        q = np.r_[p2.quelle[i2], pz.quelle[iz]]
        r = np.r_[wd2["rating"][i2], wdz["rating"][iz]]
        return X, y, q, r, i2, iz

    Xtr, ytr, qtr, rtr, tr2, trz = teil(tr, A2, Az)
    Xva, yva, _, _, va2, vaz = teil(va, A2, Az)

    # Test: Menschen (val fuer die Schwelle, test fuer den Fehlalarm), KI-Testsplits, Kanaele
    pd, pp = Pack("v3d"), Pack("p2z")
    from p2_test import merkmale
    Xd, okd = merkmale("v3d")
    gruppen = {}      # name -> (art, [(pack|None, idx|pfad)], X_alt)
    for split in ("val", "test"):
        i = rng.choice(np.flatnonzero(okd & pd.ok & (pd.split == split)), 900, replace=False)
        gruppen[f"M {split} Danbooru-Anime"] = ("m_" + split, [(pd, j) for j in np.sort(i)], np.asarray(Xd[np.sort(i)]))
        Xp, okp = merkmale("p2z", split)
        okp = okp & pp.ok & (pp.split == split)
        for qn in ("wallhaven_general", "danbooru_nichtanime"):
            i = np.sort(rng.choice(np.flatnonzero(okp & (pp.quelle == qn)), 900, replace=False))
            gruppen[f"M {split} {qn}"] = ("m_" + split, [(pp, j) for j in i], np.asarray(Xp[i]))
    for st in ["v2"] + ZUSATZ:
        p = Pack(st)
        X, ok = merkmale(st, "test")
        ok = ok & p.ok & (p.split == "test") & (p.label == 1)
        i = np.sort(rng.choice(np.flatnonzero(ok), min(1000, int(ok.sum())), replace=False))
        gruppen[f"K {st}"] = ("k_gen", [(p, j) for j in i], np.asarray(X[i]))
    for f in sorted(os.listdir(DATEN)):
        if f.startswith("kanal_") and f.endswith("_emb.npz") and os.path.isdir(
                os.path.join(f"{DATASETS}/telegram_ki", f[6:-8])):
            d = np.load(os.path.join(DATEN, f))
            i = np.sort(rng.choice(len(d["pfade"]), min(400, len(d["pfade"])), replace=False))
            gruppen[f"T {f[6:-8]}"] = ("k_kanal", [(None, str(d["pfade"][j])) for j in i], d["X"][i])
    print(f"Auswahl fertig ({time.time()-t0:.0f}s): train {len(Xtr)}, val {len(Xva)}, "
          f"{len(gruppen)} Testgruppen", flush=True)

    # --- SigLIP einbetten (zwischengespeichert) -----------------------------------------------
    if os.path.exists(CACHE):
        c = np.load(CACHE)
        S_tr, S_va, S_te = c["tr"], c["va"], c["te"]
    else:
        model, proc = embed.siglip_modell()
        ex = ThreadPoolExecutor(12)

        def einbetten(bildquelle, n, aug):
            out = np.zeros((n, 1152 * len(embed.SIGLIP_MID_SCHICHTEN)), np.float16)
            for s in range(0, n, 64):
                bilder = list(ex.map(lambda k: bildquelle(k, aug), range(s, min(n, s + 64))))
                out[s:s + len(bilder)] = embed.siglip_mid_merkmale(model, proc, bilder)
            return out

        offs = np.cumsum([0] + [len(x.label) for x in pz.packs])

        def pack_bild(q, i, aug):
            if q == "v2":
                return p2.bild(i, aug=aug)
            k = int(np.searchsorted(offs, i, side="right") - 1)     # globaler Verbund-Index -> Pack
            return pz.packs[k].bild(i - offs[k], aug=aug)

        def aus_pack(liste):
            return lambda k, aug: pack_bild(liste[k][0], liste[k][1], aug)
        S_tr = einbetten(aus_pack(tr), len(tr), True)
        print(f"  SigLIP train fertig ({time.time()-t0:.0f}s)", flush=True)
        S_va = einbetten(aus_pack(va), len(va), True)
        te = [e for g in gruppen.values() for e in g[1]]

        def te_bild(k, aug):
            p, j = te[k]
            if p is None:
                return Image.open(io.BytesIO(normiere(j)[0])).convert("RGB")
            return p.bild(j)
        S_te = einbetten(te_bild, len(te), False)
        np.savez(CACHE, tr=S_tr, va=S_va, te=S_te)
        print(f"  SigLIP fertig ({time.time()-t0:.0f}s)", flush=True)

    # --- Koepfe A und B -----------------------------------------------------------------------
    qg = qtr.copy()
    qg[np.isin(qg, MJ)] = "midjourney"
    w = rating_ausgleich(gewichte3(ytr, qg), ytr, rtr)
    grenzen = np.cumsum([0] + [len(g[1]) for g in gruppen.values()])
    X_te_alt = np.concatenate([np.asarray(g[2], np.float32) for g in gruppen.values()])
    erg = {}
    for name, (Ftr, Fva, Fte) in {
            "A CLIP+DINO": (Xtr, Xva, X_te_alt),
            "B CLIP+DINO+SigLIP": (np.concatenate([Xtr, S_tr], 1), np.concatenate([Xva, S_va], 1),
                                   np.concatenate([X_te_alt, S_te.astype(np.float32)], 1))}.items():
        sc = Skalierer().fit(Ftr, np.arange(len(Ftr)))
        a, b, c = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)
        s = np.mean([kopf(a, ytr, w, b, yva, seed)[0](c) for seed in range(3)], 0)
        teil = {g: s[grenzen[j]:grenzen[j + 1]] for j, g in enumerate(gruppen)}
        t = schwelle([teil[g] for g, v in gruppen.items() if v[0] == "m_val"], 0.01)
        mh = np.concatenate([teil[g] for g, v in gruppen.items() if v[0] == "m_test"])
        # zusaetzlich: Schwelle bei 1 % Fehlalarm im Mittel der TEST-Menschen (gleicher Fehlalarm fuer A und B)
        tt = schwelle([teil[g] for g, v in gruppen.items() if v[0] == "m_test"], 0.01)
        erg[name] = {g: {"rate": float(np.mean(teil[g] > t)), "rate_gleich": float(np.mean(teil[g] > tt)),
                         "auc": float(auc(mh, teil[g])) if v[0].startswith("k_") else None}
                     for g, v in gruppen.items() if v[0] != "m_val"}
        print(f"  {name} fertig ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'Gruppe':<32}{'A':>9}{'B':>9}{'AUC A':>9}{'AUC B':>9}")
    for g in erg["A CLIP+DINO"]:
        a, b = erg["A CLIP+DINO"][g], erg["B CLIP+DINO+SigLIP"][g]
        au = f"{a['auc']:>9.4f}{b['auc']:>9.4f}" if a["auc"] is not None else ""
        print(f"{g:<32}{a['rate']:>8.1%} {b['rate']:>8.1%}{au}")
    for art, lab in (("k_gen", "Mittel KI-Testsplits"), ("k_kanal", "Mittel Kanaele")):
        gs = [g for g, v in gruppen.items() if v[0] == art]
        print(f"{lab:<32}" + "".join(f"{np.mean([erg[k][g]['rate'] for g in gs]):>8.1%} " for k in erg)
              + "".join(f"{np.mean([erg[k][g]['auc'] for g in gs]):>9.4f}" for k in erg))
    json.dump(erg, open(os.path.join(DATEN, "vorversuch_rg.json"), "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
