r"""vorversuch_schichten.py — dichtere Schichtauswahl im kleinen SigLIP 2 (so400m, 27 Schichten, 1152 dim).

Anlass: vorversuch_giant.py zeigte, dass das Signal im spaeten Mittelteil sitzt (so400m: Schicht 17-22 am besten,
Schicht 7 schwach). Hier: Schichten 10, 12, ..., 26 neu eingebettet und gegen den Standard S5 verglichen.

(Aufbau wie vorversuch_giant.py:

Gleicher Aufbau wie vorversuch_rg.py: 60.000 Trainings- und 8.000 val-Bilder (gestoert), menschliche
val/test-Gruppen (je 900), KI-Testsplits aller Packs (je bis 1.000), alle Telegram-Kanaele (je 400).
Eingebettet wird nur das grosse Modell, dafuer jede vierte Schicht (4, 8, ..., 40); SigLIP so400m liegt
fuer alle Bilder schon vor (Spalten in RG_ALLE). Koepfe (je 3 Seeds, identische Gewichte):
  S5          so400m, Schichten 7/12/17/22/27 (heutiger Standard p8sx)
  G5          giant, Schichten 8/16/24/32/40 (gleiche Verteilung)
  G10         giant, alle zehn Schichten
  S@k, G@k    je eine einzelne Schicht -> welche Tiefe traegt das Signal?
Bewertung: Schwelle bei 1 % Fehlalarm im Mittel der TEST-Menschen (gleicher Fehlalarm fuer alle Koepfe),
Treffer und AUC auf Kanaelen und KI-Testsplits.

    python lab/detektor/vorversuch_giant.py
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
import train3                                     # noqa: E402
from gleich_fa import auc, schwelle               # noqa: E402
from lesen import DATEN, Pack, kanal_X, spalten_fuer   # noqa: E402
from norm import normiere                         # noqa: E402
from train2 import Skalierer                      # noqa: E402
from train3 import Verbund, gewichte3, kopf, lade, nimm, rating_ausgleich   # noqa: E402

ZUSATZ = ["v3z2", "v3z3", "p2z", "p3z", "p4z", "p5z"]
MJ = ["midjourney_felix", "nijijourney_p1atdev", "midjourney_v6_scrape"]
GIANT = "google/siglip2-so400m-patch14-384"
G_SCHICHTEN = tuple(range(10, 27, 2))         # 10, 12, ..., 26
S_SCHICHTEN = (7, 12, 17, 22, 27)
CACHE = os.path.join(DATEN, "vorversuch_schichten.npz")


def giant_modell():
    import torch
    from transformers import SiglipVisionModel
    return SiglipVisionModel.from_pretrained(GIANT, torch_dtype=torch.float16).cuda().eval()


def giant_merkmale(model, bilder, ex):
    import torch
    import embed
    x = torch.from_numpy(np.stack(list(ex.map(embed.siglip_vorbereiten, bilder)))).cuda().half()
    with torch.no_grad():
        hs = model(pixel_values=x, output_hidden_states=True).hidden_states
    return torch.cat([hs[s].float().mean(1) for s in G_SCHICHTEN], 1).cpu().numpy()


def main():
    train3.RG = ("clip_mid", "dino_mid", "siglip_mid")
    rng = np.random.default_rng(31)
    t0 = time.time()
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
    sS = spalten_fuer(["siglip_mid"])

    def teil(liste):
        i2 = np.array([i for q, i in liste if q == "v2"])
        iz = np.array([i for q, i in liste if q == "z"])
        X = np.concatenate([nimm(A2, i2), nimm(Az, iz)])[:, sS]
        return (X, np.r_[p2.label[i2], pz.label[iz]], np.r_[p2.quelle[i2], pz.quelle[iz]],
                np.r_[wd2["rating"][i2], wdz["rating"][iz]])

    Str, ytr, qtr, rtr = teil(tr)
    Sva, yva, _, _ = teil(va)

    pd, pp = Pack("v3d"), Pack("p2z")
    from p2_test import merkmale
    Xd, okd = merkmale("v3d")
    gruppen = {}
    for split in ("val", "test"):
        i = np.sort(rng.choice(np.flatnonzero(okd & pd.ok & (pd.split == split)), 900, replace=False))
        gruppen[f"M {split} Danbooru-Anime"] = ("m_" + split, [(pd, j) for j in i], np.asarray(Xd[i])[:, sS])
        Xp, okp = merkmale("p2z", split)
        okp = okp & pp.ok & (pp.split == split)
        for qn in ("wallhaven_general", "danbooru_nichtanime"):
            i = np.sort(rng.choice(np.flatnonzero(okp & (pp.quelle == qn)), 900, replace=False))
            gruppen[f"M {split} {qn}"] = ("m_" + split, [(pp, j) for j in i], np.asarray(Xp[i])[:, sS])
    for st in ["v2"] + ZUSATZ:
        p = Pack(st)
        X, ok = merkmale(st, "test")
        ok = ok & p.ok & (p.split == "test") & (p.label == 1)
        i = np.sort(rng.choice(np.flatnonzero(ok), min(1000, int(ok.sum())), replace=False))
        gruppen[f"K {st}"] = ("k_gen", [(p, j) for j in i], np.asarray(X[i])[:, sS])
    for f in sorted(os.listdir(DATEN)):
        if f.startswith("kanal_") and f.endswith("_emb.npz") and os.path.isdir(
                os.path.join(f"{DATASETS}/telegram_ki", f[6:-8])):
            Xk, pf = kanal_X(f[6:-8])
            i = np.sort(rng.choice(len(pf), min(400, len(pf)), replace=False))
            gruppen[f"T {f[6:-8]}"] = ("k_kanal", [(None, str(pf[j])) for j in i], Xk[i][:, sS])
    print(f"Auswahl fertig ({time.time()-t0:.0f}s): train {len(Str)}, val {len(Sva)}, {len(gruppen)} Gruppen", flush=True)

    if os.path.exists(CACHE):
        c = np.load(CACHE)
        G_tr, G_va, G_te = c["tr"], c["va"], c["te"]
    else:
        model = giant_modell()
        ex = ThreadPoolExecutor(12)
        offs = np.cumsum([0] + [len(x.label) for x in pz.packs])

        def pack_bild(q, i, aug):
            if q == "v2":
                return p2.bild(i, aug=aug)
            k = int(np.searchsorted(offs, i, side="right") - 1)
            return pz.packs[k].bild(i - offs[k], aug=aug)

        def einbetten(quelle, n, aug):
            out = np.zeros((n, 1152 * len(G_SCHICHTEN)), np.float16)
            for s in range(0, n, 32):
                bilder = list(ex.map(lambda k: quelle(k, aug), range(s, min(n, s + 32))))
                out[s:s + len(bilder)] = giant_merkmale(model, bilder, ex)
                if s % 6400 == 0:
                    print(f"    {s}/{n}  {(time.time()-t0)/60:.0f} min", flush=True)
            return out
        G_tr = einbetten(lambda k, aug: pack_bild(tr[k][0], tr[k][1], aug), len(tr), True)
        G_va = einbetten(lambda k, aug: pack_bild(va[k][0], va[k][1], aug), len(va), True)
        te = [e for g in gruppen.values() for e in g[1]]

        def te_bild(k, aug):
            p, j = te[k]
            if p is None:
                return Image.open(io.BytesIO(normiere(j)[0])).convert("RGB")
            return p.bild(j)
        G_te = einbetten(te_bild, len(te), False)
        np.savez(CACHE, tr=G_tr, va=G_va, te=G_te)
        print(f"  Schichten eingebettet ({(time.time()-t0)/60:.0f} min)", flush=True)

    qg = qtr.copy()
    qg[np.isin(qg, MJ)] = "midjourney"
    w = rating_ausgleich(gewichte3(ytr, qg), ytr, rtr)
    grenzen = np.cumsum([0] + [len(g[1]) for g in gruppen.values()])
    S_te = np.concatenate([np.asarray(g[2], np.float16) for g in gruppen.values()])

    def blk(n, k, dim):
        return slice(k * dim, (k + 1) * dim)
    D = 1152
    def spl(schichten):
        return np.concatenate([np.arange(G_SCHICHTEN.index(x) * D, (G_SCHICHTEN.index(x) + 1) * D) for x in schichten])
    varianten = {"S5 heute 7/12/17/22/27": ("S", slice(None)),
                 "D9 dicht 10..26": ("G", slice(None)),
                 "D7 14..26": ("G", spl((14, 16, 18, 20, 22, 24, 26))),
                 "D5 14/18/20/22/26": ("G", spl((14, 18, 20, 22, 26))),
                 "D5b 16/18/20/22/24": ("G", spl((16, 18, 20, 22, 24))),
                 "D4 18/20/22/24": ("G", spl((18, 20, 22, 24)))}
    for s_ in G_SCHICHTEN:
        varianten[f"D@{s_:02d}"] = ("G", spl((s_,)))
    erg = {}
    for name, (q, sp) in varianten.items():
        Ftr, Fva, Fte = (Str, Sva, S_te) if q == "S" else (G_tr, G_va, G_te)
        Ftr, Fva, Fte = Ftr[:, sp], Fva[:, sp], Fte[:, sp]
        sc = Skalierer().fit(Ftr, np.arange(len(Ftr)))
        a, b, c = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)
        s = np.mean([kopf(a, ytr, w, b, yva, seed)[0](c) for seed in range(3)], 0)
        teil_ = {g: s[grenzen[j]:grenzen[j + 1]] for j, g in enumerate(gruppen)}
        tt = schwelle([teil_[g] for g, v in gruppen.items() if v[0] == "m_test"], 0.01)
        mh = np.concatenate([teil_[g] for g, v in gruppen.items() if v[0] == "m_test"])
        kan = [g for g, v in gruppen.items() if v[0] == "k_kanal"]
        gen = [g for g, v in gruppen.items() if v[0] == "k_gen"]
        erg[name] = {"kanaele": float(np.mean([np.mean(teil_[g] > tt) for g in kan])),
                     "kanaele_auc": float(np.mean([auc(mh, teil_[g]) for g in kan])),
                     "generatoren": float(np.mean([np.mean(teil_[g] > tt) for g in gen])),
                     "generatoren_auc": float(np.mean([auc(mh, teil_[g]) for g in gen])),
                     "je_kanal": {g: float(np.mean(teil_[g] > tt)) for g in kan}}
        e = erg[name]
        print(f"  {name:<26} Kanaele {e['kanaele']:6.1%}  AUC {e['kanaele_auc']:.4f}   "
              f"Generatoren {e['generatoren']:6.1%}  AUC {e['generatoren_auc']:.4f}   ({(time.time()-t0)/60:.0f} min)", flush=True)
    json.dump(erg, open(os.path.join(DATEN, "vorversuch_schichten.json"), "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
