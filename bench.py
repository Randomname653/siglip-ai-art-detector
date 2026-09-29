r"""bench.py — production detectors vs. our heads on one fixed image set, mixed and per generator.

Image set (fixed random seed, at most N images per group, never in our heads' training):
  humans val    Danbooru anime 2026, Wallhaven general <2022, Danbooru non-anime  -> threshold only
  humans test   the same three groups, other artists/images                       -> false positives
  AI per generator: CivitAI test split per base model (creators unseen, base model seen), AIBooru per model
    tag, own generators (v2 test split), CivitAI semi-realistic/3D, DiffusionDB, Midjourney/Niji test splits
  AI per Telegram channel (never in training, mixed generators)

Same rule for ALL detectors: threshold where the mean false-positive rate over the human val groups is 1 %.
Reported: false positives per human test group, hit rate and AUC per AI group, and "mixed" averages.
Production = the seven individual models of the existing pipeline (messlatte.MODELLE); their tier logic and
meta model are not reproduced. Production scores are cached per group (bench_cache/), so adding groups or heads
is cheap. gruppen_bauen() is reused by kombination.py.

    python bench.py v3 p8x p8sx [--n 300] [--json out.json]
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
import messlatte                                  # noqa: E402
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from gleich_fa import auc, schwelle              # noqa: E402
from lesen import DATEN, Pack, kanal_X           # noqa: E402
from p2_test import merkmale                     # noqa: E402

KANAELE = f"{DATASETS}/telegram_ki"
AIBOORU = {"novelai": "AIBooru NovelAI", "nijijourney": "AIBooru Nijijourney", "dall-e": "AIBooru DALL-E"}


class Quelle:
    """Einheitlicher Bildzugriff fuer messlatte: Eintraege (pack, index) oder Dateipfad."""
    def __init__(self, eintraege):
        self.e = eintraege

    def bild(self, i, aug=False):
        from PIL import Image
        p, j = self.e[i]
        if p is None:
            try:
                with Image.open(j) as im:
                    return im.convert("RGB")
            except OSError:
                return Image.new("RGB", (256, 256), (128, 128, 128))   # defekt; wird unten ausgeschlossen
        return p.bild(j)


def defekte(alle):
    """Indizes der Kanaldateien, die sich nicht lesen lassen (abgebrochene Downloads)."""
    from PIL import Image
    kaputt = set()
    for i, (p, j) in enumerate(alle):
        if p is None:
            try:
                with Image.open(j) as im:
                    im.load()
            except OSError:
                kaputt.add(i)
    return kaputt


def gruppen_bauen(n_gr=300, n_kanal=600):
    """Der feste Bildersatz: Gruppenname -> (Art, Eintraege, Merkmale). Gleicher Zufall = gleiche Bilder,
    damit die zwischengespeicherten Produktions-Scores passen (auch fuer kombination.py)."""
    rng = np.random.default_rng(27)

    def wahl(idx, n):
        idx = np.asarray(idx)
        return np.sort(rng.choice(idx, min(n, len(idx)), replace=False)) if len(idx) > n else idx

    gruppen = collections.OrderedDict()   # name -> (art, eintraege, X)
    pd = Pack("v3d")
    Xd, okd = merkmale("v3d")
    pp = Pack("p2z")
    wd = np.load(os.path.join(DATEN, "p2z_wd.npz"))
    for split in ("val", "test"):
        i = wahl(np.flatnonzero(okd & pd.ok & (pd.split == split)), 3 * n_gr)
        gruppen[f"M {split} Danbooru-Anime 2026"] = ("m_" + split, [(pd, j) for j in i], Xd[i])
        Xp, okp = merkmale("p2z", split)
        okp = okp & pp.ok & ~wd["gesperrt"] & (pp.split == split)
        for q, n in (("wallhaven_general", "Wallhaven general <2022"), ("danbooru_nichtanime", "Danbooru-Nicht-Anime")):
            i = wahl(np.flatnonzero(okp & (pp.quelle == q)), 3 * n_gr)
            gruppen[f"M {split} {n}"] = ("m_" + split, [(pp, j) for j in i], Xp[i])
        if split == "test":
            i = wahl(np.flatnonzero(okp & (pp.quelle == "civitai_stil")), n_gr)
            gruppen["K CivitAI halbrealistisch/3D"] = ("k_gen", [(pp, j) for j in i], Xp[i])

    for st in ("v3z2", "v3z3", "v2", "p3z", "p4z", "p5z"):
        if not os.path.exists(os.path.join(DATEN, f"emb_clip_mid_{st}_test.npy")):
            continue
        p = Pack(st)
        X, ok = merkmale(st, "test")
        wst = os.path.join(DATEN, f"{st}_wd.npz")
        ok = ok & p.ok & (p.split == "test") & (p.label == 1)
        if os.path.exists(wst):
            ok &= ~np.load(wst)["gesperrt"]
        if st in ("v3z2", "v3z3"):
            for mod in sorted(set(p.model[ok & (p.quelle == "civitai")])):
                i = np.flatnonzero(ok & (p.quelle == "civitai") & (p.model == mod))
                name = f"K CivitAI {mod}"
                alt = gruppen.get(name)
                eintr = ([] if alt is None else alt[1]) + [(p, j) for j in i]
                Xs = X[i] if alt is None else np.concatenate([alt[2], X[i]])
                gruppen[name] = ("k_gen", eintr, Xs)
            if st == "v3z2":
                km = np.array([str(p.m[j].get("ki_modell", "")) for j in range(len(p.m))])
                ab = ok & (p.quelle == "aibooru")
                rest = ab.copy()
                for tag, name in AIBOORU.items():
                    m = ab & np.array([tag in k for k in km])
                    rest &= ~m
                    i = wahl(np.flatnonzero(m), n_gr)
                    gruppen["K " + name] = ("k_gen", [(p, j) for j in i], X[i])
                i = wahl(np.flatnonzero(rest), n_gr)
                gruppen["K AIBooru andere/unbekannt"] = ("k_gen", [(p, j) for j in i], X[i])
        elif st == "v2":
            for mod in sorted(set(p.model[ok & (p.quelle == "generiert")])):
                i = wahl(np.flatnonzero(ok & (p.quelle == "generiert") & (p.model == mod)), n_gr)
                gruppen[f"K eigen {mod}"] = ("k_gen", [(p, j) for j in i], X[i])
        elif st == "p3z":
            i = wahl(np.flatnonzero(ok), n_gr)
            gruppen["K DiffusionDB SD 1.x 2022"] = ("k_gen", [(p, j) for j in i], X[i])
        else:   # p4z/p5z: Midjourney Felix, Nijijourney v5, Midjourney v6
            for mod in sorted(set(p.model[ok])):
                i = wahl(np.flatnonzero(ok & (p.model == mod)), n_gr)
                gruppen[f"K {mod}"] = ("k_gen", [(p, j) for j in i], X[i])
    # CivitAI-Gruppen erst jetzt kappen (v3z2 + v3z3 zusammengefuehrt); zu kleine weglassen
    for name in [g for g in gruppen if g.startswith("K CivitAI ") and "halbreal" not in g]:
        art, e, X = gruppen[name]
        if len(e) < 40:
            del gruppen[name]
            continue
        k = wahl(np.arange(len(e)), n_gr)
        gruppen[name] = (art, [e[j] for j in k], X[k])
    for f in sorted(os.listdir(DATEN)):
        if f.startswith("kanal_") and f.endswith("_emb.npz"):
            name = f[6:-8]
            if not os.path.isdir(os.path.join(KANAELE, name)):
                continue
            Xk, pfade = kanal_X(name)
            i = wahl(np.arange(len(pfade)), n_kanal)
            gruppen["T " + name] = ("k_kanal", [(None, str(pfade[j])) for j in i], Xk[i])

    return gruppen


def schluessel(eintr):
    return hashlib.sha1("|".join(f"{'' if p is None else p.stamm}:{j}" for p, j in eintr).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("koepfe", nargs="+")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-kanal", type=int, default=600)
    ap.add_argument("--plan", action="store_true", help="nur Gruppen auflisten")
    ap.add_argument("--json", default="bench.json", help="Ausgabedatei in daten/")
    a = ap.parse_args()
    gruppen = gruppen_bauen(a.n, a.n_kanal)
    alle = [e for g in gruppen.values() for e in g[1]]
    grenzen = np.cumsum([0] + [len(g[1]) for g in gruppen.values()])
    kaputt = defekte(alle)
    teil = {g: np.array([i for i in range(grenzen[j], grenzen[j + 1]) if i not in kaputt], int)
            for j, g in enumerate(gruppen)}
    print(f"{len(alle)} Bilder in {len(gruppen)} Gruppen, {len(kaputt)} defekte Kanaldateien ausgeschlossen", flush=True)
    for g, (art, e, _) in gruppen.items():
        print(f"  {g:<40} {art:<8} {len(e):>5}")
    if a.plan:
        return 0

    scores = {}
    for tag in a.koepfe:
        k = koepfe_von(tag)
        scores[tag] = np.concatenate([bewerte(np.asarray(X, np.float32), k) for _, _, X in gruppen.values()])
    # Produktions-Scores je GRUPPE zwischenspeichern (Schluessel = Inhalt der Gruppe), damit neue
    # Gruppen (z. B. Midjourney-Testsplits) nur sich selbst kosten. Der erste Lauf speicherte je
    # Detektor eine Datei fuer den ganzen Satz (kennung) — die wird hier einmalig aufgeteilt.
    kennung = schluessel(alle)[:10]
    gkey = {g: schluessel(v[1]) for g, v in gruppen.items()}
    q = Quelle(alle)
    for key, backend, name in messlatte.MODELLE:
        det = key.replace(":", "_")
        cache = os.path.join(DATEN, "bench_cache")
        os.makedirs(cache, exist_ok=True)
        alt = os.path.join(DATEN, f"bench_{det}_{kennung}.npy")
        if os.path.exists(alt):
            s_alt = np.load(alt)
            for j, g in enumerate(gruppen):
                np.save(os.path.join(cache, f"{det}_{gkey[g]}.npy"), s_alt[grenzen[j]:grenzen[j + 1]])
        s = np.full(len(alle), np.nan, np.float32)
        fehlt = []
        for j, g in enumerate(gruppen):
            f = os.path.join(cache, f"{det}_{gkey[g]}.npy")
            if os.path.exists(f):
                s[grenzen[j]:grenzen[j + 1]] = np.load(f)
            else:
                fehlt.append(j)
        if fehlt:
            idx = np.concatenate([np.arange(grenzen[j], grenzen[j + 1]) for j in fehlt])
            fn = messlatte.score_hf if backend == "hf" else messlatte.score_deepghs
            s[idx] = np.asarray(fn(q, idx, name), np.float32)
            for j in fehlt:
                np.save(os.path.join(cache, f"{det}_{gkey[list(gruppen)[j]]}.npy"), s[grenzen[j]:grenzen[j + 1]])
            print(f"  {key}: {len(fehlt)} Gruppen neu bewertet ({len(idx)} Bilder)", flush=True)
        scores[key] = s

    mval = [g for g, v in gruppen.items() if v[0] == "m_val"]
    mtest = [g for g, v in gruppen.items() if v[0] == "m_test"]
    ki = [g for g, v in gruppen.items() if v[0].startswith("k_")]
    erg = {}
    for det, s in scores.items():
        # Schwelle: Mittel-FA 1 % auf den val-Menschen (Bisektion; fuer Wahrscheinlichkeiten in [0,1]
        # ueber Logit, damit die Bisektion denselben Bereich nutzt)
        z = s if s.min() < 0 or s.max() > 1 else np.log(np.clip(s, 1e-7, 1 - 1e-7) / np.clip(1 - s, 1e-7, 1))
        t = schwelle([z[teil[g]] for g in mval], 0.01)
        mh = np.concatenate([z[teil[g]] for g in mtest])
        r = {"schwelle": float(t), "fa": {g: float(np.mean(z[teil[g]] > t)) for g in mtest},
             "treffer": {g: float(np.mean(z[teil[g]] > t)) for g in ki},
             "auc": {g: float(auc(mh, z[teil[g]])) for g in ki}}
        gen = [g for g in ki if gruppen[g][0] == "k_gen"]
        kan = [g for g in ki if gruppen[g][0] == "k_kanal"]
        for name, gs in (("gemischt Generatoren", gen), ("gemischt Kanaele", kan), ("gemischt alle", ki)):
            r["treffer"][name] = float(np.mean([r["treffer"][g] for g in gs]))
            # gleich gewichtete AUC: Mittel der Gruppen-AUCs
            r["auc"][name] = float(np.mean([r["auc"][g] for g in gs]))
        r["fa"]["gemischt"] = float(np.mean(list(r["fa"].values())))
        erg[det] = r

    dets = list(scores)
    kurz = {d: d.split(":")[-1][:12] for d in dets}
    print(f"\n{'Treffer bei 1 % FA (val)':<40}" + "".join(f"{kurz[d]:>13}" for d in dets))
    for g in mtest:
        print(f"{'FA ' + g[7:]:<40}" + "".join(f"{erg[d]['fa'][g]:>12.1%} " for d in dets))
    print(f"{'FA gemischt':<40}" + "".join(f"{erg[d]['fa']['gemischt']:>12.1%} " for d in dets))
    for g in ki + ["gemischt Generatoren", "gemischt Kanaele", "gemischt alle"]:
        print(f"{g[:40]:<40}" + "".join(f"{erg[d]['treffer'][g]:>12.1%} " for d in dets))
    print(f"\n{'AUC (gegen alle Test-Menschen)':<40}" + "".join(f"{kurz[d]:>13}" for d in dets))
    for g in ["gemischt Generatoren", "gemischt Kanaele", "gemischt alle"]:
        print(f"{g:<40}" + "".join(f"{erg[d]['auc'][g]:>13.4f}" for d in dets))
    with open(os.path.join(DATEN, a.json), "w", encoding="utf-8") as f:
        json.dump({"n": {g: len(v[1]) for g, v in gruppen.items()}, "art": {g: v[0] for g, v in gruppen.items()},
                   "ergebnis": erg}, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
