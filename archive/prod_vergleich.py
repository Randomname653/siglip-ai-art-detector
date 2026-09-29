r"""prod_vergleich.py — v3 gegen die sieben Produktionsdetektoren, gleiche Bilder, gleiche Regel.

Fuer jeden Detektor (auch v3) wird die Schwelle so gesetzt, dass er auf
menschlicher Danbooru-Kunst von 2026 (val, Kuenstler getrennt) 1 % Fehlalarm hat.
Gemessen dann: Fehlalarm auf Danbooru-test, Treffer auf jedem KI-Kanal.
Die Produktion sieht genau die Dateien, die auch v3 sieht (Telegram-Fotos der
Kanaele; Danbooru in der Telegram-Simulation). Je Kanal hoechstens N Bilder
(fester Zufall), Scores werden je Detektor zwischengespeichert.

    python lab/detektor/prod_vergleich.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from pfade import DATASETS                       # noqa: E402
import messlatte                                  # noqa: E402
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from lesen import DATEN, Pack                    # noqa: E402

N = 3000
WURZEL = f"{DATASETS}/telegram_ki"


class Roh:
    def __init__(self, pfade):
        self.p = pfade

    def bild(self, i, aug=False):
        from PIL import Image
        with Image.open(self.p[i]) as im:
            return im.convert("RGB")


def main():
    rng = np.random.default_rng(26)
    pd = Pack("v3d")
    ok = np.load(os.path.join(DATEN, "emb_clip_mid_v3d_ok.npy")) & pd.ok
    dv, dt = np.flatnonzero(ok & (pd.split == "val")), np.flatnonzero(ok & (pd.split == "test"))
    gruppen = {"danbooru_val": [pd.m[i]["src"] for i in dv], "danbooru_test": [pd.m[i]["src"] for i in dt]}
    # v3-Merkmale: Danbooru aus den Embeddings, Kanaele aus den Kanal-Caches
    X = np.concatenate([np.load(os.path.join(DATEN, f"emb_{r}_v3d.npy"), mmap_mode="r")
                        for r in ("clip_mid", "dino_mid")], 1)
    k3 = koepfe_von("v3")
    v3 = {"danbooru_val": bewerte(np.asarray(X[dv]), k3), "danbooru_test": bewerte(np.asarray(X[dt]), k3)}
    for name in sorted(os.listdir(WURZEL)):
        c = os.path.join(DATEN, f"kanal_{name}_emb.npz")
        if not os.path.exists(c):
            continue
        d = np.load(c)
        wahl = np.sort(rng.choice(len(d["pfade"]), min(N, len(d["pfade"])), replace=False))
        gruppen[name] = [str(d["pfade"][i]) for i in wahl]
        v3[name] = bewerte(d["X"][wahl], k3)
    alle = [p for g in gruppen.values() for p in g]
    grenzen = np.cumsum([0] + [len(g) for g in gruppen.values()])
    print(f"{len(alle)} Bilder: " + ", ".join(f"{k} {len(v)}" for k, v in gruppen.items()), flush=True)

    scores = {"v3": np.concatenate(list(v3.values()))}
    roh = Roh(alle)
    for key, backend, name in messlatte.MODELLE:
        ziel = os.path.join(DATEN, f"prodvgl_{key.replace(':', '_')}.npy")
        if os.path.exists(ziel) and len(np.load(ziel)) == len(alle):
            scores[key] = np.load(ziel)
            continue
        f = messlatte.score_hf if backend == "hf" else messlatte.score_deepghs
        scores[key] = np.asarray(f(roh, np.arange(len(alle)), name))
        np.save(ziel, scores[key])
        print(f"  {key} fertig", flush=True)

    teil = {g: slice(grenzen[j], grenzen[j + 1]) for j, g in enumerate(gruppen)}
    erg = {}
    kanaele = [g for g in gruppen if not g.startswith("danbooru")]
    print(f"\n{'Detektor':<34}{'FA test':>8}" + "".join(f"{g[:11]:>12}" for g in kanaele))
    for det, s in scores.items():
        schw = np.quantile(s[teil["danbooru_val"]], 0.99)
        z = {"fa_test": float((s[teil["danbooru_test"]] > schw).mean())}
        z.update({g: float((s[teil[g]] > schw).mean()) for g in kanaele})
        erg[det] = z
        print(f"{det:<34}{z['fa_test']:>7.1%}" + "".join(f"{z[g]:>11.1%} " for g in kanaele), flush=True)
    with open(os.path.join(DATEN, "prod_vergleich.json"), "w", encoding="utf-8") as f:
        json.dump({"n": {g: len(v) for g, v in gruppen.items()}, "ergebnis": erg}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
