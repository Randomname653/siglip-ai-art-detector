r"""alt_vs_p8sx.py — old production ensemble vs. p8sx on the SAME images of a scanned folder, from the score cache.

Images posted before the AI era (Telegram date in the file name < 2022-08-01) are provably human: every flag there
is a false positive. Newer posts are a mix; their flag rate is a rough proxy for hit rate. The old decision rule
(3 models majority -> candidates -> 7 models with veto) is replayed exactly from imagesort.py.

    python alt_vs_p8sx.py [--ordner "//NAS/.../Sorted Stuff"]
"""
import argparse
import collections
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import imagesort as I   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ordner", required=True, help="gescannter Ordner (Pfad wie im Cache)")
a = ap.parse_args()
c = sqlite3.connect(f"file:{I.CACHE_DB_PATH}?mode=ro", uri=True)
sc = collections.defaultdict(dict)
for p, m, s in c.execute("SELECT path, model_name, score FROM ai_scores WHERE path LIKE ?", (a.ordner + "%",)):
    if "_ai_quarantine" not in p:
        sc[p][m] = s
t1 = [k for k, _, _ in I.AI_MODELS_T1_ALT]
alle = t1 + [k for k, _, _ in I.AI_MODELS_T2_ALT]
pfade = [p for p, d in sc.items() if "sig2:p8sx_v1" in d and all(k in d for k in t1)]
print(f"{len(pfade)} Bilder mit alten Tier-1-Scores und p8sx")

I.AI_P8SX = False          # alte Regel nachspielen
alt = {}
for p in pfade:
    d = sc[p]
    cat1, _, _ = I.ai_ensemble_decision(False, {k: d[k] for k in t1})
    if cat1 == "none":
        alt[p] = "none"
        continue
    cat, _, _, _ = I.ai_ensemble_decision_full(False, {k: d[k] for k in alle if k in d})
    alt[p] = cat
I.AI_P8SX = True

datum = {p: I.telegram_post_date(p) for p in pfade}
gruppen = {"vor 08/2022 (menschlich)": [p for p in pfade if datum[p] and datum[p] < I.AI_ERA_CUTOFF],
           "2022-08 bis 2024": [p for p in pfade if datum[p] and I.AI_ERA_CUTOFF <= datum[p] < (2025, 1, 1)],
           "2025-2026": [p for p in pfade if datum[p] and datum[p] >= (2025, 1, 1)],
           "ohne Datum": [p for p in pfade if not datum[p]]}


def quote(ps, f):
    return sum(map(f, ps)) / max(1, len(ps))


print(f"\n{'Gruppe':<28}{'n':>7}  | alt: Review  alt: auto(def_ai/high) | p8sx: Review(>=0.5)  auto(>=0.9999)")
for g, ps in gruppen.items():
    print(f"{g:<28}{len(ps):>7}  |   {quote(ps, lambda p: alt[p] != 'none'):7.2%}        {quote(ps, lambda p: alt[p] in ('def_ai', 'high')):7.2%}"
          f"        |      {quote(ps, lambda p: sc[p]['sig2:p8sx_v1'] >= 0.5):7.2%}          {quote(ps, lambda p: sc[p]['sig2:p8sx_v1'] >= 0.9999):7.2%}")
print("alte Kategorien (menschlich):", collections.Counter(alt[p] for p in gruppen["vor 08/2022 (menschlich)"]))

# p8sx bei GLEICHER Fehlalarmquote wie das alte System: Schwelle so, dass auf den Menschen gleich viele markiert sind
hum = gruppen["vor 08/2022 (menschlich)"]
neu = gruppen["2025-2026"]
fa_alt = quote(hum, lambda p: alt[p] != "none")
s_h = np.array([sc[p]["sig2:p8sx_v1"] for p in hum])
s_n = np.array([sc[p]["sig2:p8sx_v1"] for p in neu])
for fa in (fa_alt, 0.005, 0.0025, 0.001):
    k = max(1, int(round(fa * len(s_h))))
    thr = np.sort(s_h)[::-1][k - 1] if k <= len(s_h) else 1.0
    thr = np.nextafter(thr, 2)       # strikt darueber
    print(f"p8sx mit Schwelle fuer {fa:.2%} Fehlalarm (Score > {thr:.6f}): markiert {np.mean(s_n > thr):.1%} der "
          f"2025/26-Posts   (altes System: {quote(neu, lambda p: alt[p] != 'none'):.1%})")

# Wie in der Pipeline: erkannte Fotos (Foto-Gate, Ergebnis im Cache real_photo) fallen vor dem Review heraus
foto = dict(c.execute("SELECT path, is_real FROM real_photo WHERE path LIKE ?", (a.ordner + "%",)).fetchall())


def pipeline(ps, flag, auto):
    roh = [p for p in ps if flag(p)]
    nach = [p for p in roh if not foto.get(p)]
    return {"n": len(ps), "roh": len(roh) / len(ps), "nach_foto_gate": len(nach) / len(ps),
            "auto": sum(1 for p in nach if auto(p)) / len(ps)}


ergebnis = {
    "gleiche_bilder": {
        "alt": pipeline(hum, lambda p: alt[p] != "none", lambda p: alt[p] in ("def_ai", "high")),
        "p8sx": pipeline(hum, lambda p: sc[p]["sig2:p8sx_v1"] >= 0.5, lambda p: sc[p]["sig2:p8sx_v1"] >= 0.9999)},
    "neu_2025_26_markiert": {"alt": quote(neu, lambda p: alt[p] != "none"),
                             "p8sx": quote(neu, lambda p: sc[p]["sig2:p8sx_v1"] >= 0.5)}}
print(ergebnis)
import json   # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import DATEN   # noqa: E402
json.dump(ergebnis, open(os.path.join(DATEN, "alt_vs_p8sx.json"), "w"), indent=1)
