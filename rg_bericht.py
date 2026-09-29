r"""rg_bericht.py — backbone comparison table (docs/de/RUECKGRATE.md) from gleich_fa_rg.json, bench_rg.json
and rg_tempo.json: seven heads trained identically, only the frozen backbones differ.

    python rg_bericht.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kanaele import ANIME, oeffentlich          # noqa: E402
from lesen import DATEN                          # noqa: E402

KOEPFE = [("rg_C", "CLIP", ["clip_mid"]), ("rg_D", "DINOv2", ["dino_mid"]), ("rg_S", "SigLIP 2", ["siglip_mid"]),
          ("rg_CD", "CLIP + DINOv2", ["clip_mid", "dino_mid"]), ("rg_CS", "CLIP + SigLIP 2", ["clip_mid", "siglip_mid"]),
          ("rg_DS", "DINOv2 + SigLIP 2", ["dino_mid", "siglip_mid"]),
          ("p7", "CLIP + DINOv2 + SigLIP 2", ["clip_mid", "dino_mid", "siglip_mid"])]


def main():
    gf = json.load(open(os.path.join(DATEN, "gleich_fa_rg.json"), encoding="utf-8"))
    bn = json.load(open(os.path.join(DATEN, "bench_rg.json"), encoding="utf-8"))
    tempo = json.load(open(os.path.join(DATEN, "rg_tempo.json"), encoding="utf-8"))
    e, b, art = gf["ergebnis"], bn["ergebnis"], bn["art"]
    gen = [g for g in art if art[g] == "k_gen"]
    kan = list(e["p7"]["treffer"])
    md = ["# Rückgrat-Vergleich", "",
          "Welche eingefrorenen Bildmodelle („Rückgrate“) tragen wie viel zur Erkennung bei? Sieben Köpfe, "
          "identisch trainiert (gleiche Daten, Gewichtung, Seeds 0–4, Kalibrierung); nur die Rückgrate "
          "unterscheiden sich. Merkmale je Rückgrat: fünf Zwischenschichten (CLIP ViT-L/14: Blöcke 7/11/15/19/23, "
          "CLS; DINOv2-L: Schichten 8/12/16/20/24, CLS; SigLIP 2 so400m/384: Schichten 7/12/17/22/27, Mittel der "
          "Patch-Token).", "",
          f"Kanäle: 15 Telegram-KI-Kanäle, nie im Training, Schwelle je Kopf bei {gf['fa']:g} % Fehlalarm im Mittel "
          "der drei menschlichen Testgruppen. Generatoren: Testsplits je Generator aus `bench.py` (Schwelle 1 % "
          "auf den val-Menschen). Tempo: Bilder/s im Scanner auf einer RTX 4090 (nur Merkmale, ohne Dateilesen).", "",
          "| Rückgrate | Kanäle Treffer | Anime-Kanäle | Kanäle AUC | Generatoren Treffer | Generatoren AUC | "
          "Fehlalarm Test | Tempo |", "|---|---|---|---|---|---|---|---|"]
    for tag, name, rg in KOEPFE:
        if tag not in e or tag not in b:
            md.append(f"| {name} | – | – | – | – | – | – | – |")
            continue
        t = np.mean([e[tag]["treffer"][k] for k in kan])
        ta = np.mean([e[tag]["treffer"][k] for k in ANIME if k in e[tag]["treffer"]])
        au = np.mean([e[tag]["auc"][k] for k in kan])
        tg = np.mean([b[tag]["treffer"][g] for g in gen])
        ag = np.mean([b[tag]["auc"][g] for g in gen])
        fa = b[tag]["fa"]["gemischt"]
        v = 1 / sum(1 / tempo[r] for r in rg)
        md.append(f"| {name} | {t:.1%} | {ta:.1%} | {au:.4f} | {tg:.1%} | {ag:.4f} | {fa:.1%} | {v:.0f}/s |")
    md += ["", "## Je Kanal (Treffer bei gleichem Fehlalarm)", "",
           "| Kanal | " + " | ".join(n for _, n, _ in KOEPFE) + " |", "|---|" + "---|" * len(KOEPFE)]
    for k in kan:
        md.append(f"| {k} | " + " | ".join(f"{e[t]['treffer'][k]:.1%}" if t in e else "–" for t, _, _ in KOEPFE) + " |")
    md += ["", "## Je Generator (Treffer, bench.py)", "",
           "| Generator | " + " | ".join(n for _, n, _ in KOEPFE) + " |", "|---|" + "---|" * len(KOEPFE)]
    for g in gen:
        md.append(f"| {g[2:]} | " + " | ".join(f"{b[t]['treffer'][g]:.1%}" if t in b else "–" for t, _, _ in KOEPFE) + " |")
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "de", "RUECKGRATE.md"), "w", encoding="utf-8").write(
        oeffentlich("\n".join(md) + "\n"))
    print("\n".join(md[:18]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
