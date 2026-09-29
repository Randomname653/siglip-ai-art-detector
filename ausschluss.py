r"""ausschluss.py — leave-generator-out: how well does the approach detect generators it has NEVER seen?

Per experiment one or two generators (or a model family) are removed from training and validation
(train3.py --ohne-gen); everything else equals the baseline aus_0 trained alongside: same packs, seeds (0-4),
calibration (kalibriere.py), backbones (--rueckgrate). Baseline and all experiment heads then run on the same
standard test set (bench.py, production for comparison).

Per experiment: hit rate and AUC on the removed generators (baseline -> removed: the key number), mean change
on all other generator groups and channels (side effects), false positives on the test humans.
Output: daten/<praefix>schluss.json and docs/de/AUSSCHLUSS*.md.

    python ausschluss.py lauf [--praefix ausS --rueckgrate siglip_mid]
    python ausschluss.py lauf --praefix aus9 --rueckgrate siglip_mid --aug-name aug2   (p9 augmentation)
    python ausschluss.py auswerten [--praefix ausS --rueckgrate siglip_mid]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kanaele import oeffentlich                  # noqa: E402
from lesen import DATEN                          # noqa: E402

HIER = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (Kennung, Titel, Generatornamen fuer --ohne-gen, betroffene Testgruppen in bench.py)
EXPERIMENTE = [
    ("A", "Nano Banana + OpenAI (geschlossen, 2025)", ["Nano Banana", "OpenAI"],
     ["K CivitAI Nano Banana", "K CivitAI OpenAI"]),
    ("B", "Seedream + Imagen4 (geschlossen)", ["Seedream", "Imagen4"],
     ["K CivitAI Seedream", "K CivitAI Imagen4"]),
    ("C", "Grok", ["Grok"], ["K CivitAI Grok"]),
    ("D", "Flux-Familie (Flux.1/Flux.2, Anime-Flux)",
     ["Flux.1 D", "Flux.1 S", "Flux.1 Krea", "Flux.2 D", "Flux.2 Klein 9B", "Flux.2 Klein 4B", "animepro_flux"],
     ["K CivitAI Flux.1 D", "K CivitAI Flux.1 S", "K CivitAI Flux.1 Krea", "K CivitAI Flux.2 D",
      "K CivitAI Flux.2 Klein 9B", "K eigen animepro_flux"]),
    ("E", "Illustrious-Familie (Illustrious, NoobAI, WAI)",
     ["Illustrious", "NoobAI", "illustrious", "noobai", "wai_illustrious", "aibooru:illustrious", "aibooru:noobai"],
     ["K CivitAI Illustrious", "K CivitAI NoobAI", "K eigen illustrious", "K eigen noobai", "K eigen wai_illustrious"]),
    ("F", "NovelAI (v1, v2, AIBooru)", ["nai_v1", "nai_v2", "aibooru:novelai"],
     ["K eigen nai_v1", "K eigen nai_v2", "K AIBooru NovelAI"]),
    ("G", "Pony (Pony, Pony V7)", ["Pony", "Pony V7", "pony"],
     ["K CivitAI Pony", "K CivitAI Pony V7", "K eigen pony"]),
    ("H", "Midjourney + Nijijourney",
     ["Midjourney (Felix)", "Nijijourney v5", "Midjourney v6", "aibooru:nijijourney", "aibooru:midjourney"],
     ["K Midjourney (Felix)", "K Nijijourney v5", "K Midjourney v6", "K AIBooru Nijijourney",
      "T midjourney_gallery", "T mj_prompts_daily"]),
    ("I", "Stable Diffusion 1.x (SD 1.5, DiffusionDB 2022)", ["SD 1.5", "sd15", "SD 1.x 2022 (DiffusionDB)"],
     ["K CivitAI SD 1.5", "K eigen sd15", "K DiffusionDB SD 1.x 2022", "T ai_art_random"]),
    ("J", "Qwen-Image (Qwen, Qwen 2)", ["Qwen", "Qwen 2", "Qwen 2.1", "qwen_image"],
     ["K CivitAI Qwen", "K eigen qwen_image"]),
    ("K", "Z-Image (Base, Turbo, Anime)", ["ZImageBase", "ZImageTurbo", "z_anime"],
     ["K CivitAI ZImageBase", "K CivitAI ZImageTurbo", "K eigen z_anime"]),
]


PRAEFIX = "aus"     # per --praefix: zweiter Satz Ausschlusstests mit anderen Rueckgraten (z. B. "ausS")


def tag(k):
    return f"{PRAEFIX}_{k}"


def lauf(a):
    # Basiskopf mit identischen Daten/Seeds/Rueckgraten neu trainieren (aus_0), damit Basis und
    # Ausschlusskoepfe garantiert auf demselben Datenstand stehen
    liste = ([("0", "Basis (nichts entfernt)", [], [])] if a.basis == tag("0") else []) + EXPERIMENTE
    for k, titel, namen, _ in liste:
        if a.nur and k not in a.nur:
            continue
        if os.path.exists(os.path.join(DATEN, f"kopf_{tag(k)}_s4.npz")):
            print(f"{k} schon trainiert", flush=True)
            continue
        print(f"{datetime.datetime.now():%H:%M} {k}: {titel}", flush=True)
        log = open(os.path.join(DATEN, f"{PRAEFIX}schluss_{k}.log"), "w", encoding="utf-8")
        for cmd in ([PY, os.path.join(HIER, "train3.py"), "--zusatz", a.zusatz, "--tag", tag(k),
                     "--rating-ausgleich", "--kalib-p2", "--rueckgrate", a.rueckgrate, "--aug-name", a.aug_name]
                    + (["--ohne-gen", "|".join(namen)] if namen else [])
                    + (["--quelle-gruppe", a.quelle_gruppe] if a.quelle_gruppe else []),
                    [PY, os.path.join(HIER, "kalibriere.py"), tag(k)]):
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode:
                print(f"  ABBRUCH in {k} ({os.path.basename(cmd[1])}), siehe ausschluss_{k}.log", flush=True)
                return 1
    return 0


def entfernt(k):
    """Zahl der entfernten Bilder aus dem Trainingslog."""
    p = os.path.join(DATEN, f"{PRAEFIX}schluss_{k}.log")
    for z in open(p, encoding="utf-8", errors="replace") if os.path.exists(p) else []:
        if z.startswith("Ausschluss "):
            t = z.split(":")[-1].split("Bilder")[0].split("+")
            return sum(int(x) for x in t)
    return None


def auswerten(a):
    fertig = [e for e in EXPERIMENTE if os.path.exists(os.path.join(DATEN, f"kopf_{tag(e[0])}_s4.npz"))]
    koepfe = [a.basis] + [tag(e[0]) for e in fertig]
    subprocess.run([PY, os.path.join(HIER, "bench.py"), *koepfe, "--json", f"bench_{PRAEFIX}schluss.json"], check=True,
                   stdout=open(os.path.join(DATEN, f"{PRAEFIX}schluss_bench.log"), "w", encoding="utf-8"),
                   stderr=subprocess.STDOUT)
    b = json.load(open(os.path.join(DATEN, f"bench_{PRAEFIX}schluss.json"), encoding="utf-8"))
    erg, art = b["ergebnis"], b["art"]
    prod = max((d for d in erg if ":" in d), key=lambda d: erg[d]["auc"]["gemischt alle"])
    basis = erg[a.basis]
    ki = [g for g in art if art[g].startswith("k_")]
    doku = {"basis": a.basis, "stand": f"{datetime.datetime.now():%Y-%m-%d %H:%M}", "bester_produktiv": prod,
            "experimente": []}
    md = ["# Ausschlusstests: unbekannte Generatoren", "",
          f"Stand {doku['stand']}. Basiskopf **{a.basis}** (fünf Köpfe, Seeds 0–4; Rückgrate: {a.rueckgrate}). Je Experiment wird der "
          "Kopf mit identischen Daten, Seeds und Kalibrierung neu trainiert — nur ohne die genannten "
          "Generatoren (Training und Validierung). Getestet wird auf dem Standard-Testsatz (`bench.py`: "
          "Testsplits je Generator, 15 Telegram-KI-Kanäle, drei menschliche Testgruppen). Schwelle je Kopf: "
          "1 % Fehlalarm im Mittel der drei menschlichen val-Gruppen.", "",
          f"Zum Vergleich jeweils das beste Produktionsmodell (`{prod.split(':')[-1]}`), das keinen der "
          "Generatoren im Training hatte.", "",
          "| Exp. | entfernt | Bilder raus | Treffer vorher → ohne | AUC vorher → ohne | Produktion | "
          "übrige Gruppen Δ | Kanäle Δ | Fehlalarm |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, titel, namen, gruppen in fertig:
        e = erg[tag(k)]
        g = [x for x in gruppen if x in e["treffer"]]
        tv, tn = np.mean([basis["treffer"][x] for x in g]), np.mean([e["treffer"][x] for x in g])
        av, an = np.mean([basis["auc"][x] for x in g]), np.mean([e["auc"][x] for x in g])
        tp = np.mean([erg[prod]["treffer"][x] for x in g])
        rest = [x for x in ki if x not in g]
        d_rest = np.mean([e["treffer"][x] - basis["treffer"][x] for x in rest if art[x] == "k_gen"])
        d_kan = np.mean([e["treffer"][x] - basis["treffer"][x] for x in rest if art[x] == "k_kanal"])
        fa = e["fa"]["gemischt"]
        eintrag = {"id": k, "titel": titel, "entfernt": namen, "bilder_raus": entfernt(k), "gruppen": {
            x: {"treffer_basis": basis["treffer"][x], "treffer_ohne": e["treffer"][x],
                "auc_basis": basis["auc"][x], "auc_ohne": e["auc"][x], "treffer_produktion": erg[prod]["treffer"][x]}
            for x in g}, "delta_uebrige_generatoren": float(d_rest), "delta_kanaele": float(d_kan),
            "fa_gemischt": fa, "fa_basis": basis["fa"]["gemischt"]}
        doku["experimente"].append(eintrag)
        md.append(f"| {k} | {titel} | {eintrag['bilder_raus'] or '–'} | {tv:.1%} → **{tn:.1%}** | {av:.4f} → {an:.4f} | "
                  f"{tp:.1%} | {d_rest:+.1%} | {d_kan:+.1%} | {fa:.1%} |")
    md += ["", "Treffer: Anteil erkannter KI-Bilder der entfernten Generatoren (Mittel ihrer Testgruppen). "
           "„übrige Gruppen Δ“ und „Kanäle Δ“: mittlere Veränderung auf allem anderen — zeigt, ob das "
           "Entfernen auch Nachbarn schwächt. Streuung zwischen zwei Trainingsläufen mit anderen Seeds: "
           "bis ±2,6 Punkte je Kanal (p3 gegen p3b).", "", "## Je Gruppe", ""]
    for x in doku["experimente"]:
        md += [f"### {x['id']}: {x['titel']}", "", f"Entfernt: `{'`, `'.join(x['entfernt'])}`", "",
               "| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |", "|---|---|---|---|---|---|"]
        for g, v in x["gruppen"].items():
            md.append(f"| {g[2:]} | {v['treffer_basis']:.1%} | {v['treffer_ohne']:.1%} | {v['auc_basis']:.4f} | "
                      f"{v['auc_ohne']:.4f} | {v['treffer_produktion']:.1%} |")
        md.append("")
    json.dump(doku, open(os.path.join(DATEN, f"{PRAEFIX}schluss.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(os.path.join(HIER, "docs", "de", "AUSSCHLUSS.md" if PRAEFIX == "aus" else f"AUSSCHLUSS_{PRAEFIX}.md"), "w", encoding="utf-8").write(oeffentlich("\n".join(md) + "\n"))
    print("\n".join(md[:14 + len(fertig)]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("was", choices=["lauf", "auswerten"])
    ap.add_argument("--basis", default="aus_0", help="aus_0 = Basis wird mit trainiert (empfohlen)")
    ap.add_argument("--rueckgrate", default="clip_mid,dino_mid,siglip_mid")
    ap.add_argument("--zusatz", default="v3z2,v3z3,p2z,p3z,p4z,p5z")
    ap.add_argument("--quelle-gruppe", default="midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney")
    ap.add_argument("--praefix", default="aus")
    ap.add_argument("--aug-name", default="aug", help="aug2 = Alltags-Bearbeitungen (p9, Praefix aus9)")
    ap.add_argument("--nur", default="", help="nur diese Experimente, z. B. ABC")
    a = ap.parse_args()
    global PRAEFIX
    PRAEFIX = a.praefix
    if a.basis == "aus_0":
        a.basis = tag("0")
    return lauf(a) if a.was == "lauf" else auswerten(a)


if __name__ == "__main__":
    sys.exit(main())
