r"""pruefe.py — scan a folder with the detector. READ-ONLY.

No image is ever moved, renamed or deleted. The result is a CSV with one score per image and an HTML page
with the most suspicious images as a contact sheet for manual review.

Per image — exactly as in training and testing:
  0. Preview thumbnails (*_thumb) and images whose longer side is below 512 px are skipped (the heads never
     saw upscaled images; see klein_test.py for how they behave).
  1. Photo filter (imgutils anime_real, "real" >= 0.75): real photographs are out of scope and only listed.
  2. Normalization norm.normiere (longer side 512 px, JPEG q95 4:4:4), no perturbation.
  3. Features of the head's backbones (default head p8sx: SigLIP 2 so400m, five intermediate layers),
     in batches; mean logit of the ensemble heads.
  4. Verdict at the head's two calibrated thresholds: "AI (1 %)" means about 1 in 100 human artworks
     (never seen in training) would score this high; "suspicious (5 %)" likewise for 5 in 100.

    python pruefe.py "path/to/images"
    python pruefe.py <folder> --kopf p8sx --max 2000
"""
import argparse
import csv
import html
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed import wd_vorbereiten                 # noqa: E402
from kandidat import _clip, _kopf, _wd, logit_stapel    # noqa: E402
from lesen import DATEN                          # noqa: E402
from norm import normiere                        # noqa: E402

ENDUNGEN = (".jpg", ".jpeg", ".png", ".webp")
REAL_MIN = 0.75


def bilder(ordner, mit_quarantaene):
    for dp, dn, fn in os.walk(ordner):
        if not mit_quarantaene and "_ai_quarantine" in dp:
            continue
        for f in sorted(fn):
            # Telegram legt zu jedem Bild ein Vorschaubild *_thumb.jpg ab
            if f.lower().endswith(ENDUNGEN) and "_thumb." not in f.lower():
                yield os.path.join(dp, f)


def stapelrechner(rueckgrate):
    """Je Rueckgrat eine Funktion (Bilder, Threadpool) -> Merkmale, mit
    GENAU der Vorverarbeitung aus embed.py. Wer hier anders rechnet als
    dort, bewertet mit einem anderen Modell als dem trainierten."""
    import torch
    import embed
    r = {}
    if "wd_eva02" in rueckgrate:
        s, ein, aus = _wd()
        r["wd_eva02"] = lambda bs, ex: s.run(aus, {ein: np.stack(list(ex.map(embed.wd_vorbereiten, bs)))})[0]
    if "clip_l14" in rueckgrate:
        m, prep = _clip()

        def clip(bs, ex):
            with torch.no_grad():
                return m.encode_image(torch.stack([prep(b) for b in bs]).cuda().half()).float().cpu().numpy()
        r["clip_l14"] = clip
    if "clip_mid" in rueckgrate:
        m2, prep2, fang = embed.clip_mid_modell()

        def clip_mid(bs, ex):
            x = torch.stack([prep2(b) for b in bs]).cuda().half()
            fang["_n"] = x.shape[0]
            with torch.no_grad():
                m2.encode_image(x)
            return torch.cat([fang[n] for n in embed.CLIP_MID_BLOECKE], 1).cpu().numpy()
        r["clip_mid"] = clip_mid
    if "dino" in rueckgrate or "dino_mid" in rueckgrate:
        dm, dp = embed.dino_modell()
        r["dino"] = lambda bs, ex: embed.dino_merkmale(dm, dp, bs)
        r["dino_mid"] = lambda bs, ex: embed.dino_mid_merkmale(dm, dp, bs)
    if "siglip_mid" in rueckgrate:
        sm, sp = embed.siglip_modell()
        r["siglip_mid"] = lambda bs, ex: embed.siglip_mid_merkmale(sm, sp, bs, ex)
    return r


def vorbereiten(pfad):
    """Foto-Filter und Zurichtung; laeuft in Threads."""
    from imgutils.validate import anime_real
    try:
        with Image.open(pfad) as im:
            if max(im.size) < 512:
                # im Training nie vorgekommen (aufbau2 verwirft sie), ein
                # Score waere geraten — also nur vermerken
                return pfad, "zu klein", None
        lab, sc = anime_real(pfad)
        if lab == "real" and sc >= REAL_MIN:
            return pfad, "foto", None
        b = normiere(pfad)[0]
        return pfad, None, Image.open(io.BytesIO(b)).convert("RGB")
    except Exception as e:
        return pfad, f"fehler: {type(e).__name__}", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ordner")
    ap.add_argument("--kopf", default="p8sx")
    ap.add_argument("--max", type=int, default=0, help="hoechstens so viele Bilder")
    ap.add_argument("--mit-quarantaene", action="store_true")
    ap.add_argument("--aus", default=None, help="Zielordner fuer CSV und HTML")
    ap.add_argument("--top", type=int, default=200, help="so viele Verdachtsfaelle in der HTML")
    args = ap.parse_args()
    import torch

    # Ensemble (kopf_<name>_s0.._sN, train3.py) oder ein einzelner Kopf
    koepfe = []
    while os.path.exists(os.path.join(DATEN, f"kopf_{args.kopf}_s{len(koepfe)}.npz")):
        koepfe.append(_kopf(f"{args.kopf}_s{len(koepfe)}"))
    koepfe = koepfe or [_kopf(args.kopf)]
    k = koepfe[0]
    s1, s5 = float(k["schwelle_1"]), float(k["schwelle_5"])
    rg = [str(r) for r in k["rueckgrate"]]
    print(f"Kopf {args.kopf}: {len(koepfe)} Koepfe, Schwelle 1 % bei {s1:+.2f}", flush=True)
    rechner = stapelrechner(rg)

    pfade = list(bilder(args.ordner, args.mit_quarantaene))
    if args.max:
        pfade = pfade[:args.max]
    aus = args.aus or os.path.join(os.path.dirname(os.path.abspath(__file__)), "pruefung")
    os.makedirs(aus, exist_ok=True)
    name = os.path.basename(os.path.normpath(args.ordner)) or "ordner"
    print(f"{len(pfade)} Bilder in {args.ordner}", flush=True)

    ergebnis = []           # (pfad, logit oder None, vermerk)
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for s in range(0, len(pfade), 32):
            stapel = list(ex.map(vorbereiten, pfade[s:s + 32]))
            ok = [(p, b) for p, v, b in stapel if b is not None]
            for p, v, b in stapel:
                if b is None:
                    ergebnis.append((p, None, v))
            if not ok:
                continue
            X = np.concatenate([rechner[r]([b for _, b in ok], ex) for r in rg], axis=1)
            z = np.mean([logit_stapel(X, kk) for kk in koepfe], 0)
            for (p, _), zi in zip(ok, z):
                ergebnis.append((p, float(zi), ""))
            if (s // 32) % 20 == 0 and s:
                el = time.time() - t0
                print(f"  {s}/{len(pfade)}  {s/el:.1f}/s", flush=True)

    def urteil(z):
        if z is None:
            return ""
        return "KI (1 %)" if z > s1 else ("verdächtig (5 %)" if z > s5 else "unauffällig")

    csv_pfad = os.path.join(aus, f"{name}.csv")
    with open(csv_pfad, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["pfad", "ki_wahrscheinlichkeit", "urteil", "vermerk"])
        for p, z, v in sorted(ergebnis, key=lambda e: -(e[1] if e[1] is not None else -1e9)):
            w.writerow([p, "" if z is None else f"{1/(1+np.exp(-z)):.4f}", urteil(z), v])

    bewertet = [(p, z) for p, z, _ in ergebnis if z is not None]
    n1 = sum(1 for _, z in bewertet if z > s1)
    n5 = sum(1 for _, z in bewertet if s5 < z <= s1)
    fotos = sum(1 for _, _, v in ergebnis if v == "foto")
    klein = sum(1 for _, _, v in ergebnis if v == "zu klein")
    top = sorted(bewertet, key=lambda e: -e[1])[:args.top]
    kacheln = "".join(
        f'<figure><a href="file:///{html.escape(p.replace(os.sep, "/"))}">'
        f'<img loading="lazy" src="file:///{html.escape(p.replace(os.sep, "/"))}"></a>'
        f'<figcaption>{1/(1+np.exp(-z)):.3f} · {html.escape(urteil(z))}<br>'
        f'<span>{html.escape(os.path.basename(p))}</span></figcaption></figure>' for p, z in top)
    seite = (f"<!doctype html><meta charset='utf-8'><title>Prüfung {html.escape(name)}</title><style>"
             "body{margin:0;padding:16px 20px;background:#15161a;color:#e8e8ea;"
             "font:13px/1.5 system-ui,sans-serif}h1{font-size:16px;margin:0 0 4px}"
             ".sub{color:#9a9ba4;margin-bottom:14px}.raster{display:flex;flex-wrap:wrap;gap:10px}"
             "figure{margin:0;width:190px}img{width:190px;height:240px;object-fit:cover;border-radius:4px;"
             "background:#0d0e11}figcaption{font-size:11px;color:#c8cad2}figcaption span{color:#7e808a;"
             "word-break:break-all}</style>"
             f"<h1>Prüfung: {html.escape(args.ordner)}</h1><div class='sub'>{len(pfade)} Bilder · "
             f"{len(bewertet)} bewertet · {fotos} Fotos und {klein} zu kleine übersprungen · <b>{n1}</b> KI bei 1-%-Schwelle · "
             f"{n5} zusätzlich verdächtig bei 5 % · Kopf {html.escape(args.kopf)} · "
             f"nur gelesen, nichts verschoben. Die {len(top)} höchsten Scores:</div>"
             f"<div class='raster'>{kacheln}</div>")
    html_pfad = os.path.join(aus, f"{name}.html")
    with open(html_pfad, "w", encoding="utf-8") as f:
        f.write(seite)
    el = time.time() - t0
    print(f"\n{len(bewertet)} bewertet in {el/60:.1f} min ({len(pfade)/max(el,1):.1f}/s), "
          f"{fotos} Fotos, {klein} zu klein, {len(ergebnis)-len(bewertet)-fotos-klein} Fehler")
    print(f"KI bei 1-%-Schwelle: {n1} ({n1/max(len(bewertet),1):.1%}), "
          f"zusaetzlich verdaechtig bei 5 %: {n5}")
    print(f"-> {csv_pfad}\n-> {html_pfad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
