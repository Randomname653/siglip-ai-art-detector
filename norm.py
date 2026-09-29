r"""norm.py — prepare every image of both classes in exactly the same way.

The dataset is built so that human and machine images differ ONLY in origin. Raw files do not: in the first
dataset all human wallpapers had a longer side of exactly 1024 px, most generators rendered 1024-1536 px
buckets and SD 1.5 / NAI v1 smaller ones. On raw files a model could spot SD 1.5 by size alone — the trap
described in "Fake or JPEG?" (arXiv 2403.17608), with resolution instead of compression.

Therefore: everything is downscaled to a longer side of 512 px and saved identically as JPEG q95 4:4:4.
Downscaling also destroys the 8x8 grid of the first compression, for both classes. Training additionally sees
only randomly perturbed copies (stoere: rescale, crop, JPEG), identical for both classes — without it a head
reached AUC 0.987 on clean test images but 0.643 once both classes were rescaled.

Images are stored packed in ONE file plus index (fast sequential reads on a hard disk).

    python norm.py            # build the v1 manifest + pack file
    python norm.py --probe 20 # just a few, for checking
"""
import argparse
import io
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATEN, V1_DB                    # noqa: E402

# v1-Pack (norm512) aus dem eigenen Generator-Datensatz; nur noch fuer die Nachvollziehbarkeit
DB = os.environ.get("DETEKTOR_V1_DB", V1_DB)
OUT = DATEN
LANG = 512          # lange Kante nach der Normierung
QUALITAET = 95

# Die vierzehn Generatoren des Datensatzes. Die Schrittzahl-Varianten (name@N),
# chroma und animepro_flux_fp8 gehoeren nicht dazu.
sys.path.insert(1, os.path.dirname(DB))
try:                                              # nur im v1-Datensatz des Labors vorhanden
    from generate import ORDER                   # noqa: E402
except ImportError:
    ORDER = ()


def normiere(pfad):
    """Die EINE Zurichtung. Alles, was ein Modell zu sehen bekommt, laeuft hier
    durch — Messlatten, Rueckgrate, Leak-Test. Rueckgabe: JPEG-Bytes plus die
    Groesse vorher und nachher (fuer den Leak-Test auf den Rohgroessen)."""
    with Image.open(pfad) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
        f = LANG / max(w0, h0)
        w, h = max(1, round(w0 * f)), max(1, round(h0 * f))
        im = im.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=QUALITAET, subsampling=0)
    return buf.getvalue(), w0, h0, w, h


def stoere(bild, i):
    """Zufaellige, aber je Bild FESTE Stoerung — fuer beide Klassen gleich
    verteilt. Das Training sieht nur gestoerte Bilder.

    Warum (gemessen 2026-09-24): ein Kopf auf ungestoerten Bildern erreichte
    Test-AUC 0,987 — und fiel auf 0,643, sobald BEIDE Klassen nur auf 75 %
    verkleinert und mit q85 gespeichert wurden. Er hatte Verarbeitungsspuren
    gelernt: alle Negative stehen auf exakt 1024 px, die Normierung auf 512
    war fuer sie eine exakte Halbierung (ganzzahliges, sehr regelmaessiges
    Resampling-Muster), fuer die Positiven ein krummer Faktor. Eine
    Umskalierung um 1,7 % verschob den Score der Negative um +3,4 Logits.
    Auf Telegram-Bildern (q<=92, beliebig skaliert) waere er wertlos gewesen.

    Die Stoerung ueberschreibt diese Spuren: nie ganzzahliger Massstab,
    wechselnder Filter, wechselnde Qualitaet und Farbunterabtastung. Der
    Seed haengt am Index, damit Embeddings, Messlatten und Kontrollen
    dieselben Bytes sehen."""
    rng = np.random.default_rng([int(i), 20260924])
    f = rng.uniform(0.6, 0.95)
    q = int(rng.integers(70, 96))
    sub = 2 if rng.random() < 0.5 else 0
    filt = (Image.LANCZOS, Image.BICUBIC, Image.BILINEAR)[int(rng.integers(3))]
    w, h = bild.size
    b = bild.resize((max(1, round(w * f)), max(1, round(h * f))), filt)
    buf = io.BytesIO()
    b.save(buf, "JPEG", quality=q, subsampling=sub)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


_WOERTER = ("art", "new", "wip", "commission", "open", "patreon", "follow", "sketch", "daily", "color", "study",
            "fanart", "original", "character", "background", "practice", "happy", "birthday", "summer", "night",
            "done", "finally", "my", "oc", "redraw", "vs", "2021", "2019", "part", "page", "chapter", "hd", "4k",
            "wallpaper", "free", "request", "twitter", "pixiv", "full", "version", "link", "bio", "love", "this")
_SCHRIFTEN = ("arial.ttf", "arialbd.ttf", "verdana.ttf", "times.ttf", "georgia.ttf", "impact.ttf", "comic.ttf",
              "segoeui.ttf", "tahoma.ttf", "trebuc.ttf", "consola.ttf", "calibri.ttf")


def _text(b, rng):
    from PIL import ImageDraw, ImageFont
    b = b.copy()
    w, h = b.size
    d = ImageDraw.Draw(b, "RGBA")
    gr = int(h * rng.uniform(0.03, 0.12))
    try:
        f = ImageFont.truetype(_SCHRIFTEN[int(rng.integers(len(_SCHRIFTEN)))], max(9, gr))
    except OSError:
        f = ImageFont.load_default()
    for _ in range(int(rng.integers(1, 4))):
        t = " ".join(rng.choice(_WOERTER, int(rng.integers(1, 6))))
        if rng.random() < 0.3:
            t = "@" + t.replace(" ", "_")
        x, y = int(rng.uniform(0, 0.7) * w), int(rng.uniform(0, 0.92) * h)
        if rng.random() < 0.35:                      # Balken hinter dem Text (Meme-/Untertitel-Stil)
            d.rectangle((0, y - 2, w, y + gr + 4), fill=tuple(int(c) for c in rng.integers(0, 256, 3)) + (int(rng.integers(90, 230)),))
        farbe = tuple(int(c) for c in rng.integers(0, 256, 3)) + (int(rng.integers(60, 256)),)
        d.text((x, y), t, font=f, fill=farbe)
    return b


def stoere2(bild, i):
    """Wie stoere, aber vorher mit Wahrscheinlichkeit je eine Alltags-Bearbeitung — fuer beide Klassen gleich:
    Ausschnitt, Text/Wasserzeichen, Rauschen, Farbe, Graustufen, Weichzeichnen/Schaerfen, Hochskalier-Rundreise.

    Warum (robust_test.py, 28.09.): p8sx blieb bei solchen Bearbeitungen treffsicher, aber menschliche Bilder
    rutschten Richtung KI (Ausschnitt 4,2 %, Rauschen 3,5 %, Text 2,8 %, 4x-Hochskalierer 2,4 % Fehlalarm statt
    1,3 %). Der Kopf kannte nur Skalierung + JPEG. Anderer Seed als stoere, damit die Bearbeitungen nicht mit
    Skalierung/Qualitaet korrelieren. Texte und Schriften hier bewusst anders als im Robustheitstest."""
    from PIL import ImageEnhance, ImageFilter, ImageOps
    rng = np.random.default_rng([int(i), 20260928])
    b = bild
    if rng.random() < 0.3:
        w, h = b.size
        fw, fh = rng.uniform(0.45, 0.9), rng.uniform(0.45, 0.9)
        x0, y0 = rng.uniform(0, 1 - fw) * w, rng.uniform(0, 1 - fh) * h
        b = b.crop((int(x0), int(y0), int(x0 + fw * w), int(y0 + fh * h)))
    if rng.random() < 0.1:
        f = rng.uniform(0.35, 0.6)
        w, h = b.size
        b = b.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS).resize((w, h), Image.BICUBIC)
        if rng.random() < 0.6:
            b = b.filter(ImageFilter.UnsharpMask(rng.uniform(1, 2.5), int(rng.integers(60, 160)), 2))
    if rng.random() < 0.2:
        b = ImageEnhance.Color(b).enhance(rng.uniform(0.7, 1.4))
        b = ImageEnhance.Brightness(b).enhance(rng.uniform(0.85, 1.15))
        b = ImageEnhance.Contrast(b).enhance(rng.uniform(0.85, 1.2))
    if rng.random() < 0.05:
        b = ImageOps.grayscale(b).convert("RGB")
    r = rng.random()
    if r < 0.08:
        b = b.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.6)))
    elif r < 0.16:
        b = b.filter(ImageFilter.UnsharpMask(rng.uniform(1, 3), int(rng.integers(50, 180)), 2))
    if rng.random() < 0.2:
        b = _text(b, rng)
    if rng.random() < 0.15:
        a = np.asarray(b, np.float32)
        a += rng.normal(0, rng.uniform(1.5, 8), a.shape)
        b = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return stoere(b, i)


def _arbeit(args):
    i, pfad = args
    try:
        return i, normiere(pfad), None
    except Exception as e:
        return i, None, f"{type(e).__name__}: {e}"


def manifest():
    """Alle Bilder beider Klassen mit Split. Negative zuerst, dann die
    Positiven nach Modell — sortiert nach Pfad, damit die HDD beim Lesen
    moeglichst am Stueck arbeitet."""
    con = sqlite3.connect(DB, timeout=60)
    split = {}
    zeilen = []
    for w, loc, sp, grp in con.execute(
            "SELECT wh_id, local, split, grp FROM images WHERE prompt IS NOT NULL "
            "AND prompt != '' AND excl IS NULL AND split IS NOT NULL"):
        split[w] = (sp, grp)
        zeilen.append(dict(wh_id=w, model="mensch", label=0, split=sp,
                           grp=grp, src=loc))
    for w, m, loc in con.execute("SELECT wh_id, model, local FROM generated"):
        if m not in ORDER or w not in split:
            continue
        sp, grp = split[w]
        zeilen.append(dict(wh_id=w, model=m, label=1, split=sp, grp=grp, src=loc))
    con.close()
    zeilen.sort(key=lambda z: (z["label"], z["src"].replace("/", "\\").lower()))
    return zeilen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--worker", type=int, default=16)
    args = ap.parse_args()

    zeilen = manifest()
    if args.probe:
        # aus jeder Klasse und jedem Modell ein paar
        gesehen, auswahl = {}, []
        for z in zeilen:
            if gesehen.get(z["model"], 0) < max(1, args.probe // 15):
                auswahl.append(z)
                gesehen[z["model"]] = gesehen.get(z["model"], 0) + 1
        zeilen = auswahl
    n = len(zeilen)
    print(f"{n} Bilder: {sum(1 for z in zeilen if z['label']==0)} Mensch, "
          f"{sum(1 for z in zeilen if z['label']==1)} Maschine", flush=True)

    os.makedirs(OUT, exist_ok=True)
    stamm = "probe" if args.probe else "norm512"
    bin_pfad = os.path.join(OUT, stamm + ".bin")
    off = np.zeros(n, np.int64)
    lng = np.zeros(n, np.int32)
    groesse = np.zeros((n, 4), np.int32)     # w0, h0, w, h
    fehler = {}

    t0 = time.time()
    pos = 0
    # Ergebnisse kommen in beliebiger Reihenfolge; sie werden in der
    # Reihenfolge des Manifests geschrieben, damit die Packdatei spaeter
    # am Stueck gelesen werden kann.
    puffer = {}
    naechst = 0
    with open(bin_pfad, "wb") as f, \
            ProcessPoolExecutor(max_workers=args.worker) as ex:
        for k, (i, erg, err) in enumerate(ex.map(
                _arbeit, ((i, z["src"]) for i, z in enumerate(zeilen)),
                chunksize=32)):
            puffer[i] = (erg, err)
            while naechst in puffer:
                erg, err = puffer.pop(naechst)
                if erg is None:
                    fehler[naechst] = err
                    off[naechst], lng[naechst] = -1, 0
                else:
                    b, w0, h0, w, h = erg
                    f.write(b)
                    off[naechst], lng[naechst] = pos, len(b)
                    groesse[naechst] = (w0, h0, w, h)
                    pos += len(b)
                naechst += 1
            if (k + 1) % 5000 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{n}  {(k+1)/el:.0f}/s  "
                      f"{pos/2**30:.1f} GB  Rest {(n-k-1)/((k+1)/el)/60:.1f} min",
                      flush=True)

    np.save(os.path.join(OUT, stamm + "_off.npy"), off)
    np.save(os.path.join(OUT, stamm + "_len.npy"), lng)
    np.save(os.path.join(OUT, stamm + "_size.npy"), groesse)
    for i, z in enumerate(zeilen):
        z["i"] = i
    with open(os.path.join(OUT, stamm + "_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(zeilen, f)
    el = time.time() - t0
    print(f"\n{n - len(fehler)} normiert, {len(fehler)} Fehler, "
          f"{pos/2**30:.2f} GB in {el/60:.1f} min ({n/el:.0f}/s)")
    for i, e in list(fehler.items())[:10]:
        print("  FEHLER", zeilen[i]["src"], e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
