r"""aufbau2.py — Datensatz v2: Wallhaven + Generatoren, dazu echtes Telegram-Material.

Anlass (2026-09-24): der Detektor aus v1 schlug auf eigenen Testbildern die
Produktion weit, auf echtem Telegram-Material aber nicht (AUC 0,878 gegen
legekka 0,937), und seine 1-%-Schwelle erzeugte dort 16,9 % Fehlalarm. Sein
Begriff von "menschlich" war nur Wallhaven. Entscheidung des Users: der
Detektor soll ALLEIN stehen, besser gemacht mit dem alten Telegram-Material.

Vier Quellen:

    wallhaven   Mensch   69.693   Split wie v1 (Motivgruppen)
    generiert   KI       95.693   Split wie v1 (gleiche Motive wie Wallhaven)
    telegram    Mensch   Posts von VOR 2022 aus unbearbeitet — da gab es die
                         Generatoren nicht. Split nach KANAL, nicht nach Bild.
    ai_only     KI       <DATASETS>\Ai Only — zwei Telegram-Exporte aus
                         KI-Kanaelen (2023-2026) plus 300 lose Dateien.
                         Split nach Kanal: der grosse Export trainiert, der
                         kleine plus die losen testen.

Nach Kanal, weil ein Kanal einen Stil, wiederkehrende Motive und Reposts hat.
Liegen seine Bilder in Training und Test, misst der Test den Kanal. Mehrere
Exporte zeigen denselben Kanal (88 Exporte, 77 Kanaele; ein Kanal liegt
viermal vor) — kanalgruppen.json fasst sie zusammen, gebildet ueber
gemeinsame Dateinamen. Je Kanal hoechstens MAX_KANAL Bilder, sonst stellt ein
Riesenkanal mit 322.000 Posts den halben Datensatz.

Die Falle, die hier vorab entschaerft wird: Telegram-Bilder haben eine echte
Telegram-Kompression hinter sich (lange Kante <= 1280, JPEG um q87, 4:2:0),
Wallhaven und die Generatoren nicht. Ohne Ausgleich hiesse "war schon mal auf
Telegram" fuer den Detektor "menschlich" — dieselbe Klasse von Fehler wie das
Resampling-Leck aus v1. Deshalb bekommen Wallhaven und die Generatoren vor der
Zurichtung eine simulierte Hochlade-Kompression. Danach tragen alle vier
Quellen dieselbe Vorgeschichte, und norm.stoere kommt beim Training oben drauf.

    python lab/detektor/aufbau2.py --plan
    python lab/detektor/aufbau2.py
"""
import argparse
import collections
import io
import json
import os
import random
import re
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import TELEGRAM_EINGANG                # noqa: E402
from pfade import DATASETS                       # noqa: E402
from lesen import DATEN                          # noqa: E402
from norm import LANG, QUALITAET, manifest as manifest_v1   # noqa: E402

TELEGRAM = TELEGRAM_EINGANG
AI_ONLY = f"{DATASETS}/Ai Only"
AI_TEST = {"ChatExport_2026-05-31 (1)", "."}      # kleiner Export + lose Dateien
MAX_KANAL = 5000
STAMM = "v2"


def hochladen(im):
    """Was Telegram beim Hochladen tut: lange Kante hoechstens 1280, JPEG q87 4:2:0."""
    w, h = im.size
    if max(w, h) > 1280:
        f = 1280 / max(w, h)
        im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=87, subsampling=2)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def normiere2(pfad, hoch):
    """Wie norm.normiere, optional mit Hochlade-Simulation davor. Bilder mit
    langer Kante unter 512 werden verworfen — sie muessten hochgerechnet werden."""
    with Image.open(pfad) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
        if max(w0, h0) < LANG:
            raise ValueError("zu klein")
        if hoch:
            im = hochladen(im)
        f = LANG / max(im.size)
        w, h = max(1, round(im.size[0] * f)), max(1, round(im.size[1] * f))
        im = im.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=QUALITAET, subsampling=0)
    return buf.getvalue(), w0, h0, w, h


def _arbeit(a):
    i, pfad, hoch = a
    try:
        return i, normiere2(pfad, hoch), None
    except Exception as e:
        return i, None, f"{type(e).__name__}: {e}"


def telegram_zeilen(rng):
    grp = json.load(open(os.path.join(DATEN, "kanalgruppen.json"), encoding="utf-8"))
    kanal = {k.split("|", 1)[1]: v for k, v in grp.items() if k.startswith("mensch|")}
    je = collections.defaultdict(list)
    for exp in sorted(os.listdir(TELEGRAM)):
        d = os.path.join(TELEGRAM, exp)
        if not os.path.isdir(d) or exp not in kanal:
            continue
        for dp, dn, fn in os.walk(d):
            if "_ai_quarantine" in dp:
                continue
            for f in fn:
                m = re.search(r"(photo_\d+@\d\d-\d\d-(\d{4})_[\d-]+)", f)
                if m and int(m.group(2)) < 2022 and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    je[kanal[exp]].append((m.group(1), os.path.join(dp, f)))
    zeilen = {}
    for k, lst in je.items():
        # Duplikat-Exporte liefern denselben Post mehrfach — einmal genuegt
        einzeln = {}
        for name, p in lst:
            einzeln.setdefault(name, p)
        wahl = sorted(einzeln.items())
        rng.shuffle(wahl)
        zeilen[k] = [p for _, p in wahl[:MAX_KANAL]]
    return zeilen


def kanal_split(zeilen):
    """Ganze Kanaele auf val/test/train verteilen: groesste zuerst dorthin, wo
    relativ zum Soll am meisten fehlt — wie split.py, nur mit Kanaelen."""
    soll = {"test": 0.15, "val": 0.10, "train": 0.75}
    ges = sum(len(v) for v in zeilen.values())
    hat = {k: 0 for k in soll}
    out = {}
    for k in sorted(zeilen, key=lambda k: (-len(zeilen[k]), zlib.crc32(k.encode()))):
        s = max(soll, key=lambda s: (soll[s] * ges - hat[s]) / (soll[s] * ges))
        out[k] = s
        hat[s] += len(zeilen[k])
    return out, hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--worker", type=int, default=20)
    args = ap.parse_args()
    rng = random.Random(20260924)

    zeilen = []
    for z in manifest_v1():
        z = dict(z, quelle="wallhaven" if z["label"] == 0 else "generiert", hoch=True)
        zeilen.append(z)

    tg = telegram_zeilen(rng)
    wo, hat = kanal_split(tg)
    for k, ps in tg.items():
        for p in ps:
            zeilen.append(dict(wh_id="", model="telegram", label=0, split=wo[k], grp=k,
                               src=p, quelle="telegram", hoch=False))
    ki = collections.defaultdict(list)
    for dp, dn, fn in os.walk(AI_ONLY):
        exp = os.path.relpath(dp, AI_ONLY).split(os.sep)[0]
        for f in fn:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                ki[exp].append(os.path.join(dp, f))
    for exp, ps in ki.items():
        if exp in AI_TEST:
            sp = ["test"] * len(ps)
        else:
            # val aus dem Trainingskanal: die juengsten 10 % (nach Postdatum),
            # nur fuer Schwelle und fruehes Stoppen
            def datum(p):
                m = re.search(r"@(\d\d)-(\d\d)-(\d{4})", p)
                return (m.group(3), m.group(2), m.group(1)) if m else ("0", "0", "0")
            ps = sorted(ps, key=datum)
            cut = int(len(ps) * 0.9)
            sp = ["train"] * cut + ["val"] * (len(ps) - cut)
        for p, s in zip(ps, sp):
            zeilen.append(dict(wh_id="", model="ai_only", label=1, split=s, grp="ai:" + exp,
                               src=p, quelle="ai_only", hoch=False))

    tab = collections.Counter((z["quelle"], z["split"]) for z in zeilen)
    print(f"{'Quelle':<11}{'train':>9}{'val':>8}{'test':>8}")
    for q in ("wallhaven", "generiert", "telegram", "ai_only"):
        print(f"{q:<11}" + "".join(f"{tab[(q, s)]:>{9 if s == 'train' else 8}}"
                                   for s in ("train", "val", "test")))
    print(f"Telegram-Kanaele: {len(tg)}  -> "
          + ", ".join(f"{s} {sum(1 for k in wo if wo[k] == s)}" for s in ("train", "val", "test")))
    if args.plan:
        return 0

    # sortiert nach Pfad, damit die HDD am Stueck liest
    zeilen.sort(key=lambda z: z["src"].replace("/", "\\").lower())
    n = len(zeilen)
    off = np.zeros(n, np.int64)
    lng = np.zeros(n, np.int32)
    groesse = np.zeros((n, 4), np.int32)
    fehler = collections.Counter()
    t0, pos, puffer, naechst = time.time(), 0, {}, 0
    with open(os.path.join(DATEN, STAMM + ".bin"), "wb") as f, \
            ProcessPoolExecutor(max_workers=args.worker) as ex:
        for k, (i, erg, err) in enumerate(ex.map(
                _arbeit, ((i, z["src"], z["hoch"]) for i, z in enumerate(zeilen)),
                chunksize=32)):
            puffer[i] = (erg, err)
            while naechst in puffer:
                erg, err = puffer.pop(naechst)
                if erg is None:
                    fehler[err.split(":")[0] + (": zu klein" if "zu klein" in err else "")] += 1
                    off[naechst] = -1
                else:
                    b, w0, h0, w, h = erg
                    f.write(b)
                    off[naechst], lng[naechst] = pos, len(b)
                    groesse[naechst] = (w0, h0, w, h)
                    pos += len(b)
                naechst += 1
            if (k + 1) % 10000 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{n}  {(k+1)/el:.0f}/s  {pos/2**30:.1f} GB  "
                      f"Rest {(n-k-1)/((k+1)/el)/60:.0f} min", flush=True)
    np.save(os.path.join(DATEN, STAMM + "_off.npy"), off)
    np.save(os.path.join(DATEN, STAMM + "_len.npy"), lng)
    np.save(os.path.join(DATEN, STAMM + "_size.npy"), groesse)
    for i, z in enumerate(zeilen):
        z["i"] = i
    with open(os.path.join(DATEN, STAMM + "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(zeilen, f)
    print(f"\n{n - sum(fehler.values())} von {n} gepackt, {pos/2**30:.1f} GB, "
          f"{(time.time()-t0)/60:.0f} min. Verworfen: {dict(fehler)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
