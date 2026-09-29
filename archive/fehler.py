r"""fehler.py — woran scheitert der v2-Kopf auf dem echten KI-Kanal?

Auf den eigenen Generatoren 96 % Treffer, auf dem Telegram-KI-Kanal nur rund
die Haelfte. Bevor irgendetwas nachtrainiert wird, soll klar sein, WAS die
uebersehenen Bilder gemeinsam haben — davon haengt ab, ob neue Generatoren,
andere Stoerungen im Training oder eine andere Zurichtung helfen.

Verglichen werden uebersehene gegen erkannte KI-Bilder (Schwelle: 1 %
Fehlalarm auf den menschlichen Testkanaelen) nach
  - Groesse der Rohdatei (Pixel, Bytes je Pixel), JPEG-Qualitaet aus der
    Quantisierungstabelle
  - Postdatum (aus dem Dateinamen) — alte KI (2022/23) sieht anders aus
  - Inhalt: WD-Tags (Stil, Medium) — Anteil uebersehen je Tag
Dazu je ein Kontaktbogen uebersehener und erkannter Bilder.

Nur lesend; Ergebnis in daten/fehler.json und pruefung/fehler_*.jpg.

    python lab/detektor/fehler.py
"""
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from lesen import DATEN, Pack                    # noqa: E402

TAG = "quelle_clip_mid_dino_mid"
KANAL = "ai:ChatExport_2026-05-31 (1)"
AUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pruefung")


def jpeg_q(im):
    """Grobe JPEG-Qualitaet aus der Luminanz-Quantisierungstabelle (IJG-Skala)."""
    q = getattr(im, "quantization", None)
    if not q:
        return None
    t = np.array(q[0], float)
    basis = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                      14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                      18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                      49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)
    if len(t) != 64:
        return None
    s = np.median(t / basis * 100)          # Pillow liefert die Tabelle in natuerlicher Reihenfolge
    return float(100 - s / 2) if s <= 100 else float(5000 / s)


def roh(pfad):
    try:
        with Image.open(pfad) as im:
            return {"q": jpeg_q(im), "bytes": os.path.getsize(pfad), "fmt": im.format}
    except Exception:
        return {"q": None, "bytes": None, "fmt": None}


def datum(pfad):
    m = re.search(r"@(\d\d)-(\d\d)-(\d{4})", pfad)
    return f"{m.group(3)}-{m.group(2)}" if m else None


def bogen(pack, idx, s, name, n=40):
    kacheln = []
    for i in idx[:n]:
        b = pack.bild(i)
        b.thumbnail((220, 220))
        k = Image.new("RGB", (220, 236), (20, 20, 24))
        k.paste(b, ((220 - b.width) // 2, (220 - b.height) // 2))
        kacheln.append(k)
    sp = 8
    z = (len(kacheln) + sp - 1) // sp
    c = Image.new("RGB", (220 * sp, 236 * z), (20, 20, 24))
    for j, k in enumerate(kacheln):
        c.paste(k, ((j % sp) * 220, (j // sp) * 236))
    c.save(os.path.join(AUS, f"fehler_{name}.jpg"), quality=85)


def main():
    pack = Pack("v2")
    m = pack.m
    d = np.load(os.path.join(DATEN, f"score_v2_{TAG}_B.npy"))
    idx, s = d[:, 0].astype(int), d[:, 1]
    y = pack.label[idx]
    schwelle = float(np.quantile(s[y == 0], 0.99))
    ki = (pack.grp[idx] == KANAL)
    ii, ss = idx[ki], s[ki]
    weg = ss <= schwelle
    print(f"KI-Kanal: {len(ii)} Bilder, uebersehen {weg.sum()} ({weg.mean():.1%}) "
          f"bei Schwelle {schwelle:.2f} (1 % FA auf {int((y==0).sum())} Menschenbildern)")
    erg = {"schwelle": schwelle, "n": int(len(ii)), "uebersehen": float(weg.mean())}

    # --- Rohdateien
    pfade = [m[i]["src"] for i in ii]
    with ThreadPoolExecutor(16) as ex:
        r = list(ex.map(roh, pfade))
    w0, h0 = pack.size[ii, 0], pack.size[ii, 1]
    lang = np.maximum(w0, h0)
    px = (w0 * h0).astype(float)
    bpp = np.array([x["bytes"] / p if x["bytes"] else np.nan for x, p in zip(r, px)])
    q = np.array([x["q"] if x["q"] is not None else np.nan for x in r])
    seit = np.array([max(w0[j], h0[j]) / max(1, min(w0[j], h0[j])) for j in range(len(ii))])

    def vergleich(name, v, grenzen):
        zeilen = []
        for a, b in zip(grenzen[:-1], grenzen[1:]):
            k = (v >= a) & (v < b)
            if k.sum() >= 30:
                zeilen.append((f"{a:g}-{b:g}", int(k.sum()), float(weg[k].mean())))
        print(f"\n{name}:  (Anteil uebersehen je Bereich)")
        for lab, n, f in zeilen:
            print(f"  {lab:<14} n={n:>5}  uebersehen {f:.1%}")
        erg[name] = zeilen

    print(f"\nMedian uebersehen / erkannt:")
    for lab, v in (("lange Kante", lang), ("Bytes je Pixel", bpp), ("JPEG-q", q), ("Seitenverh.", seit),
                   ("Score", ss)):
        print(f"  {lab:<16} {np.nanmedian(v[weg]):>9.3f}  {np.nanmedian(v[~weg]):>9.3f}")
    vergleich("lange_kante", lang, [0, 600, 800, 1000, 1200, 1281, 5000])
    vergleich("bytes_je_pixel", bpp, [0, .1, .15, .2, .25, .3, .4, 10])
    vergleich("jpeg_q", q, [0, 80, 85, 88, 90, 92, 95, 101])
    vergleich("seitenverhaeltnis", seit, [1, 1.05, 1.3, 1.45, 1.6, 1.9, 10])

    mon = np.array([datum(p) or "?" for p in pfade])
    print("\nPostdatum (Quartal):")
    quart = np.array([f"{x[:4]}-Q{(int(x[5:])-1)//3+1}" if x != "?" else "?" for x in mon])
    erg["quartal"] = []
    for k in sorted(set(quart)):
        mk = quart == k
        if mk.sum() >= 20:
            print(f"  {k:<8} n={mk.sum():>5}  uebersehen {weg[mk].mean():.1%}")
            erg["quartal"].append((k, int(mk.sum()), float(weg[mk].mean())))

    # Trainingskanal zum Vergleich: wann postete der?
    tr = np.flatnonzero((pack.quelle == "ai_only") & (pack.split != "test") & pack.ok)
    qt = collections.Counter(f"{x[:4]}-Q{(int(x[5:])-1)//3+1}" for x in
                             (datum(m[i]["src"]) for i in tr) if x)
    print("\nTrainingskanal, Posts je Quartal:", dict(sorted(qt.items())))
    erg["train_quartal"] = dict(sorted(qt.items()))

    # --- Inhalt: WD-Tags auf den zugerichteten Bildern
    from imgutils.tagging import get_wd14_tags
    tags = []
    for j, i in enumerate(ii):
        _, gen, _ = get_wd14_tags(pack.bild(i), model_name="EVA02_Large", general_threshold=0.35)
        tags.append(set(gen))
        if j % 1000 == 0:
            print(f"  Tags {j}/{len(ii)}", flush=True)
    zaehl = collections.Counter(t for ts in tags for t in ts)
    zeilen = []
    for t, n in zaehl.items():
        if n < 60:
            continue
        mk = np.array([t in ts for ts in tags])
        zeilen.append((t, n, float(weg[mk].mean())))
    zeilen.sort(key=lambda z: -z[2])
    print(f"\nTags mit dem hoechsten Anteil uebersehen (Basis {weg.mean():.1%}):")
    for t, n, f in zeilen[:30]:
        print(f"  {t:<28} n={n:>5}  {f:.1%}")
    print("\n... und dem niedrigsten:")
    for t, n, f in zeilen[-15:]:
        print(f"  {t:<28} n={n:>5}  {f:.1%}")
    erg["tags"] = zeilen

    # --- Kontaktboegen: sicher uebersehen, knapp uebersehen, erkannt
    os.makedirs(AUS, exist_ok=True)
    rng = np.random.default_rng(0)
    o = np.argsort(ss)
    bogen(pack, ii[o[:40]], ss, "tiefste")
    bogen(pack, ii[rng.choice(np.flatnonzero(weg), 40, replace=False)], ss, "uebersehen_zufall")
    bogen(pack, ii[rng.choice(np.flatnonzero(~weg), 40, replace=False)], ss, "erkannt_zufall")
    with open(os.path.join(DATEN, "fehler.json"), "w", encoding="utf-8") as f:
        json.dump(erg, f, indent=1, ensure_ascii=False)
    print("\nFERTIG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
