r"""kanal2.py — der zweite echte KI-Kanal als unabhaengiger Test.

ChatExport_2026-09-25 ist ein Repost-Bot von aibooru.online: je Nachricht das
Telegram-Foto (echte Telegram-Kompression) plus Link auf den AIBooru-Post, in
der Folgenachricht die Originaldatei files/<post-id>.<ext>. Keine seiner Posts
ist im Training — aibooru.py holt nur IDs unter 150.000, der Kanal beginnt
bei 152.533.

Schritt 1 (--liste): Paare aus dem HTML lesen, die AIBooru-Tags der Posts
holen, schutz.py auf Booru-Tags anwenden (gesperrte Posts verschwinden aus
jeder weiteren Verwendung; die Dateien des Users bleiben unberuehrt).
Schritt 2 (--score): gegen die vier menschlichen Testkanaele aus v2 messen
  - Telegram-Fotos durch den Kopf (so kommen sie im Einsatz an)
  - Originale durch UNSERE Hochlade-Simulation -> prueft, ob die Simulation
    der echten Telegram-Kompression entspricht
  - die sieben Produktionsdetektoren auf den Telegram-Fotos

    python lab/detektor/kanal2.py --liste
    python lab/detektor/kanal2.py --score --kopf v2_quelle_clip_mid_dino_mid
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from lesen import DATEN, Pack                    # noqa: E402
from schutz import figurenregel_gilt, gesperrt_tags, gesperrt_text   # noqa: E402

# der Download brach einmal ab und wurde neu gestartet: beide Ordner lesen,
# je AIBooru-Post zaehlt ein Foto
# seit 2026-09-25 abends nach _zum_loeschen verschoben (siehe <DATASETS>\QUELLEN.md)
KANAELE = [f"{DATASETS}/_zum_loeschen/ChatExport_2026-09-25",
           f"{DATASETS}/_zum_loeschen/ChatExport_2026-09-25 (1)"]
LISTE = os.path.join(DATEN, "kanal2.json")


def paare():
    """[(post_id, photo_pfad, datei_pfad)], je Post einmal, ueber alle Export-Ordner."""
    gesehen, aus = set(), []
    for k in KANAELE:
        if os.path.isdir(k):
            for i, f, d in paare_in(k):
                if i not in gesehen:
                    gesehen.add(i)
                    aus.append((i, f, d))
    return aus


def paare_in(KANAL):
    """[(post_id, photo_pfad, datei_pfad)] eines Exports in Nachrichtenreihenfolge."""
    def nr(p):
        m = re.search(r"messages(\d*)\.html$", p)
        return int(m.group(1) or 1)
    seiten = sorted(glob.glob(os.path.join(KANAL, "messages*.html")), key=nr)
    foto, aus, dateien = None, [], {}
    for s in seiten:
        t = open(s, encoding="utf-8").read()
        for m in re.finditer(r'href="(photos/[^"]+)"|aibooru\.online/posts/(\d+)|href="(files/[^"]+)"', t):
            if m.group(1) and "_thumb" not in m.group(1):
                foto = m.group(1)
            elif m.group(2) and foto:
                aus.append((int(m.group(2)), os.path.join(KANAL, foto)))
                foto = None
            elif m.group(3) and "_thumb" not in m.group(3):
                stamm = os.path.splitext(os.path.basename(m.group(3)))[0]
                if stamm.isdigit():
                    dateien[int(stamm)] = os.path.join(KANAL, m.group(3))
    return [(i, f, dateien.get(i)) for i, f in aus if os.path.exists(f)]


def liste():
    import requests
    P = paare()
    print(f"{len(P)} Telegram-Fotos mit AIBooru-Post, {sum(1 for p in P if p[2])} mit Original", flush=True)
    ids = sorted({p[0] for p in P})
    info = {}
    s = requests.Session()
    s.headers["User-Agent"] = "ai-art-detector-lab/0.1 (research)"
    for a in range(0, len(ids), 100):
        teil = ids[a:a + 100]
        r = s.get("https://aibooru.online/posts.json",
                  params={"tags": "id:" + ",".join(map(str, teil)), "limit": 200}, timeout=90)
        for p in r.json():
            info[p["id"]] = p
        time.sleep(0.5)
    zeilen, gesperrt, fehlt = [], 0, 0
    for i, foto, datei in P:
        p = info.get(i)
        if p is None:
            fehlt += 1          # geloescht oder nicht mehr oeffentlich: Tags unbekannt -> nicht verwenden
            continue
        figuren = p.get("tag_string_character", "") + " " + p.get("tag_string_copyright", "")
        if gesperrt_tags(p.get("tag_string", "")) or (figurenregel_gilt(p.get("rating")) and gesperrt_text(figuren)):
            gesperrt += 1
            continue
        zeilen.append({"id": i, "foto": foto, "datei": datei, "rating": p.get("rating"),
                       "model": p.get("tag_string_model", ""), "meta": p.get("tag_string_meta", "")})
    with open(LISTE, "w", encoding="utf-8") as f:
        json.dump(zeilen, f, indent=0, ensure_ascii=False)
    print(f"verwendbar {len(zeilen)}, gesperrt {gesperrt}, ohne Tags {fehlt} -> {LISTE}")


def score(kopfname):
    from PIL import Image
    from kandidat import _kopf, logit
    from norm import normiere
    from pruefe import stapelrechner
    from civitai import hochladen_bytes
    from schutz import SCHWELLE, gesperrt_wd
    from imgutils.tagging import get_wd14_tags
    from sklearn.metrics import roc_auc_score, roc_curve

    Z = json.load(open(LISTE, encoding="utf-8"))
    # zweite Sicherung: WD-Tags auf dem Foto
    Z = [z for z in Z if not gesperrt_wd(get_wd14_tags(z["foto"], model_name="EVA02_Large",
                                                         general_threshold=SCHWELLE)[1])]
    print(f"{len(Z)} Posts nach WD-Sperre", flush=True)
    k = _kopf(kopfname)
    rg = [str(r) for r in k["rueckgrate"]]
    rechner = stapelrechner(rg)

    def bild_foto(z):
        return Image.open(io.BytesIO(normiere(z["foto"])[0])).convert("RGB")

    def bild_sim(z):
        with Image.open(z["datei"]) as im:
            daten, _ = hochladen_bytes(im.convert("RGB"))
        return Image.open(io.BytesIO(normiere(io.BytesIO(daten))[0])).convert("RGB")

    from concurrent.futures import ThreadPoolExecutor
    erg_s = {}
    with ThreadPoolExecutor(8) as ex:
        for art, fn, zz in (("foto", bild_foto, Z), ("simuliert", bild_sim, [z for z in Z if z["datei"]])):
            s = []
            for a in range(0, len(zz), 32):
                bs = list(ex.map(fn, zz[a:a + 32]))
                X = np.concatenate([rechner[r](bs, ex) for r in rg], 1)
                s += [logit(x, k) for x in X]
            erg_s[art] = (np.array([z["id"] for z in zz]), np.array(s))

    # menschliche Seite: die vier Testkanaele, derselbe Kopf (Scores aus train2)
    tag = kopfname.replace("v2_", "")
    d = np.load(os.path.join(DATEN, f"score_v2_{tag}_B.npy"))
    pack = Pack("v2")
    mensch = d[pack.label[d[:, 0].astype(int)] == 0, 1]

    def kenn(neg, pos):
        y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        f, t, _ = roc_curve(y, np.r_[neg, pos])
        return roc_auc_score(y, np.r_[neg, pos]), float(np.interp(0.01, f, t)), float(np.interp(0.05, f, t))

    erg = {"n": len(Z), "kopf": kopfname}
    schw = float(k["schwelle_1"])
    for art, (ids, s) in erg_s.items():
        a, t1, t5 = kenn(mensch, s)
        erg[art] = {"n": len(s), "auc": a, "tpr1": t1, "tpr5": t5, "tr_val_schwelle": float((s > schw).mean())}
        print(f"  neu, {art:<10} n={len(s):>5}  AUC {a:.4f}  @1% {t1:.1%}  @5% {t5:.1%}  "
              f"(val-Schwelle: {erg[art]['tr_val_schwelle']:.1%})", flush=True)
    np.save(os.path.join(DATEN, f"kanal2_{tag}.npy"), np.c_[erg_s["foto"][0], erg_s["foto"][1]])

    # Produktion auf den Telegram-Fotos, Menschen aus telegram4 (ohne Vorschaubilder)
    import messlatte

    class Roh:
        def bild(self, i, aug=False):
            with Image.open(Z[i]["foto"]) as im:
                return im.convert("RGB")
    tb = np.load(os.path.join(DATEN, "t4_idx.npy"))
    keep = pack.ok[tb] & (pack.label[tb] == 0)
    for key, backend, name in messlatte.MODELLE:
        ziel = os.path.join(DATEN, f"kanal2_prod_{key.replace(':', '_')}.npy")
        if os.path.exists(ziel) and len(np.load(ziel)) == len(Z):
            s = np.load(ziel)
        else:
            f = messlatte.score_hf if backend == "hf" else messlatte.score_deepghs
            s = np.asarray(f(Roh(), np.arange(len(Z)), name))
            np.save(ziel, s)
        neg = np.load(os.path.join(DATEN, f"t4_{key.replace(':', '_')}.npy"))[keep]
        a, t1, t5 = kenn(neg, s)
        erg["prod_" + key] = {"auc": a, "tpr1": t1, "tpr5": t5}
        print(f"  Prod {key:<34} AUC {a:.4f}  @1% {t1:.1%}  @5% {t5:.1%}", flush=True)
    with open(os.path.join(DATEN, f"kanal2_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(erg, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--kopf", default="v2_quelle_clip_mid_dino_mid")
    args = ap.parse_args()
    if args.liste:
        liste()
    if args.score:
        score(args.kopf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
