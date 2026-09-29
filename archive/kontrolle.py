r"""kontrolle.py — nutzt ein trainierter Kopf das Seitenverhaeltnis?

Nach der Normierung trennt das Seitenverhaeltnis allein noch mit AUC 0,997
(leak.py): Generatoren rechnen in Buckets (Vielfache von 64 px), echte
Wallpaper haben beliebige Formate. Beide Rueckgrate schneiden deshalb einen
quadratischen Mittelausschnitt aus — CLIP von sich aus, WD-EVA02 seit
2026-09-24 ebenfalls. Ob das Leck damit wirklich zu ist, prueft diese
Gegenprobe:

Fuer jedes Negativ ist bekannt, in welchem Bucket seine Positiven erzeugt
wurden (images.bucket_w/h). Manche Negative haben zufaellig fast exakt ein
Bucket-Seitenverhaeltnis, andere liegen weit daneben. HAETTE der Kopf gelernt
"Bucket-Format = KI", kaemen ihm die bucket-nahen Negative verdaechtiger vor:
hoehere Scores, mehr Fehlalarme. Gemessen wird beides.

    python lab/detektor/kontrolle.py score_clip_l14_logreg
"""
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from pfade import V1_DB  # noqa: E402
from lesen import DATEN, Pack                    # noqa: E402

DB = V1_DB


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "score_clip_l14_logreg"
    stamm = sys.argv[sys.argv.index("--pack") + 1] if "--pack" in sys.argv else "norm512"
    pack = Pack(stamm)
    if stamm != "norm512":
        # v2: die Bucket-Korrelation unten ist ohnehin verfaelscht (Genre) — nur die Eingriffe
        kopf = sys.argv[sys.argv.index("--kopf") + 1]
        if "--eingriff" in sys.argv:
            eingriff(pack, name, kopf=kopf)
        if "--robust" in sys.argv:
            robust(pack, name, kopf=kopf)
        return 0
    te_s = np.load(os.path.join(DATEN, name + ".npy"))
    idx, s = te_s[:, 0].astype(int), te_s[:, 1]

    con = sqlite3.connect(DB, timeout=60)
    bucket = {w: bw / bh for w, bw, bh in con.execute(
        "SELECT wh_id, bucket_w, bucket_h FROM images WHERE bucket_w IS NOT NULL")}
    con.close()

    neg = pack.model[idx] == "mensch"
    w, h = pack.size[idx, 2].astype(float), pack.size[idx, 3].astype(float)
    asp = w / h
    b = np.array([bucket.get(x, np.nan) for x in pack.wh_id[idx]])
    d = np.abs(np.log(asp / b))          # relative Abweichung vom Bucket

    k = neg & np.isfinite(d)
    sn, dn = s[k], d[k]
    thr1 = np.quantile(sn, 0.99)         # Schwelle bei 1 % Fehlalarm
    thr5 = np.quantile(sn, 0.95)
    nah = dn < np.quantile(dn, 0.25)
    fern = dn > np.quantile(dn, 0.75)

    from scipy.stats import spearmanr
    r, p = spearmanr(sn, dn)
    print(f"{name}: {k.sum()} Negative im Test")
    print(f"  Abweichung vom Bucket (|log|): Median {np.median(dn):.4f}, "
          f"nahes Viertel < {np.quantile(dn,0.25):.4f}, fernes > {np.quantile(dn,0.75):.4f}")
    print(f"  {'':<22}{'Score Median':>14}{'FA bei 1 %':>12}{'FA bei 5 %':>12}")
    for lab, m in (("bucket-nah (25 %)", nah), ("bucket-fern (25 %)", fern)):
        print(f"  {lab:<22}{np.median(sn[m]):>14.3f}"
              f"{(sn[m] > thr1).mean():>12.3%}{(sn[m] > thr5).mean():>12.3%}")
    print(f"  Spearman(Score, Abweichung) = {r:+.4f}  (p = {p:.3g})")
    print("  ACHTUNG, verfaelscht: bucket-nah sind vor allem QUADRATISCHE Bilder")
    print("  (Figuren), bucket-fern breite Wallpaper (Landschaft). Ein Unterschied")
    print("  kann am Genre liegen. Beweiskraeftig ist erst der Eingriff unten.")
    if "--eingriff" in sys.argv:
        eingriff(pack, name)
    if "--robust" in sys.argv:
        robust(pack, name)
    return 0


def robust(pack, name, n=1200, kopf=None):
    """Haelt die Trennung, wenn BEIDE Klassen identisch gestoert werden?

    Der Eingriff hat gezeigt: der Kopf reagiert extrem auf Resampling-Spuren.
    Die Negative stehen alle auf exakt 1024 px, norm.py hat sie also um genau
    den Faktor 2 verkleinert — ein ganzzahliges, sehr regelmaessiges Muster;
    die Positiven um krumme Faktoren (1344 -> 512 = 2,625). Lernt der Kopf
    das, bricht er auf echten Bildern ein: Telegram rekomprimiert (q<=92) und
    skaliert beliebig.

    Hier bekommen Mensch und Maschine GLEICHE Stoerungen. Faellt die AUC
    stark, hing die Leistung an Verarbeitungsspuren statt an der Herkunft."""
    import io
    from PIL import Image
    from kandidat import embedding, _kopf, logit
    k = _kopf(kopf or name.replace("score_", "").replace("_logreg", ""))  # _mlp bleibt
    rng = np.random.default_rng(2)
    te = pack.wo(split="test")
    neg = rng.choice(te[pack.label[te] == 0], n, replace=False)
    pos = rng.choice(te[pack.label[te] == 1], n, replace=False)
    idx = np.r_[neg, pos]
    y = np.r_[np.zeros(n), np.ones(n)]

    def score_von(bild):
        x = np.concatenate([embedding(bild, r) for r in k["rueckgrate"]])
        return logit(x, k)

    def gestoert(i, f, q):
        b = pack.bild(i)
        if f != 1.0:
            w, h = b.size
            b = b.resize((round(w * f), round(h * f)), Image.LANCZOS)
        buf = io.BytesIO()
        b.save(buf, "JPEG", quality=q, subsampling=0 if q >= 95 else 2)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    from sklearn.metrics import roc_auc_score
    print(f"\nRobustheit: {n} Mensch + {n} Maschine aus test, beide gleich gestoert")
    for lab, f, q in (("unveraendert (q95)", 1.0, 95),
                      ("nur JPEG q90", 1.0, 90),
                      ("x0,9 + q95", 0.9, 95),
                      ("x0,9 + q90", 0.9, 90),
                      ("x0,75 + q85", 0.75, 85)):
        s = np.array([score_von(gestoert(i, f, q)) for i in idx])
        thr = np.quantile(s[:n], 0.99)
        print(f"  {lab:<20} AUC {roc_auc_score(y, s):.4f}   "
              f"Treffer bei 1 % FA {(s[n:] > thr).mean():.3f}")


def eingriff(pack, name, n=1500, kopf=None):
    """Der saubere Test: das eine verdaechtige Merkmal gezielt veraendern.

    Nach dem Mittelausschnitt sieht CLIP vom Seitenverhaeltnis nur noch die
    Laenge der kurzen Kante: 288 beim 16:9-Negativ, 293 beim Positiv aus dem
    Bucket 1344x768 — 1,7 % Massstab. Hier werden Negative um genau diesen
    Faktor hoch- und Positive herunterskaliert. Nutzt der Kopf den Massstab,
    wandern die Negative Richtung KI und die Positiven Richtung Mensch. Ein
    Eingriff statt einer Korrelation — kein Genre kann ihn verfaelschen."""
    import io
    from PIL import Image
    from kandidat import embedding, _kopf, logit
    from norm import QUALITAET
    k = _kopf(kopf or name.replace("score_", "").replace("_logreg", ""))  # _mlp bleibt
    rng = np.random.default_rng(1)

    def score_von(bild):
        x = np.concatenate([embedding(bild, r) for r in k["rueckgrate"]])
        return logit(x, k)

    def skaliert(i, f):
        b = pack.bild(i)
        w, h = b.size
        b = b.resize((round(w * f), round(h * f)), Image.LANCZOS)
        buf = io.BytesIO()
        b.save(buf, "JPEG", quality=QUALITAET, subsampling=0)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    te = pack.wo(split="test")
    kurz = np.minimum(pack.size[te, 2], pack.size[te, 3])
    # v1: Positive aus dem Bucket 1344x768 enden bei 293; v2 (Hochlade-Simulation
    # 1344 -> 1280 davor) bei 292. Das trifft, was in den Daten vorkommt.
    kp = 293 if (kurz[pack.quelle[te] == "generiert"] == 293).any() else 292
    neg = te[(pack.model[te] == "mensch") & (kurz == 288)]
    pos = te[(pack.quelle[te] == "generiert") & (kurz == kp)]
    neg = rng.choice(neg, min(n, len(neg)), replace=False)
    pos = rng.choice(pos, min(n, len(pos)), replace=False)
    print(f"\nEingriff: {len(neg)} Negative (kurze Kante 288) x {kp}/288, "
          f"{len(pos)} Positive ({kp}) x 288/{kp}")
    for lab, idx, f in (("Negative", neg, kp / 288), ("Positive", pos, 288 / kp)):
        vor = np.array([score_von(skaliert(i, 1.0)) for i in idx])
        nach = np.array([score_von(skaliert(i, f)) for i in idx])
        dz = nach - vor
        print(f"  {lab:<9} Score vorher {np.median(vor):+.3f}  nachher "
              f"{np.median(nach):+.3f}   Verschiebung Median {np.median(dz):+.4f}, "
              f"Mittel {dz.mean():+.4f} (Streuung {dz.std():.4f})")
    print("  Leck hiesse: Negative deutlich nach oben, Positive deutlich nach unten.")


if __name__ == "__main__":
    sys.exit(main())
