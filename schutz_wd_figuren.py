r"""schutz_wd_figuren.py — apply the character rule to the IMAGE (WD character tags), not only to text.

The character/franchise list used to be matched against captions and booru tags only; channels without text and
web sources without character tags were protected by the tag block alone. The WD tagger recognizes characters
in the image. Rule as agreed: characters/franchises only for non-general images. Every image whose stored
rating is not "general" (or missing) is re-tagged. Hits: lab file deleted, manifest line reduced to "gesperrt",
path appended to daten/gesperrt_quellen.txt. --probe only logs (schutz_wd_figuren_treffer.jsonl).

    python schutz_wd_figuren.py [--nur source1,source2] [--probe]
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from lesen import DATEN                                               # noqa: E402
from schutz import SCHWELLE, figurenregel_gilt, gesperrt_text, gesperrt_wd   # noqa: E402

LISTE = os.path.join(DATEN, "gesperrt_quellen.txt")
QUELLEN = [f"{DATASETS}/civitai", f"{DATASETS}/civitai_stil", f"{DATASETS}/diffusiondb",
           f"{DATASETS}/midjourney_felix", f"{DATASETS}/nijijourney_p1atdev", f"{DATASETS}/midjourney_v6_scrape",
           f"{DATASETS}/aibooru"] + sorted(glob.glob(f"{DATASETS}/telegram_ki/*"))
FIGUR_SCHWELLE = 0.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nur", default="")
    ap.add_argument("--probe", action="store_true",
                    help="nur protokollieren (schutz_wd_figuren_treffer.jsonl), nichts loeschen oder sperren")
    a = ap.parse_args()
    from imgutils.tagging import get_wd14_tags
    nur = set(a.nur.split(",")) if a.nur else None
    gesamt = {"geprueft": 0, "gesperrt": 0}
    # Protokoll je Treffer (ohne Bild): Quelle, Rating, ausloesende Figur — zum Nachpruefen der Regel
    protokoll = open(os.path.join(DATEN, "schutz_wd_figuren_treffer.jsonl"), "a", encoding="utf-8")
    t0 = time.time()
    for q in QUELLEN:
        name = os.path.basename(q)
        mf = os.path.join(q, "manifest.jsonl")
        if (nur and name not in nur) or not os.path.exists(mf):
            continue
        zeilen = [json.loads(z) for z in open(mf, encoding="utf-8")]
        def pfad(e):     # Telegram-Kanaele: "datei"; AIBooru-Repost-Bot: "telegram" (+ "original")
            return e.get("datei") or e.get("telegram")
        pruef = [i for i, e in enumerate(zeilen) if pfad(e) and not e.get("grund")
                 and str(e.get("rating") or "").lower() not in ("general", "g")
                 and os.path.exists(pfad(e))]

        def eins(i):
            try:
                r, g, ch = get_wd14_tags(pfad(zeilen[i]), model_name="EVA02_Large",
                                         general_threshold=SCHWELLE, character_threshold=FIGUR_SCHWELLE)
                return i, max(r, key=r.get), g, ch
            except Exception:
                return i, None, {}, {}

        gesperrt = []
        with ThreadPoolExecutor(4) as ex:
            for i, rating, g, ch in ex.map(eins, pruef):
                e = zeilen[i]
                if rating is None:
                    continue
                if not e.get("rating"):
                    e["rating"] = rating
                treffer = gesperrt_text(" ".join(ch).replace("_", " ")) if figurenregel_gilt(rating) else []
                if gesperrt_wd(g):
                    treffer = treffer + ["tag"]
                if treffer:
                    protokoll.write(json.dumps({"quelle": name, "rating": rating, "treffer": treffer,
                                                "figuren": {k: round(float(v), 2) for k, v in ch.items()},
                                                "id": e.get("id", e.get("msg"))}, ensure_ascii=False) + "\n")
                    protokoll.flush()
                    gesperrt += [e[f] for f in ("datei", "telegram", "original") if e.get(f)]
                    zeilen[i] = {k: e[k] for k in ("id", "msg", "sha256", "kanal", "art") if k in e}
                    zeilen[i]["grund"] = "gesperrt: figur (wd)"
        if a.probe:
            gesamt["geprueft"] += len(pruef)
            gesamt["gesperrt"] += len(gesperrt)
            print(f"{name:<24} geprueft {len(pruef):>6}  wuerde sperren {len(gesperrt):>5}   "
                  f"({(time.time()-t0)/60:.0f} min)", flush=True)
            continue
        for p in gesperrt:
            try:
                os.remove(p)
            except OSError:
                pass
        with open(LISTE, "a", encoding="utf-8") as f:
            for p in gesperrt:
                f.write(p + "\n")
        with open(mf + ".tmp", "w", encoding="utf-8") as f:
            for e in zeilen:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(mf + ".tmp", mf)
        gesamt["geprueft"] += len(pruef)
        gesamt["gesperrt"] += len(gesperrt)
        print(f"{name:<24} geprueft {len(pruef):>6}  gesperrt {len(gesperrt):>5}   "
              f"({(time.time()-t0)/60:.0f} min)", flush=True)
    print(f"FERTIG: {gesamt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
