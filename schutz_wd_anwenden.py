r"""schutz_wd_anwenden.py — apply the hits of a schutz_wd_figuren.py --probe run without re-tagging.

    python schutz_wd_anwenden.py --quellen civitai,civitai_stil,diffusiondb,midjourney_felix,nijijourney_p1atdev
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from lesen import DATEN                          # noqa: E402

ORDNER = {"civitai": f"{DATASETS}/civitai", "civitai_stil": f"{DATASETS}/civitai_stil",
          "diffusiondb": f"{DATASETS}/diffusiondb", "midjourney_felix": f"{DATASETS}/midjourney_felix",
          "nijijourney_p1atdev": f"{DATASETS}/nijijourney_p1atdev",
          "midjourney_v6_scrape": f"{DATASETS}/midjourney_v6_scrape", "aibooru": f"{DATASETS}/aibooru"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quellen", required=True)
    a = ap.parse_args()
    treffer = collections.defaultdict(set)
    for z in open(os.path.join(DATEN, "schutz_wd_figuren_treffer.jsonl"), encoding="utf-8"):
        t = json.loads(z)
        treffer[t["quelle"]].add(str(t["id"]))
    liste = open(os.path.join(DATEN, "gesperrt_quellen.txt"), "a", encoding="utf-8")
    for q in a.quellen.split(","):
        mf = os.path.join(ORDNER[q], "manifest.jsonl")
        zeilen = [json.loads(z) for z in open(mf, encoding="utf-8")]
        n = 0
        for i, e in enumerate(zeilen):
            if e.get("datei") and not e.get("grund") and str(e.get("id", e.get("msg"))) in treffer[q]:
                try:
                    os.remove(e["datei"])
                except OSError:
                    pass
                liste.write(e["datei"] + "\n")
                zeilen[i] = {k: e[k] for k in ("id", "msg", "sha256") if k in e}
                zeilen[i]["grund"] = "gesperrt: figur (wd)"
                n += 1
        with open(mf + ".tmp", "w", encoding="utf-8") as f:
            for e in zeilen:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(mf + ".tmp", mf)
        print(f"{q:<22} {n:>5} gesperrt (Protokoll: {len(treffer[q])})", flush=True)
    liste.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
