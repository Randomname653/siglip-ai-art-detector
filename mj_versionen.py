r"""mj_versionen.py — hit rate per Midjourney version (parsed from the prompt) on the Telegram MJ channels.

Channels: midjourney_gallery, mj_prompts_daily, leonardo. Each head at its own stored threshold.

    python mj_versionen.py v3 p8x p8sx
"""
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from lesen import DATEN, kanal_X                 # noqa: E402

KANAELE = ("midjourney_gallery", "mj_prompts_daily", "leonardo")


def gruppe(v):
    v = (v or "").lower().strip()
    if not v:
        return "ohne Angabe"
    if "niji" in v:
        return "niji"
    if v.startswith("test"):
        return "test/testp"
    m = re.match(r"v?(\d+)", v)
    return f"v{m.group(1)}" if m else v


def main():
    tags = sys.argv[1:]
    X, gr = [], []
    for k in KANAELE:
        ver = {}
        for z in open(f"{DATASETS}/telegram_ki/{k}/manifest.jsonl", encoding="utf-8"):
            e = json.loads(z)
            if e.get("datei"):
                ver[e["datei"].lower()] = e.get("version", "")
        Xk, pfade = kanal_X(k)
        X.append(Xk)
        gr += [gruppe(ver.get(str(p).lower())) for p in pfade]
    X, gr = np.concatenate(X), np.array(gr)
    erg = {}
    for t in tags:
        k = koepfe_von(t)
        s = np.concatenate([bewerte(X[i:i + 2000], k) for i in range(0, len(X), 2000)]) > float(k[0]["schwelle_1"])
        erg[t] = {g: (int((gr == g).sum()), float(s[gr == g].mean())) for g in sorted(set(gr))}
    ordnung = sorted(set(gr), key=lambda g: (g == "ohne Angabe", g))
    print(f"{'Version':<14}{'n':>7}" + "".join(f"{t:>9}" for t in tags))
    for g in ordnung:
        print(f"{g:<14}{erg[tags[0]][g][0]:>7}" + "".join(f"{erg[t][g][1]:>8.1%} " for t in tags))
    with open(os.path.join(DATEN, "mj_versionen_p.json"), "w", encoding="utf-8") as f:
        json.dump(erg, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
