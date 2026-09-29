r"""dubletten_kanaele.py — Bilder einer Trainingsquelle, die auch in einem Telegram-Testkanal stecken.

Die Telegram-KI-Kanaele sind reiner Test. Holt eine Web-Quelle dieselben Bilder (Repost,
z. B. DiffusionDB-Bild im Kanal "AI art.random"), waere der Test geschoent. Vergleich per
dHash (64 bit, 9x8 Graustufen) — robust gegen Verkleinern und JPEG; Treffer bei Hamming <= 6.
Treffer bekommen im Manifest der Quelle grund "dublette: <kanal>" und fallen damit aus dem Pack.

ACHTUNG (2026-09-27, DiffusionDB): 583 Treffer, in einer Stichprobe von 10 Paaren KEIN echtes
Duplikat — dHash kollidiert bei dunklen Bildern mit weichen Verlaeufen. Als Vorfilter harmlos
(verwirft nur etwas zu viel), als Nachweis untauglich: echter Abgleich per Embedding in
embed_dubletten.py (32x32-Pixel-Miniaturen).

    python lab/detektor/dubletten_kanaele.py diffusiondb
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from lesen import DATEN                          # noqa: E402

KANAELE = f"{DATASETS}/telegram_ki"
GRENZE = 6


def dhash(pfad):
    try:
        im = Image.open(pfad).convert("L").resize((9, 8), Image.LANCZOS)
        a = np.asarray(im, np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        return int(np.packbits(bits).view(">u8")[0])
    except Exception:
        return None


def hashes(pfade):
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(dhash, pfade))


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def main():
    quelle = sys.argv[1]
    mf = f"{DATASETS}/{quelle}/manifest.jsonl"
    cache = os.path.join(DATEN, "kanal_dhash.npz")
    kpf = sorted(glob.glob(os.path.join(KANAELE, "*", "telegram", "*.jpg")))
    if os.path.exists(cache) and len(np.load(cache)["pfade"]) == len(kpf):
        d = np.load(cache)
        kh, kpf = d["h"], list(d["pfade"])
    else:
        h = hashes(kpf)
        ok = [i for i, x in enumerate(h) if x is not None]
        kh = np.array([h[i] for i in ok], np.uint64)
        kpf = [kpf[i] for i in ok]
        np.savez(cache, h=kh, pfade=np.array(kpf))
    print(f"{len(kh)} Kanalbilder gehasht", flush=True)

    zeilen = [json.loads(z) for z in open(mf, encoding="utf-8")]
    idx = [i for i, e in enumerate(zeilen) if e.get("datei") and not e.get("grund")]
    qh = hashes([zeilen[i]["datei"] for i in idx])
    treffer = 0
    for j, i in enumerate(idx):
        if qh[j] is None:
            continue
        d = popcount64(np.bitwise_xor(kh, np.uint64(qh[j])))
        k = int(d.argmin())
        if d[k] <= GRENZE:
            kanal = kpf[k].split(os.sep)[-3]
            zeilen[i]["grund"] = f"dublette: {kanal}"
            zeilen[i]["dublette_von"] = kpf[k]
            treffer += 1
    with open(mf + ".tmp", "w", encoding="utf-8") as f:
        for e in zeilen:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(mf + ".tmp", mf)
    print(f"{quelle}: {len(idx)} Bilder geprueft, {treffer} Dubletten in Telegram-Kanaelen -> ausgeschlossen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
