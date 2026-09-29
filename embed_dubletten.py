r"""embed_dubletten.py — Telegram test images that (nearly) identically appear in a training source.

Proof by pixels, not features: every image as a 32x32 grayscale thumbnail, mean-free and normalized, cosine on
the GPU. A repost of the same image (downscaled, Telegram JPEG) stays near 1 (known pairs: 0.9999); different
images reach that only for almost flat images. Why not dHash (dubletten_kanaele.py): collides on dark images with
soft gradients. Why not the detector features: the same image perturbed vs. clean had cosine 0.82, the best
DIFFERENT image 0.79. --ausschliessen marks hits in the source manifest ("dublette: <channel>").

    python embed_dubletten.py <source> [--grenze 0.97] [--ausschliessen]
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from lesen import DATEN                          # noqa: E402

KANAELE = f"{DATASETS}/telegram_ki"


def mini(pfad):
    try:
        im = Image.open(pfad).convert("L").resize((32, 32), Image.BILINEAR)
        v = np.asarray(im, np.float32).ravel()
        v -= v.mean()
        n = np.linalg.norm(v)
        return v / n if n > 1e-3 else None
    except Exception:
        return None


def minis(pfade):
    with ThreadPoolExecutor(16) as ex:
        m = list(ex.map(mini, pfade))
    ok = [i for i, v in enumerate(m) if v is not None]
    return np.stack([m[i] for i in ok]).astype(np.float16), [pfade[i] for i in ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("quelle")
    ap.add_argument("--grenze", type=float, default=0.97)
    ap.add_argument("--ausschliessen", action="store_true",
                    help="Treffer im Manifest der Quelle als grund \"dublette: <kanal>\" markieren")
    a = ap.parse_args()
    cache = os.path.join(DATEN, "kanal_mini32.npz")
    kpf = sorted(glob.glob(os.path.join(KANAELE, "*", "telegram", "*.jpg")))
    if os.path.exists(cache):
        d = np.load(cache)
        K, kpf = d["m"], list(d["pfade"])
    else:
        K, kpf = minis(kpf)
        np.savez(cache, m=K, pfade=np.array(kpf))
    qpf = [e["datei"] for e in (json.loads(z) for z in open(f"{DATASETS}/{a.quelle}/manifest.jsonl",
                                                             encoding="utf-8")) if e.get("datei")]
    Q, qpf = minis(qpf)
    print(f"{len(kpf)} Kanalbilder, {len(qpf)} Bilder aus {a.quelle}", flush=True)
    Qg = torch.as_tensor(Q).cuda()
    best, arg = [], []
    for s in range(0, len(K), 8192):
        r = torch.as_tensor(K[s:s + 8192]).cuda() @ Qg.T
        m = r.max(1)
        best.append(m.values.float().cpu().numpy())
        arg.append(m.indices.cpu().numpy())
    best, arg = np.concatenate(best), np.concatenate(arg)
    kanal = np.array([p.split(os.sep)[-3] for p in kpf])
    erg, paare = {}, []
    for k in sorted(set(kanal)):
        m = np.flatnonzero((kanal == k) & (best >= a.grenze))
        erg[k] = [kpf[i] for i in m]
        paare += [(float(best[i]), kpf[i], qpf[arg[i]]) for i in m]
        print(f"  {k:<22} {int((kanal == k).sum()):>6}  Treffer >= {a.grenze}: {len(m):>5}", flush=True)
    print("Verteilung bester Kosinus je Kanalbild:",
          {q: round(float(np.quantile(best, q)), 3) for q in (0.5, 0.99, 0.999, 0.9999)})
    with open(os.path.join(DATEN, f"dubletten_{a.quelle}.json"), "w", encoding="utf-8") as fh:
        json.dump({"grenze": a.grenze, "treffer": erg, "paare": sorted(paare, reverse=True)}, fh)
    if a.ausschliessen and paare:
        raus = {q.lower(): k.split(os.sep)[-3] for _, k, q in paare}
        mf = f"{DATASETS}/{a.quelle}/manifest.jsonl"
        zeilen = [json.loads(z) for z in open(mf, encoding="utf-8")]
        for e in zeilen:
            if e.get("datei") and e["datei"].lower() in raus and not e.get("grund"):
                e["grund"] = f"dublette: {raus[e['datei'].lower()]}"
        with open(mf + ".tmp", "w", encoding="utf-8") as fh:
            for e in zeilen:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(mf + ".tmp", mf)
        print(f"{len(raus)} Bilder aus {a.quelle} als Dublette ausgeschlossen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
