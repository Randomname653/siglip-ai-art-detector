r"""messlatte.py — scoring functions for the seven production detectors.

The existing pipeline used seven models (three in tier 1, four deepghs variants in tier 2): deepghs
mobilenetv3_sce(_dist)(_v1), caformer_s36_plus_sce, caformer_s36_sce_v1, legekka/AI-Anime-Image-Detector-ViT,
saltacc/anime-ai-detect. Each detector gets the same image as our heads and applies its own preprocessing.
Used by bench.py and kombination.py.

    python messlatte.py [--split val]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import DATEN, Pack                    # noqa: E402

# (Schluessel, Backend, Name) — wie AI_MODELS_T1 + AI_MODELS_T2 in imagesort.py
MODELLE = [
    ("deepghs:mobilenetv3_sce_dist_v1", "deepghs", "mobilenetv3_sce_dist_v1"),
    ("hf:legekka", "hf", "legekka/AI-Anime-Image-Detector-ViT"),
    ("hf:saltacc", "hf", "saltacc/anime-ai-detect"),
    ("deepghs:mobilenetv3_sce", "deepghs", "mobilenetv3_sce"),
    ("deepghs:mobilenetv3_sce_dist", "deepghs", "mobilenetv3_sce_dist"),
    ("deepghs:caformer_s36_plus_sce", "deepghs", "caformer_s36_plus_sce"),
    ("deepghs:caformer_s36_sce_v1", "deepghs", "caformer_s36_sce_v1"),
]


AUG = False


def score_hf(pack, idx, name, batch=64):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    proc = AutoImageProcessor.from_pretrained(name)
    mod = AutoModelForImageClassification.from_pretrained(name).cuda().eval().half()
    # Wie imagesort.py: den KI-Index ueber den Labelnamen suchen, nicht fest
    # verdrahten — saltacc hat "ai" auf 0, legekka auf 1.
    ai = [int(i) for i, l in mod.config.id2label.items()
          if l.lower() in ("ai", "artificial", "fake")]
    assert len(ai) == 1, mod.config.id2label
    ai = ai[0]
    out = np.zeros(len(idx), np.float32)
    with ThreadPoolExecutor(8) as ex:
        for s in range(0, len(idx), batch):
            bilder = list(ex.map(lambda i: pack.bild(i, aug=AUG), idx[s:s + batch]))
            x = proc(images=bilder, return_tensors="pt")["pixel_values"]
            with torch.no_grad():
                p = mod(pixel_values=x.cuda().half()).logits.float().softmax(-1)
            out[s:s + len(bilder)] = p[:, ai].cpu().numpy()
    return out


def score_deepghs(pack, idx, name):
    from imgutils.validate.aicheck import get_ai_created_score

    def eins(i):
        return float(get_ai_created_score(pack.bild(i, aug=AUG), model_name=name))
    # ONNX gibt die GIL frei, Threads bringen hier viel (siehe imagesort.py).
    with ThreadPoolExecutor(8) as ex:
        return np.array(list(ex.map(eins, idx)), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--aug", action="store_true",
                    help="dieselbe Stoerung wie beim Training (norm.stoere)")
    args = ap.parse_args()
    global AUG
    AUG = args.aug
    tag = args.split + ("_aug" if args.aug else "")

    pack = Pack()
    idx = pack.wo(split=args.split)
    print(f"{args.split}: {len(idx)} Bilder "
          f"({int((pack.label[idx]==0).sum())} Mensch, "
          f"{int((pack.label[idx]==1).sum())} Maschine)", flush=True)
    np.save(os.path.join(DATEN, f"mess_{tag}_idx.npy"), idx)

    for key, backend, name in MODELLE:
        ziel = os.path.join(DATEN, f"mess_{tag}_{key.replace(':', '_')}.npy")
        if os.path.exists(ziel):
            print(f"  {key}: schon da", flush=True)
            continue
        t0 = time.time()
        s = (score_hf if backend == "hf" else score_deepghs)(pack, idx, name)
        np.save(ziel, s)
        print(f"  {key}: {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
