r"""vae_rekon.py — "aligned" AI examples: human training images passed through a generator's autoencoder.

Idea (Rajan et al. ICLR 2025 "Aligned datasets"; B-Free CVPR 2025): encode a HUMAN image with a latent-diffusion
VAE and decode it again, without any denoising. The result keeps content and style of the human original; only the
generator's decoder trace differs. Labelled AI, next to its untouched original (labelled human), the head can no
longer use "looks polished / modern style" as a cue and has to use the generation trace — the main source of our
false positives. Pilot (vae_probe.py, 29.09.): a linear probe on frozen SigLIP features separates original and
reconstruction with AUC 0.973 (Flux VAE) / 0.994 (SDXL VAE), while p8sx flags only 3-4 % of reconstructions.

Leaks avoided on purpose:
  - size: images are reflect-padded to a multiple of 16 and cropped back -> exactly the original size;
  - compression: stored as lossless PNG; normiere() then applies the same single JPEG q95 as for the originals;
  - split: every reconstruction inherits the split of its human source image (test sources -> test only).

Sources: human rows of packs v2 (telegram, wallhaven), v3z2 (danbooru_train), p2z (wallhaven_general,
danbooru_nichtanime) — the pack bytes (normalized 512 px). Four VAE families: SD 1.x (NovelAI v1 VAE), SDXL
(Illustrious), Flux (AnimePro-FLUX = FLUX.1 VAE), Qwen-Image. Each source image is used by one VAE only.

    python vae_rekon.py [--je-vae 12000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS, PROJEKT               # noqa: E402
from lesen import DATEN, Pack                    # noqa: E402

MODELLE = os.environ.get("LAB_MODELS", os.path.join(PROJEKT, "lab", "dataset", "models"))
ZIEL = f"{DATASETS}/vae_rekon"
VAES = ("sd1", "sdxl", "flux", "qwen")
QUELLEN = (("v2", ("telegram", "wallhaven")), ("v3z2", ("danbooru_train",)),
           ("p2z", ("wallhaven_general", "danbooru_nichtanime")))


def lade_vae(name):
    from diffusers import AutoencoderKL, AutoencoderKLQwenImage
    if name == "sd1":
        return AutoencoderKL.from_pretrained(f"{MODELLE}/nai_v1", subfolder="vae", torch_dtype=torch.float32)
    if name == "sdxl":
        return AutoencoderKL.from_single_file(f"{MODELLE}/illustrious.safetensors",
                                              config="stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae",
                                              torch_dtype=torch.float32)
    if name == "flux":
        return AutoencoderKL.from_pretrained(f"{MODELLE}/animepro_flux", subfolder="vae", torch_dtype=torch.bfloat16)
    if name == "qwen":
        return AutoencoderKLQwenImage.from_pretrained(f"{MODELLE}/qwen_image", subfolder="vae",
                                                      torch_dtype=torch.bfloat16)
    raise ValueError(name)


@torch.no_grad()
def rekonstruiere(vae, bild, video=False):
    a = np.asarray(bild, np.float32) / 127.5 - 1
    h, w = a.shape[:2]
    ph, pw = (-h) % 16, (-w) % 16
    a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="reflect")          # auffuellen, nicht beschneiden
    x = torch.from_numpy(a).permute(2, 0, 1)[None].to("cuda", vae.dtype)
    if video:
        x = x[:, :, None]                                               # Qwen-VAE erwartet (B, C, T, H, W)
    lat = vae.encode(x).latent_dist.mode()
    y = vae.decode(lat).sample
    if video:
        y = y[:, :, 0]
    y = y[0].float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy()[:h, :w]
    return Image.fromarray(((y + 1) * 127.5).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--je-vae", type=int, default=12000)
    a = ap.parse_args()
    rng = np.random.default_rng(29)
    pool = []
    for stamm, qs in QUELLEN:
        p = Pack(stamm)
        wd = np.load(os.path.join(DATEN, f"{stamm}_wd.npz"))
        for i in np.flatnonzero(p.ok & (p.label == 0) & np.isin(p.quelle, qs) & ~wd["gesperrt"]):
            pool.append((stamm, int(i)))
    pool = [pool[i] for i in rng.permutation(len(pool))]
    print(f"{len(pool)} menschliche Quellbilder, je VAE {a.je_vae}", flush=True)
    packs = {s: Pack(s) for s, _ in QUELLEN}
    ratings = {s: np.load(os.path.join(DATEN, f"{s}_wd.npz"))["rating"] for s, _ in QUELLEN}
    os.makedirs(ZIEL, exist_ok=True)
    mfp = os.path.join(ZIEL, "manifest.jsonl")
    fertig = set()
    if os.path.exists(mfp):
        fertig = {json.loads(z)["datei"] for z in open(mfp, encoding="utf-8")}
    mf = open(mfp, "a", encoding="utf-8")
    for k, name in enumerate(VAES):
        teil = pool[k * a.je_vae:(k + 1) * a.je_vae]
        os.makedirs(os.path.join(ZIEL, name), exist_ok=True)
        vae = lade_vae(name).to("cuda").eval()
        t0 = time.time()
        for n, (stamm, i) in enumerate(teil):
            p = packs[stamm]
            ziel = os.path.join(ZIEL, name, f"{stamm}_{i:07d}.png")
            if ziel in fertig:
                continue
            try:
                rekonstruiere(vae, p.bild(i), video=(name == "qwen")).save(ziel, compress_level=1)
            except Exception as e:
                print(f"  Fehler {stamm}/{i}: {e}", flush=True)
                continue
            mf.write(json.dumps({"datei": ziel, "vae": name, "split": str(p.split[i]),
                                 "grp": f"vae:{stamm}:{p.grp[i]}", "quelle_src": str(p.quelle[i]),
                                 "src_original": str(p.m[i].get("src", "")), "rating": int(ratings[stamm][i])},
                                ensure_ascii=False) + "\n")
            if n % 1000 == 0:
                mf.flush()
                print(f"  {name} {n}/{len(teil)}  {n / max(1e-6, time.time() - t0):.1f}/s", flush=True)
        mf.flush()
        del vae
        torch.cuda.empty_cache()
    mf.close()
    print("FERTIG")


if __name__ == "__main__":
    main()
