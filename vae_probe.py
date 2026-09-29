r"""vae_probe.py — pilot for "aligned" training data: can our features see a generator's autoencoder trace?

Human test images are passed through a latent-diffusion VAE (encode -> decode, no denoising). The reconstruction
has the same content and style as the original; only the generator's decoder trace differs. If a linear probe on
the frozen SigLIP features separates original vs. reconstruction (AUC near 1), reconstructions can serve as
style-matched AI examples in training (Rajan et al. ICLR 2025, B-Free CVPR 2025) — the head could then no longer
use "looks polished" as a cue. Also reported: how often p8sx already flags the reconstructions.

    python vae_probe.py [--n 2000] [--vae flux,sdxl]
"""
import argparse
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from danbooru_test import bewerte, koepfe_von    # noqa: E402
from lesen import DATEN, Pack                    # noqa: E402
from pfade import PROJEKT                        # noqa: E402
from pruefe import stapelrechner                 # noqa: E402
from robust_test import scanner                  # noqa: E402

MODELLE = os.environ.get("LAB_MODELS", os.path.join(PROJEKT, "lab", "dataset", "models"))


def lade_vae(name):
    from diffusers import AutoencoderKL
    if name == "flux":
        return AutoencoderKL.from_pretrained(f"{MODELLE}/animepro_flux", subfolder="vae", torch_dtype=torch.bfloat16)
    if name == "sdxl":
        return AutoencoderKL.from_single_file(f"{MODELLE}/illustrious.safetensors",
                                              config="stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae",
                                              torch_dtype=torch.float32)
    raise ValueError(name)


@torch.no_grad()
def rekonstruiere(vae, bild):
    w, h = bild.size
    w2, h2 = w - w % 16, h - h % 16
    b = bild.crop((0, 0, w2, h2))
    x = torch.from_numpy(np.asarray(b, np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to("cuda", vae.dtype)
    lat = vae.encode(x).latent_dist.mode()
    y = vae.decode(lat).sample[0].float().clamp(-1, 1)
    return Image.fromarray(((y.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8))


def auc(neg, pos):
    x = np.concatenate([neg, pos])
    r = x.argsort().argsort().astype(np.float64) + 1
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--vae", default="flux,sdxl")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    k = koepfe_von("p8sx")
    rg = [str(r) for r in k[0]["rueckgrate"]]
    s1 = float(k[0]["schwelle_1"])
    rng = np.random.default_rng(5)
    pd, p2, pz = Pack("v3d"), Pack("v2"), Pack("p2z")
    quellen = [(pd, rng.choice(pd.wo(split="test"), a.n // 2, replace=False)),
               (p2, rng.choice(np.flatnonzero(p2.ok & (p2.split == "test") & (p2.quelle == "telegram")), a.n // 4,
                               replace=False)),
               (pz, rng.choice(np.flatnonzero(pz.ok & (pz.split == "test") & (pz.quelle == "wallhaven_general")),
                               a.n // 4, replace=False))]
    with ThreadPoolExecutor(8) as ex:
        orig = [b for p, idx in quellen for b in ex.map(lambda j, p=p: p.bild(int(j)), idx)]
    rechner = stapelrechner(rg)

    def merkmale(bilder):
        with ThreadPoolExecutor(8) as ex:
            bs = list(ex.map(scanner, bilder))
            return np.concatenate([np.concatenate([rechner[r](bs[s:s + 32], ex) for r in rg], 1)
                                   for s in range(0, len(bs), 32)]).astype(np.float32)

    X0 = merkmale(orig)
    z0 = bewerte(X0, k)
    erg = {"n": len(orig), "fa_original": float(np.mean(z0 > s1))}
    print(f"{len(orig)} menschliche Testbilder; p8sx markiert Originale: {erg['fa_original']:.2%}", flush=True)
    for name in a.vae.split(","):
        vae = lade_vae(name).to("cuda").eval()
        rec = [rekonstruiere(vae, b) for b in orig]
        del vae
        torch.cuda.empty_cache()
        X1 = merkmale(rec)
        z1 = bewerte(X1, k)
        # linearer Test, gruppiert nach Bild (Original und Rekonstruktion nie getrennt auf train/test)
        X = np.concatenate([X0, X1])
        y = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
        g = np.r_[np.arange(len(X0)), np.arange(len(X1))]
        pr = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            sk = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=0.1, max_iter=2000).fit(sk.transform(X[tr]), y[tr])
            pr[te] = m.decision_function(sk.transform(X[te]))
        erg[name] = {"p8sx_markiert_rekonstruktion": float(np.mean(z1 > s1)),
                     "p8sx_logit_verschiebung_median": float(np.median(z1 - z0)),
                     "p8sx_auc_orig_vs_rek": auc(z0, z1),
                     "linear_probe_auc": auc(pr[y == 0], pr[y == 1])}
        print(f"  {name}: p8sx markiert Rekonstruktionen {erg[name]['p8sx_markiert_rekonstruktion']:.2%} "
              f"(Logit-Verschiebung {erg[name]['p8sx_logit_verschiebung_median']:+.2f}, AUC {erg[name]['p8sx_auc_orig_vs_rek']:.3f})"
              f" | linearer Test Original vs. Rekonstruktion: AUC {erg[name]['linear_probe_auc']:.4f}", flush=True)
        rec[0].save(os.path.join(DATEN, f"vae_probe_{name}_beispiel.jpg"), quality=95)
    json.dump(erg, open(os.path.join(DATEN, "vae_probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
