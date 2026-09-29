r"""kandidat.py — a trained head as a function "path -> logit", plus the batched matrix version.

Every image goes through EXACTLY the normalization used in training (norm.normiere: longer side 512 px,
JPEG q95 4:4:4) and the same backbone preprocessing as embed.py. Anything done differently here would
measure a different model than the one trained.

`logit_stapel(X, head)` applies a saved head (MLP: Linear -> GELU -> Linear, or logistic regression) to a
feature matrix; the ensemble score is the mean over heads (danbooru_test.bewerte).
"""
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed import wd_vorbereiten                 # noqa: E402
from lesen import DATEN                          # noqa: E402
from norm import normiere                        # noqa: E402

_cache = {}


def _clip():
    if "clip" not in _cache:
        import glob
        import open_clip
        import torch
        gew = glob.glob(os.path.expanduser(
            r"~\.cache\huggingface\hub\models--laion--CLIP-ViT-L-14-CommonPool.XL-s13B-b90K"
            r"\snapshots\*\open_clip_pytorch_model.bin"))[0]
        model, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained=gew)
        _cache["clip"] = (model.cuda().eval().half(), prep)
    return _cache["clip"]


def _wd():
    if "wd" not in _cache:
        from imgutils.tagging import wd14
        sess = wd14._get_wd14_model("EVA02_Large")
        aus = [o.name for o in sess.get_outputs() if "fc_norm" in o.name]
        _cache["wd"] = (sess, sess.get_inputs()[0].name, aus)
    return _cache["wd"]


def embedding(bild, rueckgrat):
    if rueckgrat == "clip_l14":
        import torch
        model, prep = _clip()
        with torch.no_grad():
            return model.encode_image(prep(bild)[None].cuda().half()).float().cpu().numpy()[0]
    if rueckgrat == "wd_eva02":
        sess, ein, aus = _wd()
        return sess.run(aus, {ein: wd_vorbereiten(bild)[None]})[0][0]
    if rueckgrat == "clip_mid":
        import torch
        from embed import CLIP_MID_BLOECKE, clip_mid_modell
        if "clip_mid" not in _cache:
            _cache["clip_mid"] = clip_mid_modell()
        model, prep, fang = _cache["clip_mid"]
        fang["_n"] = 1
        with torch.no_grad():
            model.encode_image(prep(bild)[None].cuda().half())
        return torch.cat([fang[nr] for nr in CLIP_MID_BLOECKE], 1).cpu().numpy()[0]
    if rueckgrat == "dino":
        from embed import dino_merkmale, dino_modell
        if "dino" not in _cache:
            _cache["dino"] = dino_modell()
        return dino_merkmale(*_cache["dino"], [bild])[0]
    if rueckgrat == "dino_mid":
        from embed import dino_mid_merkmale, dino_modell
        if "dino" not in _cache:
            _cache["dino"] = dino_modell()
        return dino_mid_merkmale(*_cache["dino"], [bild])[0]
    raise ValueError(rueckgrat)


def _kopf(name):
    if ("kopf", name) not in _cache:
        k = np.load(os.path.join(DATEN, f"kopf_{name}.npz"), allow_pickle=False)
        _cache[("kopf", name)] = {x: k[x] for x in k.files}
    return _cache[("kopf", name)]


def logit(x, k):
    """Den gespeicherten Kopf anwenden — logistische Regression oder MLP
    (Linear -> GELU -> Linear, wie in train.kopf_mlp)."""
    z = (x - k["mean"]) / k["scale"]
    if str(k.get("art", "logreg")) == "mlp":
        from scipy.special import erf
        h = z @ k["w1"].T + k["b1"]
        h = 0.5 * h * (1 + erf(h / np.sqrt(2)))       # exakte GELU wie torch
        return float((h @ k["w2"].T + k["b2"])[0])
    return float(z @ k["coef"] + k["intercept"][0])


def logit_stapel(X, k, stueck=4096):
    """logit() fuer viele Zeilen auf einmal — gleiche Rechnung, als Matrix.
    Bild fuer Bild in Python kostete beim Danbooru-Test ueber eine Stunde CPU."""
    from scipy.special import erf
    out = []
    for a in range(0, len(X), stueck):
        z = (np.asarray(X[a:a + stueck], np.float32) - k["mean"]) / k["scale"]
        if str(k.get("art", "logreg")) == "mlp":
            h = z @ k["w1"].T + k["b1"]
            h = 0.5 * h * (1 + erf(h / np.sqrt(2)))
            out.append((h @ k["w2"].T + k["b2"])[:, 0])
        else:
            out.append(z @ k["coef"] + k["intercept"][0])
    return np.concatenate(out)


def score(pfad, kopf="clip_l14"):
    k = _kopf(kopf)
    b = normiere(pfad)[0]
    bild = Image.open(io.BytesIO(b)).convert("RGB")
    x = np.concatenate([embedding(bild, r) for r in k["rueckgrate"]])
    return float(1 / (1 + np.exp(-logit(x, k))))


def score_clip(pfad):
    return score(pfad, "clip_l14")


def score_wd(pfad):
    return score(pfad, "wd_eva02")


def score_beide(pfad):
    return score(pfad, "wd_eva02_clip_l14")


# Die robust trainierten Koepfe (auf gestoerten Bildern, siehe norm.stoere).
# Nur diese taugen fuer fremdes Material — die obigen haben Verarbeitungs-
# spuren gelernt und fallen bei x0,75 + q85 auf AUC 0,64.
def score_clip_aug(pfad):
    return score(pfad, "clip_l14_aug")


def score_wd_aug(pfad):
    return score(pfad, "wd_eva02_aug")


def score_beide_aug(pfad):
    return score(pfad, "wd_eva02_clip_l14_aug")


def score_clip_mlp_aug(pfad):
    return score(pfad, "clip_l14_mlp_aug")


def score_wd_mlp_aug(pfad):
    return score(pfad, "wd_eva02_mlp_aug")


def score_beide_mlp_aug(pfad):
    return score(pfad, "wd_eva02_clip_l14_mlp_aug")
