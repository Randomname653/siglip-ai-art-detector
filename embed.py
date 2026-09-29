r"""embed.py — frozen image features for a pack, one backbone at a time.

The detector is a small head on FROZEN features: fast enough to compare many variants (a head trains in
minutes), and honest — a large network fine-tuned on the data learns this dataset's quirks more easily than
the generators' traces.

Backbones (intermediate layers work far better than the final embedding):
    siglip_mid  SigLIP 2 so400m/384, mean patch token of layers 7/12/17/22/27 (5 x 1152) — default
    clip_mid    OpenCLIP ViT-L/14 (LAION CommonPool XL), CLS of blocks 7/11/15/19/23 (5 x 1024)
    dino_mid    DINOv2-L, CLS of layers 8/12/16/20/24 (5 x 1024)
    wd_eva02, clip_l14, dino   final-layer variants from the first experiments

    python embed.py --rueckgrat siglip_mid --pack <pack> --aug            # perturbed, for training
    python embed.py --rueckgrat siglip_mid --pack <pack> --nur-split test # clean, for testing
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import DATEN, Pack                    # noqa: E402


# --- WD-EVA02 -----------------------------------------------------------------

def wd_vorbereiten(bild, ziel=448):
    """Quadratischer MITTELAUSSCHNITT, bikubisch auf 448, float 0..255,
    RGB -> BGR.

    NICHT wie imgutils zum Taggen (weisses Quadrat drumherum). Die Balken
    waeren ein Leck: ihre Dicke kodiert das Seitenverhaeltnis exakt, und das
    ist bei den Generatoren auf Buckets gerastert (Vielfache von 64 px), bei
    echten Wallpapern beliebig. leak.py hat gemessen: allein das
    Seitenverhaeltnis trennt nach der Normierung noch mit AUC 0,997. Ein
    16:9-Negativ wird 512x288, sein Positiv aus dem Bucket 1344x768 wird
    512x293 — fuenf Pixel, und ein Kopf auf den Embeddings koennte sie finden.

    Der Mittelausschnitt entfernt das Seitenverhaeltnis aus dem Bild, genau
    wie es CLIPs eigene Vorverarbeitung (Resize kurze Kante + CenterCrop)
    ohnehin tut."""
    w, h = bild.size
    m = min(w, h)
    l, o = (w - m) // 2, (h - m) // 2
    q = bild.crop((l, o, l + m, o + m))
    if m != ziel:
        q = q.resize((ziel, ziel), Image.BICUBIC)
    return np.asarray(q, np.float32)[:, :, ::-1]


def wd_lauf(pack, batch):
    from imgutils.tagging import wd14
    sess = wd14._get_wd14_model("EVA02_Large")
    ein = sess.get_inputs()[0].name
    aus = [o.name for o in sess.get_outputs() if "fc_norm" in o.name]
    assert len(aus) == 1, [o.name for o in sess.get_outputs()]

    def vor(i):
        return wd_vorbereiten(pack.bild(i, aug=AUG))

    def rechne(idx, ex):
        x = np.stack(list(ex.map(vor, idx)))
        return sess.run(aus, {ein: x})[0]
    return rechne, 1024


# --- CLIP ViT-L/14 ------------------------------------------------------------

def clip_gewichte():
    """OpenCLIP ViT-L/14 (LAION CommonPool XL) von Hugging Face — aus dem lokalen Cache, sonst geladen."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download("laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K", "open_clip_pytorch_model.bin")


def clip_lauf(pack, batch):
    import open_clip
    import torch
    model, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained=clip_gewichte())
    model = model.cuda().eval().half()

    def vor(i):
        return prep(pack.bild(i, aug=AUG))

    def rechne(idx, ex):
        x = torch.stack(list(ex.map(vor, idx))).cuda().half()
        with torch.no_grad():
            return model.encode_image(x).float().cpu().numpy()
    return rechne, 768


# --- CLIP ViT-L/14, Zwischenschichten -------------------------------------------

CLIP_MID_BLOECKE = (7, 11, 15, 19, 23)   # von 24 Bloecken, 0-basiert


def clip_mid_modell():
    """CLIP ViT-L/14 mit Haken an fuenf Transformer-Bloecken; liefert deren
    CLS-Token hintereinander (5 x 1024 = 5120 Dimensionen).

    Das Endembedding von CLIP ist auf BEDEUTUNG trainiert ("ein Maedchen mit
    Schwert"). KI-Spuren sind eher Textur, Rauschen, Linienfuehrung — die
    stecken in den mittleren Schichten. RINE (Koutlis & Papadopoulos, ECCV
    2024) zeigt grosse Gewinne bei unbekannten Generatoren, wenn man die
    Zwischenschichten mitnimmt. Dasselbe Netz, ein Durchlauf, nur an mehr
    Stellen ausgelesen."""
    import open_clip
    model, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained=clip_gewichte())
    model = model.cuda().eval().half()
    fang = {}
    bloecke = model.visual.transformer.resblocks

    def haken(nr):
        def h(_m, _in, out):
            o = out[0] if isinstance(out, tuple) else out
            # CLS ist Token 0; je nach open_clip-Version (N, L, D) oder (L, N, D)
            fang[nr] = (o[:, 0] if o.shape[0] == fang["_n"] else o[0]).float()
        return h
    for nr in CLIP_MID_BLOECKE:
        bloecke[nr].register_forward_hook(haken(nr))
    return model, prep, fang


def clip_mid_lauf(pack, batch):
    import torch
    model, prep, fang = clip_mid_modell()

    def vor(i):
        return prep(pack.bild(i, aug=AUG))

    def rechne(idx, ex):
        x = torch.stack(list(ex.map(vor, idx))).cuda().half()
        fang["_n"] = x.shape[0]
        with torch.no_grad():
            model.encode_image(x)
        return torch.cat([fang[nr] for nr in CLIP_MID_BLOECKE], 1).cpu().numpy()
    return rechne, 1024 * len(CLIP_MID_BLOECKE)


# --- DINOv2-Large ----------------------------------------------------------------

def dino_modell():
    """DINOv2-Large (Meta, selbstueberwacht, ohne Sprache trainiert). Liefert
    das CLS-Token und den Mittelwert der Patch-Token hintereinander, 2048 dim.
    Das CLS fasst das Bild zusammen, die Patches tragen lokale Struktur — dort
    sitzen KI-Spuren. Die eigene Vorverarbeitung schneidet ebenfalls mittig
    quadratisch aus, das Seitenverhaeltnis kommt also auch hier nicht an."""
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large").cuda().eval().half()
    return model, proc


def dino_merkmale(model, proc, bilder):
    import torch
    x = proc(images=bilder, return_tensors="pt")["pixel_values"].cuda().half()
    with torch.no_grad():
        h = model(pixel_values=x).last_hidden_state.float()
    return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy()


def dino_lauf(pack, batch):
    model, proc = dino_modell()

    def rechne(idx, ex):
        return dino_merkmale(model, proc, list(ex.map(lambda i: pack.bild(i, aug=AUG), idx)))
    return rechne, 2048


DINO_MID_SCHICHTEN = (8, 12, 16, 20, 24)   # hidden_states-Index: 0 = Einbettung, i = nach Block i


def dino_mid_merkmale(model, proc, bilder):
    """CLS-Token aus fuenf Zwischenschichten von DINOv2 — dieselbe Idee wie
    clip_mid, die bei CLIP die Trefferquote auf den eigenen Generatoren von 65
    auf 95 % gehoben hat."""
    import torch
    x = proc(images=bilder, return_tensors="pt")["pixel_values"].cuda().half()
    with torch.no_grad():
        hs = model(pixel_values=x, output_hidden_states=True).hidden_states
    return torch.cat([hs[s][:, 0].float() for s in DINO_MID_SCHICHTEN], 1).cpu().numpy()


def dino_mid_lauf(pack, batch):
    model, proc = dino_modell()

    def rechne(idx, ex):
        return dino_mid_merkmale(model, proc, list(ex.map(lambda i: pack.bild(i, aug=AUG), idx)))
    return rechne, 1024 * len(DINO_MID_SCHICHTEN)


# --- SigLIP 2 (Zwischenschichten) ----------------------------------------------
# Drittes Rueckgrat (Vorversuch 2026-09-27): anders trainiert als CLIP (Sigmoid-Verlust, WebLI) und
# DINOv2 (selbstueberwacht). SigLIP hat kein CLS-Token -> je Schicht Mittel ueber die Patch-Token.

SIGLIP_NAME = "google/siglip2-so400m-patch14-384"
SIGLIP_MID_SCHICHTEN = (7, 12, 17, 22, 27)   # hidden_states-Index von 27 Bloecken


def siglip_modell():
    import torch
    from transformers import AutoImageProcessor, SiglipVisionModel
    proc = AutoImageProcessor.from_pretrained(SIGLIP_NAME)
    model = SiglipVisionModel.from_pretrained(SIGLIP_NAME, torch_dtype=torch.float16).cuda().eval()
    return model, proc


def siglip_vorbereiten(bild):
    """Wie SiglipImageProcessor (384x384 bilinear, /255, Mittel 0,5, Streuung 0,5), aber als einzelne
    Funktion fuer den Threadpool — der Prozessor rechnet sonst in einem Thread und bremst die GPU
    auf ~45 % (gemessen 2026-09-27: 48 Bilder/s)."""
    x = np.asarray(bild.convert("RGB").resize((384, 384), Image.BILINEAR), np.float32)
    return ((x / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)


def siglip_mid_merkmale(model, proc, bilder, ex=None):
    import torch
    if ex is not None:
        x = torch.from_numpy(np.stack(list(ex.map(siglip_vorbereiten, bilder)))).cuda().half()
    else:
        x = torch.from_numpy(np.stack([siglip_vorbereiten(b) for b in bilder])).cuda().half()
    with torch.no_grad():
        hs = model(pixel_values=x, output_hidden_states=True).hidden_states
    return torch.cat([hs[s].float().mean(1) for s in SIGLIP_MID_SCHICHTEN], 1).cpu().numpy()


def siglip_mid_lauf(pack, batch):
    model, proc = siglip_modell()

    def rechne(idx, ex):
        return siglip_mid_merkmale(model, proc, list(ex.map(lambda i: pack.bild(i, aug=AUG), idx)), ex)
    return rechne, 1152 * len(SIGLIP_MID_SCHICHTEN)


AUG = False     # per --aug: jedes Bild durch norm.stoere (siehe dort)

RUECKGRATE = {"wd_eva02": wd_lauf, "clip_l14": clip_lauf, "clip_mid": clip_mid_lauf,
              "dino": dino_lauf, "dino_mid": dino_mid_lauf, "siglip_mid": siglip_mid_lauf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rueckgrat", required=True, choices=list(RUECKGRATE))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--aug2", action="store_true", help="norm.stoere2 (Alltags-Bearbeitungen), Datei *_aug2_*")
    ap.add_argument("--splits", default=None, help="nur diese Splits einbetten, Dateiname unveraendert (z. B. train,val)")
    ap.add_argument("--pack", default="norm512", help="norm512 (v1) oder v2")
    ap.add_argument("--nur-split", default=None,
                    help="nur diesen Split einbetten (z. B. test fuer die ungestoerten Endtests)")
    args = ap.parse_args()
    global AUG
    AUG = 2 if args.aug2 else args.aug

    pack = Pack(args.pack)
    idx = np.flatnonzero(pack.ok)
    if args.nur_split:
        idx = idx[pack.split[idx] == args.nur_split]
    if args.splits:
        idx = idx[np.isin(pack.split[idx], args.splits.split(","))]
    if args.probe:
        idx = idx[np.linspace(0, len(idx) - 1, args.probe).astype(int)]
    ziel = os.path.join(DATEN, f"emb_{args.rueckgrat}" + ("_aug2" if args.aug2 else "_aug" if args.aug else "")
                        + ("" if args.pack == "norm512" else "_" + args.pack)
                        + (f"_{args.nur_split}" if args.nur_split else "")
                        + ("_probe" if args.probe else "") + ".npy")
    rechne, dim = RUECKGRATE[args.rueckgrat](pack, args.batch)

    emb = np.zeros((len(pack), dim), np.float16)
    fertig = np.zeros(len(pack), bool)
    t0 = time.time()
    with ThreadPoolExecutor(12) as ex:
        for s in range(0, len(idx), args.batch):
            b = idx[s:s + args.batch]
            emb[b] = rechne(b, ex)
            fertig[b] = True
            if (s // args.batch) % 100 == 0 and s:
                el = time.time() - t0
                print(f"  {s}/{len(idx)}  {s/el:.0f}/s  "
                      f"Rest {(len(idx)-s)/(s/el)/60:.1f} min", flush=True)
    np.save(ziel, emb)
    np.save(ziel.replace(".npy", "_ok.npy"), fertig)
    el = time.time() - t0
    print(f"{len(idx)} Embeddings ({dim} dim) in {el/60:.1f} min "
          f"({len(idx)/el:.0f}/s) -> {ziel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
