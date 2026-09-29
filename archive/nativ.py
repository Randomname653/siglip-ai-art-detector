r"""nativ.py — sehen die Rueckgrate mehr, wenn das Bild NICHT verkleinert wird?

CLIP und DINOv2 bekommen bisher das 512er-Bild auf 224 px heruntergerechnet
(Faktor 2,3). Genau dabei gehen die feinen Pixelspuren eines Generators
verloren — Rauschmuster, VAE-Artefakte, Kantenverlauf. Hier sehen dieselben
Netze fuenf 224er-Ausschnitte in Originalaufloesung (Mitte + vier Ecken),
die Zwischenschicht-Merkmale werden ueber die Ausschnitte gemittelt.

Vergleich auf einer Teilmenge, alles andere gleich (gleiche Bilder, gleiche
Stoerungen, gleicher MLP-Kopf, Gewichtung je Quelle):
  verkleinert  = die vorhandenen Embeddings (emb_*_v2)
  nativ        = die neuen Ausschnitt-Embeddings
  beide        = hintereinander
Gemessen wie train2.py: A (Wallhaven gegen Generatoren, gestoert) und
B (Telegram-Test, ungestoert, ohne Vorschaubilder).

    python lab/detektor/nativ.py
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
import embed                                     # noqa: E402
from lesen import DATEN, Pack                    # noqa: E402
from train import kopf_mlp, auc                  # noqa: E402
from train2 import Skalierer, gewichte, tpr_bei  # noqa: E402

GROESSE = 224
CLIP_MW, CLIP_SA = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
DINO_MW, DINO_SA = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def ausschnitte(bild):
    """Fuenf 224er-Ausschnitte ohne Skalierung. Nur wenn die kurze Kante unter
    224 liegt (sehr lange Formate), wird minimal vergroessert."""
    from PIL import Image
    w, h = bild.size
    if min(w, h) < GROESSE:
        f = GROESSE / min(w, h)
        bild = bild.resize((max(GROESSE, round(w * f)), max(GROESSE, round(h * f))), Image.LANCZOS)
        w, h = bild.size
    a = np.asarray(bild, np.float32) / 255.0
    xs, ys = (0, (w - GROESSE) // 2, w - GROESSE), (0, (h - GROESSE) // 2, h - GROESSE)
    pos = [(xs[1], ys[1]), (xs[0], ys[0]), (xs[2], ys[0]), (xs[0], ys[2]), (xs[2], ys[2])]
    return np.stack([a[y:y + GROESSE, x:x + GROESSE] for x, y in pos])      # (5, 224, 224, 3)


def main():
    import torch
    pack = Pack("v2")
    rng = np.random.default_rng(7)

    def teil(maske, n):
        i = np.flatnonzero(maske & pack.ok)
        return np.sort(rng.choice(i, min(n, len(i)), replace=False))

    tr = np.concatenate([teil((pack.split == "train") & (pack.quelle == q), n) for q, n in
                         (("wallhaven", 12000), ("generiert", 12000), ("telegram", 12000),
                          ("ai_only", 12000))])
    va = np.concatenate([teil((pack.split == "val") & (pack.quelle == q), n) for q, n in
                         (("wallhaven", 1500), ("generiert", 1500), ("telegram", 4000),
                          ("ai_only", 1000))])
    te_a = np.concatenate([teil((pack.split == "test") & (pack.quelle == q), n) for q, n in
                           (("wallhaven", 3000), ("generiert", 3000))])
    te_b = np.flatnonzero(pack.ok & (pack.split == "test") & np.isin(pack.quelle, ["telegram", "ai_only"]))
    print(f"train {len(tr)}  val {len(va)}  A {len(te_a)}  B {len(te_b)}", flush=True)

    ziel = os.path.join(DATEN, "nativ_emb.npz")
    if os.path.exists(ziel):
        z = np.load(ziel)
        E = {k: z[k] for k in z.files}
    else:
        cm, _, fang = embed.clip_mid_modell()
        dm, _ = embed.dino_modell()
        cmw = torch.tensor(CLIP_MW, device="cuda").view(1, 3, 1, 1).half()
        csa = torch.tensor(CLIP_SA, device="cuda").view(1, 3, 1, 1).half()
        dmw = torch.tensor(DINO_MW, device="cuda").view(1, 3, 1, 1).half()
        dsa = torch.tensor(DINO_SA, device="cuda").view(1, 3, 1, 1).half()

        def rechne(idx, aug, ex):
            cr = np.stack(list(ex.map(lambda i: ausschnitte(pack.bild(i, aug=aug)), idx)))
            n = len(idx)
            x = torch.from_numpy(cr.reshape(n * 5, GROESSE, GROESSE, 3)).cuda().half().permute(0, 3, 1, 2)
            with torch.no_grad():
                fang["_n"] = x.shape[0]
                cm.encode_image((x - cmw) / csa)
                c = torch.cat([fang[b] for b in embed.CLIP_MID_BLOECKE], 1)
                hs = dm(pixel_values=(x - dmw) / dsa, output_hidden_states=True).hidden_states
                d = torch.cat([hs[s][:, 0].float() for s in embed.DINO_MID_SCHICHTEN], 1)
            c = c.view(n, 5, -1).mean(1)
            d = d.view(n, 5, -1).mean(1)
            return torch.cat([c, d], 1).cpu().numpy().astype(np.float16)

        E = {}
        t0 = time.time()
        with ThreadPoolExecutor(12) as ex:
            for name, idx, aug in (("tr", tr, True), ("va", va, True), ("a", te_a, True), ("b", te_b, False)):
                out = []
                for s in range(0, len(idx), 32):
                    out.append(rechne(idx[s:s + 32], aug, ex))
                    if (s // 32) % 100 == 0 and s:
                        print(f"  {name} {s}/{len(idx)}  {(time.time()-t0)/60:.1f} min", flush=True)
                E[name] = np.concatenate(out)
        np.savez(ziel, **E)
        print(f"Ausschnitt-Embeddings in {(time.time()-t0)/60:.1f} min", flush=True)

    # die vorhandenen, verkleinerten Merkmale fuer dieselben Bilder
    def alt(idx, aug):
        teile = []
        for r in ("clip_mid", "dino_mid"):
            name = f"emb_{r}{'_aug' if aug else ''}_v2{'' if aug else '_test'}"
            X = np.load(os.path.join(DATEN, name + ".npy"), mmap_mode="r")
            teile.append(np.asarray(X[idx]))
        return np.concatenate(teile, 1)
    V = {"tr": alt(tr, True), "va": alt(va, True), "a": alt(te_a, True), "b": alt(te_b, False)}

    erg = {}
    for art in ("verkleinert", "nativ", "beide"):
        M = V if art == "verkleinert" else E if art == "nativ" else \
            {k: np.concatenate([V[k], E[k]], 1) for k in V}
        sc = Skalierer().fit(M["tr"], np.arange(len(tr)))
        clf, info = kopf_mlp(sc.transform(M["tr"]), pack.label[tr], sc.transform(M["va"]),
                             pack.label[va], gewichte(pack, tr, "quelle"))
        sa = clf.decision_function(sc.transform(M["a"]))
        ya = pack.label[te_a]
        sb = clf.decision_function(sc.transform(M["b"]))
        yb = pack.label[te_b]
        kan = (yb == 0) | (pack.grp[te_b] == "ai:ChatExport_2026-05-31 (1)")
        erg[art] = {"val_auc": info["val_auc"],
                    "A_auc": auc(sa, ya), "A_tpr1": tpr_bei(sa[ya == 0], sa[ya == 1]),
                    "B_auc": auc(sb, yb), "B_tpr1": tpr_bei(sb[yb == 0], sb[yb == 1]),
                    "B_tpr5": tpr_bei(sb[yb == 0], sb[yb == 1], 0.05),
                    "kanal_tpr1": tpr_bei(sb[kan & (yb == 0)], sb[kan & (yb == 1)])}
        e = erg[art]
        print(f"{art:<12} A AUC {e['A_auc']:.4f} @1% {e['A_tpr1']:.1%} | B AUC {e['B_auc']:.4f} "
              f"@1% {e['B_tpr1']:.1%} @5% {e['B_tpr5']:.1%} | KI-Kanal @1% {e['kanal_tpr1']:.1%}", flush=True)
        np.save(os.path.join(DATEN, f"nativ_score_{art}_B.npy"), np.c_[te_b, sb])
    with open(os.path.join(DATEN, "nativ.json"), "w", encoding="utf-8") as f:
        json.dump(erg, f, indent=1)
    print("FERTIG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
