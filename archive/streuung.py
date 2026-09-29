r"""streuung.py — wie stark schwankt der Telegram-Test zwischen Laeufen, und
waehlt das fruehe Stoppen die richtige Epoche?

Anlass (2026-09-25): derselbe Kopf auf denselben Merkmalen kam mit allen
216k Trainingsbildern auf B-AUC 0,818, mit einer ausgewogenen Teilmenge von
44k auf 0,853. Bevor irgendein Unterschied von ein paar Hundertstel etwas
bedeuten darf, muss das Rauschen bekannt sein.

Je Lauf 15 Epochen ohne Stopp; nach jeder Epoche:
  val_alle  AUC auf dem ganzen val (danach stoppt train.kopf_mlp heute)
  val_tg    AUC nur Telegram-Menschen gegen Ai-Only-val
  B         AUC und @1 % auf dem Telegram-Test — NUR zur Diagnose, gewaehlt
            wird nie danach
Laeufe: voll x Seeds 0-2, Teilmenge (je Quelle 12k) x Seeds 0-2.

    python lab/detektor/streuung.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from lesen import DATEN, Pack                    # noqa: E402
from train import auc                            # noqa: E402
from train2 import Skalierer, gewichte, tpr_bei  # noqa: E402

RG = ("clip_mid", "dino_mid")


def lade(name):
    return [np.load(os.path.join(DATEN, f"emb_{r}_{name}.npy"), mmap_mode="r") for r in RG]


def nimm(teile, idx):
    return np.concatenate([np.asarray(t[idx]) for t in teile], 1)


def lauf(Xtr, ytr, wtr, Xva, yva, tg_va, Xb, yb, seed, epochen=15):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.GELU(), nn.Dropout(0.3),
                        nn.Linear(512, 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    Xt = torch.from_numpy(Xtr)
    yt, wt = torch.tensor(ytr, dtype=torch.float32), torch.tensor(wtr, dtype=torch.float32)

    def vor(X):
        with torch.no_grad():
            return torch.cat([net(torch.from_numpy(X[s:s + 4096]).cuda().float()).squeeze(1).cpu()
                              for s in range(0, len(X), 4096)]).numpy()
    verlauf = []
    for ep in range(epochen):
        net.train()
        perm = torch.randperm(len(Xt))
        for s in range(0, len(perm), 1024):
            b = perm[s:s + 1024]
            loss = (nn.functional.binary_cross_entropy_with_logits(
                net(Xt[b].cuda().float()).squeeze(1), yt[b].cuda(), reduction="none")
                * wt[b].cuda()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        sv, sb = vor(Xva), vor(Xb)
        verlauf.append({"ep": ep, "val_alle": auc(sv, yva), "val_tg": auc(sv[tg_va], yva[tg_va]),
                        "B_auc": auc(sb, yb), "B_tpr1": tpr_bei(sb[yb == 0], sb[yb == 1])})
        v = verlauf[-1]
        print(f"    ep {ep:>2}  val_alle {v['val_alle']:.4f}  val_tg {v['val_tg']:.4f}  "
              f"B {v['B_auc']:.4f} / {v['B_tpr1']:.1%}", flush=True)
    return verlauf


def main():
    pack = Pack("v2")
    A = lade("aug_v2")
    T = lade("v2_test")
    ok = pack.ok & np.all([np.load(os.path.join(DATEN, f"emb_{r}_aug_v2_ok.npy")) for r in RG], 0)
    tr = np.flatnonzero(ok & (pack.split == "train"))
    va = np.flatnonzero(ok & (pack.split == "val"))
    tb = np.flatnonzero(pack.ok & (pack.split == "test") & np.isin(pack.quelle, ["telegram", "ai_only"]))
    tg_va = np.isin(pack.quelle[va], ["telegram", "ai_only"])

    print("lade val und B ...", flush=True)
    Xva_roh, Xb_roh = nimm(A, va), nimm(T, tb)
    yva, yb = pack.label[va], pack.label[tb]
    erg = {}
    rng = np.random.default_rng(11)
    for art in ("teil", "voll"):
        for seed in (0, 1, 2):
            if art == "voll":
                idx = tr
            else:
                r = np.random.default_rng(100 + seed)
                idx = np.sort(np.concatenate([
                    r.choice(tr[pack.quelle[tr] == q], min(12000, int((pack.quelle[tr] == q).sum())),
                             replace=False) for q in ("wallhaven", "generiert", "telegram", "ai_only")]))
            print(f"\n{art} seed {seed}: {len(idx)} Trainingsbilder", flush=True)
            Xtr_roh = nimm(A, idx)
            sc = Skalierer().fit(Xtr_roh, np.arange(len(idx)))
            v = lauf(sc.transform(Xtr_roh), pack.label[idx], gewichte(pack, idx, "quelle"),
                     sc.transform(Xva_roh), yva, tg_va, sc.transform(Xb_roh), yb, seed)
            del Xtr_roh
            erg[f"{art}_{seed}"] = v
            with open(os.path.join(DATEN, "streuung.json"), "w", encoding="utf-8") as f:
                json.dump(erg, f, indent=1)
    print("\nZusammenfassung (Epoche gewaehlt nach val_alle | nach val_tg):")
    for k, v in erg.items():
        a = max(v, key=lambda e: e["val_alle"])
        t = max(v, key=lambda e: e["val_tg"])
        print(f"  {k:<8} B {a['B_auc']:.4f}/{a['B_tpr1']:.1%} (ep {a['ep']})  |  "
              f"B {t['B_auc']:.4f}/{t['B_tpr1']:.1%} (ep {t['ep']})")
    print("FERTIG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
