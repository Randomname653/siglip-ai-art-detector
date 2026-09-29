r"""train3.py — train an ensemble of MLP heads on frozen features.

  - Weighting: each class carries half of the loss; within a class every source counts equally, so large or
    easy sources (e.g. locally generated images) cannot dominate. --quelle-gruppe merges several sources into
    one weighting group (the three Midjourney sources count as one).
  - --rating-ausgleich re-weights so the content rating is equally distributed in both classes
    ("explicit = AI" must not be learnable).
  - Ensemble of N heads with different seeds (--koepfe, --seed-basis); logits are averaged. Single heads varied
    by up to 0.04 AUC between seeds.
  - --rueckgrate selects the backbones (e.g. siglip_mid); --ohne-gen removes generators entirely from training
    and validation (leave-generator-out experiments, ausschluss.py).
  - Early stopping on the validation split; thresholds are set afterwards with kalibriere.py.

    python train3.py --zusatz v3z2,v3z3,p2z,p3z,p4z,p5z --rueckgrate siglip_mid --rating-ausgleich --kalib-p2         --quelle-gruppe midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney --tag p8s
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import DATEN, Pack                    # noqa: E402
from train import auc                            # noqa: E402
from train2 import Skalierer, tpr_bei            # noqa: E402

RG = ("clip_mid", "dino_mid")
MENSCH_Q = ("wallhaven", "telegram")
AUGSUF = "_aug"   # --aug-name aug2: Trainings-/Val-Merkmale aus norm.stoere2


def lade(pack_name, aug, split=None):
    teile, oks = [], []
    for r in RG:
        n = f"emb_{r}{AUGSUF if aug else ''}_{pack_name}{'_' + split if split else ''}"
        teile.append(np.load(os.path.join(DATEN, n + ".npy"), mmap_mode="r"))
        oks.append(np.load(os.path.join(DATEN, n + "_ok.npy")))
    return teile, np.all(oks, 0)


class Sicht:
    """Mehrere Merkmals-Arrays (je Pack eins) als ein Array mit globalen Indizes."""
    def __init__(self, arrays):
        self.a = arrays
        self.off = np.cumsum([0] + [len(x) for x in arrays])

    def __len__(self):
        return int(self.off[-1])

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx), self.a[0].shape[1]), self.a[0].dtype)
        for p, x in enumerate(self.a):
            m = (idx >= self.off[p]) & (idx < self.off[p + 1])
            if m.any():
                out[m] = x[idx[m] - self.off[p]]
        return out


class Verbund:
    """Mehrere Zusatzpacks (v3z2, v3z3, ...) wie ein Pack: Aufstockungen brauchen
    nur ein kleines neues Pack statt den ganzen Neubau (2026-09-26)."""
    FELDER = ("label", "quelle", "split", "model", "grp", "ok")

    def __init__(self, namen):
        self.namen = namen
        self.packs = [Pack(n) for n in namen]
        for f in self.FELDER:
            setattr(self, f, np.concatenate([getattr(p, f) for p in self.packs]))
        self.m = [z for p in self.packs for z in p.m]

    def __len__(self):
        return len(self.label)

    def lade(self, aug, split=None):
        teile, oks = [], []
        for r in RG:
            arr, ok = [], []
            for n in self.namen:
                s = f"emb_{r}{AUGSUF if aug else ''}_{n}{'_' + split if split else ''}"
                arr.append(np.load(os.path.join(DATEN, s + ".npy"), mmap_mode="r"))
                ok.append(np.load(os.path.join(DATEN, s + "_ok.npy")))
            teile.append(Sicht(arr))
            oks.append(np.concatenate(ok))
        return teile, np.all(oks, 0)

    def wd(self):
        d = [np.load(os.path.join(DATEN, f"{n}_wd.npz")) for n in self.namen]
        return {k: np.concatenate([x[k] for x in d]) for k in ("rating", "gesperrt")}


def gen_maske(pack, namen):
    """Zeilen, die zu einem der Generatoren gehoeren: Pack-Modell (CivitAI-Basismodell, eigener
    Generator, DiffusionDB, Midjourney ...) exakt, oder "aibooru:<teil>" als Teilstring im
    AIBooru-Modell-Tag (ki_modell). Fuer Ausschlusstests (--ohne-gen)."""
    exakt = {n for n in namen if not n.startswith("aibooru:")}
    ab = [n[8:] for n in namen if n.startswith("aibooru:")]
    m = np.isin(pack.model, list(exakt))
    if ab:
        km = np.array([str(z.get("ki_modell", "")).lower() for z in pack.m])
        m |= (pack.quelle == "aibooru") & np.array([any(t in k for t in ab) for k in km])
    return m


def nimm(teile, idx):
    return np.concatenate([np.asarray(t[idx]) for t in teile], 1)


def gewichte3(y, quelle):
    w = np.zeros(len(y), np.float32)
    for c in (0, 1):
        qs = np.unique(quelle[y == c])
        for q in qs:
            m = (y == c) & (quelle == q)
            w[m] = 0.5 / len(qs) / m.sum()
    return w / w.mean()


def rating_ausgleich(w, y, rating):
    """Gewichte so umverteilen, dass das WD-Rating innerhalb jeder Klasse gleich
    verteilt ist (Ziel: Mittel der beiden Klassenverteilungen). Dann traegt
    "explizit" keine Information ueber die Klasse — Ai Only ist zu 90 %
    explizit, die Menschen kaum. Faktoren auf [0,1; 10] begrenzt."""
    w = w.copy()
    p = np.zeros((2, 4))
    for c in (0, 1):
        for r in range(4):
            p[c, r] = w[(y == c) & (rating == r)].sum()
        p[c] /= max(p[c].sum(), 1e-9)
    ziel = p.mean(0)
    for c in (0, 1):
        for r in range(4):
            m = (y == c) & (rating == r)
            if p[c, r] > 0:
                w[m] *= np.clip(ziel[r] / p[c, r], 0.1, 10)
    print("  Rating je Klasse vorher (g/s/q/e):", np.round(p, 3).tolist(), "Ziel", np.round(ziel, 3).tolist())
    return w / w.mean()


def kopf(Xtr, ytr, wtr, Xva, yva, seed, epochen=40):
    """Wie train.kopf_mlp, mit Seed; fruehes Stoppen auf val (alle Quellen)."""
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
    best = (-1, None, 0)
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
        a = auc(vor(Xva), yva)
        if a > best[0]:
            best = (a, {k: v.clone() for k, v in net.state_dict().items()}, ep)
        elif ep - best[2] >= 6:
            break
    net.load_state_dict(best[1])
    net.eval()
    return vor, {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}, best


def kanal2_merkmale():
    """Clip-/DINO-Mitte der Kanal-2-Fotos (nach beiden Sperren), zwischengespeichert."""
    ziel = os.path.join(DATEN, "kanal2_emb.npz" if tuple(RG) == ("clip_mid", "dino_mid")
                        else f"kanal2_emb_{'_'.join(RG)}.npz")
    ids_ok = set(np.load(os.path.join(DATEN, "kanal2_quelle_clip_mid_dino_mid.npy"))[:, 0].astype(int))
    Z = [z for z in json.load(open(os.path.join(DATEN, "kanal2.json"), encoding="utf-8"))
         if z["id"] in ids_ok]
    if os.path.exists(ziel):
        d = np.load(ziel)
        if len(d["ids"]) == len(Z):
            return d["X"]
    import io
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from norm import normiere
    from pruefe import stapelrechner
    rechner = stapelrechner(list(RG))
    out = []
    with ThreadPoolExecutor(8) as ex:
        for a in range(0, len(Z), 32):
            bs = list(ex.map(lambda z: Image.open(io.BytesIO(normiere(z["foto"])[0])).convert("RGB"),
                             Z[a:a + 32]))
            out.append(np.concatenate([rechner[r](bs, ex) for r in RG], 1).astype(np.float16))
    X = np.concatenate(out)
    np.savez(ziel, X=X, ids=np.array([z["id"] for z in Z]))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--zusatz", default="v3z2", help="ein oder mehrere Zusatzpacks, mit Komma")
    ap.add_argument("--ohne-web", action="store_true", help="Vergleich: nur v2 trainieren")
    ap.add_argument("--koepfe", type=int, default=5)
    ap.add_argument("--ohne-gen", default="", help="Generatoren (mit |) aus Training UND val entfernen, "
                                                    "z. B. 'Nano Banana|OpenAI|aibooru:novelai' (Ausschlusstest)")
    ap.add_argument("--quelle-gruppe", default="", help="Quellen fuer die Gewichtung zusammenfassen, "
                    "z. B. 'midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney' (mehrere mit ;)")
    ap.add_argument("--rueckgrate", default="clip_mid,dino_mid",
                    help="Rueckgrate in RG_ALLE-Reihenfolge, z. B. clip_mid,dino_mid,siglip_mid (ab p7)")
    ap.add_argument("--seed-basis", type=int, default=0, help="Seeds basis..basis+koepfe-1 (Wiederholungslauf: 100)")
    ap.add_argument("--rating-ausgleich", action="store_true")
    ap.add_argument("--hart", default="", help="FAKTOR:ANTEIL, z. B. 3:0.02 — oberste 2 %% der Trainings-Menschen "
                                              "(nach --hart-ref) mit dreifachem Gewicht")
    ap.add_argument("--hart-ref", default="p8sx")
    ap.add_argument("--aug-name", default="aug", help="aug (norm.stoere) oder aug2 (norm.stoere2, ab p9)")
    ap.add_argument("--kalib-p2", action="store_true",
                    help="Schwelle auf allen menschlichen val-Bildern nach 2022 und Nicht-Anime "
                         "(Danbooru 2026 + danbooru_nichtanime + wallhaven_general), nicht nur Anime")
    args = ap.parse_args()
    global RG, AUGSUF
    AUGSUF = "_" + args.aug_name
    from lesen import RG_ALLE
    RG = tuple(sorted(args.rueckgrate.split(","), key=RG_ALLE.index))   # feste Spaltenreihenfolge
    print(f"Rueckgrate: {RG}", flush=True)

    p2, pz = Pack("v2"), Verbund(args.zusatz.split(","))
    A2, ok2 = lade("v2", True)
    Az, okz = pz.lade(True)
    T2, okt2 = lade("v2", False, "test")
    Tz, oktz = pz.lade(False, "test")
    # schutz.py auf allen Quellen (tags_pack.py): gesperrte Bilder nirgends verwenden
    wd2 = np.load(os.path.join(DATEN, "v2_wd.npz"))
    wdz = pz.wd()
    ok2 &= p2.ok & ~wd2["gesperrt"]
    okz &= pz.ok & ~wdz["gesperrt"]
    print(f"gesperrt: v2 {int(wd2['gesperrt'].sum())}, Zusatz {int(wdz['gesperrt'].sum())}", flush=True)
    if args.ohne_gen:
        namen = [n.strip() for n in args.ohne_gen.split("|") if n.strip()]
        r2, rz = gen_maske(p2, namen) & ok2, gen_maske(pz, namen) & okz
        ok2 &= ~gen_maske(p2, namen)
        okz &= ~gen_maske(pz, namen)
        print(f"Ausschluss {namen}: {int(r2.sum())} + {int(rz.sum())} Bilder aus Training/val/Test entfernt", flush=True)
        if r2.sum() + rz.sum() == 0:
            raise SystemExit("Ausschluss trifft nichts — Generatorname falsch?")

    def sel(p, ok, split, web=False):
        m = ok & (p.split == split)
        return np.flatnonzero(m)
    tr2, va2 = sel(p2, ok2, "train"), sel(p2, ok2, "val")
    trz, vaz = (np.array([], int), np.array([], int)) if args.ohne_web else \
        (sel(pz, okz, "train"), sel(pz, okz, "val"))
    print(f"train v2 {len(tr2)} + web {len(trz)}, val v2 {len(va2)} + web {len(vaz)}", flush=True)

    Xtr = np.concatenate([nimm(A2, tr2), nimm(Az, trz)]) if len(trz) else nimm(A2, tr2)
    ytr = np.r_[p2.label[tr2], pz.label[trz]]
    qtr = np.r_[p2.quelle[tr2], pz.quelle[trz]]
    Xva = np.concatenate([nimm(A2, va2), nimm(Az, vaz)]) if len(vaz) else nimm(A2, va2)
    yva = np.r_[p2.label[va2], pz.label[vaz]]
    qva = np.r_[p2.quelle[va2], pz.quelle[vaz]]
    hart = None
    if args.hart:
        # harte Negative: die menschlichen Trainingsbilder, die ein bestehender Kopf am ehesten fuer KI haelt,
        # bekommen mehr Gewicht (Fehlalarm-Recherche 29.09.; Rohmerkmale, vor dem Skalieren)
        from danbooru_test import bewerte, koepfe_von
        faktor, anteil = (float(x) for x in args.hart.split(":"))
        ref = koepfe_von(args.hart_ref)
        mh = np.flatnonzero(ytr == 0)
        z = np.concatenate([bewerte(np.asarray(Xtr[mh[i:i + 50000]], np.float32), ref)
                            for i in range(0, len(mh), 50000)])
        hart = mh[z >= np.quantile(z, 1 - anteil)]
        print(f"harte Negative: {len(hart)} von {len(mh)} Menschen (oberste {anteil:.1%} nach {args.hart_ref}) "
              f"x{faktor:g}", flush=True)
    sc = Skalierer().fit(Xtr, np.arange(len(Xtr)))
    Xtr, Xva = sc.transform(Xtr, inplace=True), sc.transform(Xva, inplace=True)
    if args.quelle_gruppe:
        # nur fuer die Gewichtung: mehrere Quellen zaehlen wie eine (sonst verduennen z. B. drei
        # Midjourney-Quellen den Anteil der Anime-Quellen innerhalb der KI-Klasse)
        qtr = qtr.copy()
        for regel in args.quelle_gruppe.split(";"):
            links, name = regel.split("=")
            qtr[np.isin(qtr, links.split(","))] = name
        print(f"Gewichtungsgruppen: {sorted(set(qtr[ytr == 1]))}", flush=True)
    wtr = gewichte3(ytr, qtr)
    if args.rating_ausgleich:
        rtr = np.r_[wd2["rating"][tr2], wdz["rating"][trz]]
        wtr = rating_ausgleich(wtr, ytr, rtr)
    if hart is not None:
        wtr[hart] *= faktor
        wtr = wtr / wtr.mean()

    # Testmengen
    te2 = np.flatnonzero(ok2 & (p2.split == "test"))
    tb = te2[okt2[te2] & np.isin(p2.quelle[te2], ["telegram", "ai_only"])]
    ta = te2[np.isin(p2.quelle[te2], ["wallhaven", "generiert"])]
    tw = np.flatnonzero(oktz & okz & (pz.split == "test"))
    Xb = sc.transform(nimm(T2, tb))
    Xa = sc.transform(nimm(A2, ta))
    Xw = sc.transform(nimm(Tz, tw))
    Xk2 = sc.transform(kanal2_merkmale())

    # Kalibrierung: menschliche Danbooru-Kunst von 2026 (Pack v3d, split val,
    # Kuenstler getrennt vom Test). Die alte Telegram-val-Schwelle (3 Kanaele vor
    # 2022) gab auf aktueller Kunst 6-7 % statt 1 % Fehlalarm (danbooru_test.py).
    pd = Pack("v3d")
    Td, okd = lade("v3d", False)
    dv = np.flatnonzero(okd & pd.ok & (pd.split == "val"))
    Xdv = sc.transform(nimm(Td, dv))

    s = {k: [] for k in ("va", "b", "a", "w", "k2", "dv")}
    gewichte_alle = []
    for seed in range(args.koepfe):
        t0 = time.time()
        vor, gew, best = kopf(Xtr, ytr, wtr, Xva, yva, args.seed_basis + seed)
        for k, X in (("va", Xva), ("b", Xb), ("a", Xa), ("w", Xw), ("k2", Xk2), ("dv", Xdv)):
            s[k].append(vor(X))
        gewichte_alle.append(gew)
        print(f"  Kopf {seed}: val AUC {best[0]:.4f} (Epoche {best[2]}), {time.time()-t0:.0f}s", flush=True)
    m = {k: np.mean(v, 0) for k, v in s.items()}

    # Schwellen auf dem Ensemble-Mittel; jeder Kopf traegt sie mit (pruefe.py mittelt)
    kal = m["dv"]
    if args.kalib_p2:
        mp2 = (yva == 0) & np.isin(qva, ["danbooru_nichtanime", "wallhaven_general"])
        kal = np.r_[m["dv"], m["va"][mp2]]
        print(f"Kalibrierung Phase 2: {len(m['dv'])} Danbooru-2026 + {int(mp2.sum())} Nicht-Anime-val", flush=True)
    schw = {f: float(np.quantile(kal, 1 - f)) for f in (0.01, 0.05)}
    tg = (qva == "telegram") & (yva == 0)
    schw_tg = {f: float(np.quantile(m["va"][tg], 1 - f)) for f in (0.01, 0.05)}
    print(f"Schwelle 1 %: {schw[0.01]:+.2f} (Danbooru-val, {len(dv)} Bilder) — "
          f"alt Telegram-val {schw_tg[0.01]:+.2f}", flush=True)
    for seed, gew in enumerate(gewichte_alle):
        np.savez(os.path.join(DATEN, f"kopf_{args.tag}_s{seed}.npz"), art="mlp", mean=sc.mean_,
                 scale=sc.scale_, w1=gew["0.weight"], b1=gew["0.bias"], w2=gew["3.weight"],
                 b2=gew["3.bias"], rueckgrate=np.array(RG), schwelle_1=schw[0.01], schwelle_5=schw[0.05],
                 ensemble=args.koepfe)
    yb = p2.label[tb]
    mensch_b = m["b"][yb == 0]
    erg = {"tag": args.tag, "ohne_web": args.ohne_web, "koepfe": args.koepfe, "schwellen": schw,
           "schwellen_telegram_alt": schw_tg,
           "rating_ausgleich": args.rating_ausgleich}

    def zeile(name, pos, einzeln=None):
        y = np.r_[np.zeros(len(mensch_b)), np.ones(len(pos))]
        r = {"n": int(len(pos)), "auc": auc(np.r_[mensch_b, pos], y), "tpr1": tpr_bei(mensch_b, pos),
             "tpr5": tpr_bei(mensch_b, pos, 0.05), "tr_val_schwelle": float((pos > schw[0.01]).mean())}
        if einzeln is not None:
            r["auc_einzeln"] = [auc(np.r_[e[0], e[1]], y) for e in einzeln]
        erg[name] = r
        ez = "" if einzeln is None else "  einzeln " + " ".join(f"{a:.3f}" for a in r["auc_einzeln"])
        print(f"  {name:<24} n={r['n']:>6}  AUC {r['auc']:.4f}  @1% {r['tpr1']:.1%}  @5% {r['tpr5']:.1%}"
              f"  val-Schw. {r['tr_val_schwelle']:.1%}{ez}", flush=True)

    print(f"\nFehlalarm auf den Testkanaelen bei val-Schwelle 1 %: {(mensch_b > schw[0.01]).mean():.2%}")
    erg["fa_test"] = float((mensch_b > schw[0.01]).mean())
    kanal1 = (yb == 1) & (p2.grp[tb] == "ai:ChatExport_2026-05-31 (1)")
    zeile("B KI-Kanal 1", m["b"][kanal1],
          [(sb[yb == 0], sb[kanal1]) for sb in s["b"]])
    zeile("K2 KI-Kanal 2", m["k2"], [(sb[yb == 0], sk) for sb, sk in zip(s["b"], s["k2"])])
    for q in ("civitai", "aibooru"):
        mq = pz.quelle[tw] == q
        if mq.any():
            zeile(f"WEB {q}", m["w"][mq])
    for b in sorted(set(pz.model[tw])):
        mb = pz.model[tw] == b
        if mb.sum() >= 40:
            zeile(f"   {b}", m["w"][mb])
    ya = p2.label[ta]
    erg["A"] = {"auc": auc(m["a"], ya), "tpr1": tpr_bei(m["a"][ya == 0], m["a"][ya == 1])}
    print(f"  A eigene Generatoren       AUC {erg['A']['auc']:.4f}  @1% {erg['A']['tpr1']:.1%}")
    np.save(os.path.join(DATEN, f"score_{args.tag}_B.npy"), np.c_[tb, m["b"]])
    np.save(os.path.join(DATEN, f"score_{args.tag}_K2.npy"), m["k2"])
    with open(os.path.join(DATEN, f"ergebnis_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(erg, f, indent=1)
    print("FERTIG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
