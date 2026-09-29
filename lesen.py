r"""lesen.py — access to the packed, normalized images and to cached features.

Every later step reads ONLY through here, so every model sees exactly the same bytes and nobody reads the raw
files with their revealing sizes by accident. Pack.ok excludes preview thumbnails, images blocked by the
child-protection rules (gesperrt_quellen.txt) and images found to be mislabeled human (falsch_menschlich.txt).
kanal_X() returns the features of a Telegram test channel (CLIP | DINOv2 | SigLIP columns, blocked images
removed).
"""
import io
import json
import mmap
import os

import numpy as np
from PIL import Image

from pfade import DATEN, TELEGRAM_KI   # noqa: E402,F401  (Pfade zentral, per Umgebungsvariable)

# Merkmale liegen immer in dieser Reihenfolge nebeneinander: CLIP-Mitte (5120) | DINO-Mitte (5120) |
# SigLIP-Mitte (5760, ab p7). Ein Kopf nimmt die ersten so vielen Spalten, wie er beim Training hatte
# (danbooru_test.bewerte) — Koepfe mit zwei Rueckgraten laufen so unveraendert weiter.
RG_ALLE = ("clip_mid", "dino_mid", "siglip_mid")
RG_DIM = {"clip_mid": 5120, "dino_mid": 5120, "siglip_mid": 5760}


def spalten_fuer(rueckgrate):
    """Spaltenindizes eines Kopfes (Rueckgrate in RG_ALLE-Reihenfolge) im vollen Merkmalsblock."""
    off = np.cumsum([0] + [RG_DIM[r] for r in RG_ALLE])
    return np.concatenate([np.arange(off[RG_ALLE.index(r)], off[RG_ALLE.index(r) + 1]) for r in rueckgrate])


def rueckgrate_da(praefix, rest):
    """Die Rueckgrate aus RG_ALLE, fuer die emb_<rg><rest>.npy existiert — ohne Luecke von vorn."""
    import os
    da = []
    for r in RG_ALLE:
        if not os.path.exists(os.path.join(DATEN, f"{praefix}{r}{rest}.npy")):
            break
        da.append(r)
    return da


def kanal_X(name):
    """Kanal-Merkmale: CLIP+DINO aus kanal_<name>_emb.npz, SigLIP aus kanal_<name>_siglip.npz
    angehaengt, wenn vorhanden (gleiche Reihenfolge der Pfade wird geprueft)."""
    import os
    d = np.load(os.path.join(DATEN, f"kanal_{name}_emb.npz"))
    X, pfade = d["X"], d["pfade"]
    s = os.path.join(DATEN, f"kanal_{name}_siglip.npz")
    if os.path.exists(s):
        e = np.load(s)
        if len(e["pfade"]) == len(pfade) and (e["pfade"][:50] == pfade[:50]).all():
            X = np.concatenate([X, e["X"]], 1)
    # nachtraeglich gesperrte Bilder (schutz_nachtrag.py, schutz_wd_figuren.py) zaehlen in keinem Test
    liste = os.path.join(DATEN, "gesperrt_quellen.txt")
    if os.path.exists(liste):
        gesperrt = {z.strip().lower() for z in open(liste, encoding="utf-8") if z.strip()}
        m = np.array([str(p).lower() not in gesperrt for p in pfade])
        if not m.all():
            X, pfade = X[m], pfade[m]
    return X, pfade


class Pack:
    def __init__(self, stamm="norm512"):
        self.stamm = stamm
        with open(os.path.join(DATEN, stamm + "_manifest.json"),
                  encoding="utf-8") as f:
            self.m = json.load(f)
        self.off = np.load(os.path.join(DATEN, stamm + "_off.npy"))
        self.len = np.load(os.path.join(DATEN, stamm + "_len.npy"))
        self.size = np.load(os.path.join(DATEN, stamm + "_size.npy"))
        self._f = open(os.path.join(DATEN, stamm + ".bin"), "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.label = np.array([z["label"] for z in self.m], np.int8)
        self.model = np.array([z["model"] for z in self.m])
        self.split = np.array([z["split"] for z in self.m])
        self.wh_id = np.array([z["wh_id"] for z in self.m])
        # v2 kennt vier Quellen (aufbau2.py); v1 nur Wallhaven und Generatoren
        self.quelle = np.array([z.get("quelle", "wallhaven" if z["label"] == 0 else "generiert")
                                for z in self.m])
        self.grp = np.array([z.get("grp", "") or "" for z in self.m])
        # Telegram-Vorschaubilder (*_thumb.jpg): in "Ai Only" die Haelfte der
        # Bilder, bei den Menschen-Kanaelen fast keine — klein und stark
        # komprimiert waeren sie ein Leck ("Vorschaubild = KI"), und im Test
        # zaehlten sie jeden Post doppelt. Gefunden 2026-09-25 (fehler.py).
        self.thumb = np.array(["_thumb." in (z.get("src") or "").lower() for z in self.m])
        # Nachtraeglich gesperrte Quelldateien (schutz_nachtrag.py): gelten in jedem Pack als
        # nicht vorhanden — nie trainiert, nie getestet, nie angezeigt.
        gesperrt = set()
        p = os.path.join(DATEN, "gesperrt_quellen.txt")
        if os.path.exists(p):
            gesperrt = {z.strip().lower() for z in open(p, encoding="utf-8") if z.strip()}
        self.gesperrt = np.array([(z.get("src") or "").lower() in gesperrt for z in self.m])
        # Falsch als menschlich gefuehrte Bilder (danbooru_nachpruefen.py: inzwischen als KI getaggt) —
        # getrennt von der Schutz-Sperrliste, aber genauso ueberall ausgeschlossen.
        falsch = set()
        p = os.path.join(DATEN, "falsch_menschlich.txt")
        if os.path.exists(p):
            falsch = {z.strip().lower() for z in open(p, encoding="utf-8") if z.strip()}
        self.falsch_menschlich = np.array([(z.get("src") or "").lower() in falsch for z in self.m])
        self.ok = (self.off >= 0) & ~self.thumb & ~self.gesperrt & ~self.falsch_menschlich

    def __len__(self):
        return len(self.m)

    def bytes(self, i):
        return self._mm[self.off[i]:self.off[i] + self.len[i]]

    def bild(self, i, aug=False):
        b = Image.open(io.BytesIO(self.bytes(i))).convert("RGB")
        if aug == 2:
            from norm import stoere2
            b = stoere2(b, i)
        elif aug:
            from norm import stoere
            b = stoere(b, i)
        return b

    def wo(self, split=None, model=None, label=None):
        """Indizes nach Split, Modell, Klasse — in Packreihenfolge, damit
        das Lesen am Stueck bleibt."""
        k = self.ok.copy()
        if split is not None:
            k &= np.isin(self.split, [split] if isinstance(split, str) else split)
        if model is not None:
            k &= np.isin(self.model, [model] if isinstance(model, str) else model)
        if label is not None:
            k &= self.label == label
        return np.flatnonzero(k)
