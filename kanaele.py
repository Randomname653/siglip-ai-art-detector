"""kanaele.py — names of the Telegram AI test channels.

Public documents use neutral labels ("Anime channel A", "Midjourney channel B", ...). The mapping from the
lab's channel folders to real channel names and labels lives in an optional, git-ignored `kanaele_lokal.py`:

    KANAELE = {"<folder>": ("<real channel name>", "<public label>", <is anime: bool>), ...}
    WEITERE = {"<other name not to publish>": "<replacement>", ...}

Without it every channel is shown under its folder name.
"""
import re

try:
    from kanaele_lokal import KANAELE, WEITERE   # noqa: F401
except ImportError:
    KANAELE, WEITERE = {}, {}

ANIME = tuple(k for k, v in KANAELE.items() if v[2])
NAME = {k: v[0] for k, v in KANAELE.items()}          # echte Namen (lokale Berichte)
LABEL = {k: v[1] for k, v in KANAELE.items()}         # neutrale Bezeichnungen (oeffentliche Dokumente)


def label(schluessel):
    return LABEL.get(schluessel, schluessel)


def oeffentlich(text):
    """Echte Kanalnamen und Ordnerschluessel in einem Text durch die neutralen Bezeichnungen ersetzen."""
    paare = [(v[0], v[1]) for v in KANAELE.values()] + list(WEITERE.items())
    for alt, neu in sorted(paare, key=lambda p: -len(p[0])):
        text = text.replace(alt, neu)
    for k in sorted(LABEL, key=len, reverse=True):
        text = re.sub(rf"(?<![\w/]){re.escape(k)}(?![\w/])", LABEL[k], text)
    return text
