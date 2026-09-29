"""pfade.py — all machine-specific locations in one place.

Every path can be overridden with an environment variable, or all at once in an optional file
`pfade_lokal.py` next to this one (git-ignored; plain assignments, e.g. DATASETS = "D:/datasets").

    DETEKTOR_DATEN      working data: packs, embeddings, trained heads, logs, results
    DETEKTOR_DATASETS   downloaded/sorted image sources, one folder per source with manifest.jsonl
    DETEKTOR_NAS        local wallpaper archive (Wallhaven etc.), used only for human pre-2022 art
    DETEKTOR_SCHLUESSEL folder with API key files (civitai_key.txt); keys can also come from env vars
    DETEKTOR_PROJEKT    the surrounding image-sorting project (only for the scripts in archive/ and
                        the Telegram inbox of aufbau2.py)
"""
import os

_HIER = os.path.dirname(os.path.abspath(__file__))

DATEN = os.environ.get("DETEKTOR_DATEN", os.path.join(_HIER, "daten"))
DATASETS = os.environ.get("DETEKTOR_DATASETS", os.path.join(_HIER, "datasets"))
NAS = os.environ.get("DETEKTOR_NAS", os.path.join(_HIER, "nas"))
SCHLUESSEL = os.environ.get("DETEKTOR_SCHLUESSEL", _HIER)
PROJEKT = os.environ.get("DETEKTOR_PROJEKT", os.path.dirname(os.path.dirname(_HIER)))

try:                                   # lokale Pfade des Labor-Rechners, nicht im Repository
    from pfade_lokal import *          # noqa: F401,F403
except ImportError:
    pass
for _n in ("DATEN", "DATASETS", "NAS", "SCHLUESSEL", "PROJEKT"):     # Umgebungsvariablen gewinnen immer
    if f"DETEKTOR_{_n}" in os.environ:
        globals()[_n] = os.environ[f"DETEKTOR_{_n}"]

TELEGRAM_KI = os.path.join(DATASETS, "telegram_ki")
V1_DB = os.path.join(PROJEKT, "lab", "dataset", "dataset.db")
TELEGRAM_EINGANG = os.path.join(PROJEKT, "unbearbeitet")
