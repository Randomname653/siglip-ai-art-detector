r"""schutz_nachpruefen.py — bereits gespeicherte CivitAI-Bilder gegen schutz.py pruefen.

Die Sperre kam erst nach den ersten Downloads dazu (2026-09-25). Jedes
gespeicherte Bild wird neu getaggt; Treffer werden geloescht und im Manifest
auf ID + "gesperrt" reduziert.

    python lab/detektor/schutz_nachpruefen.py
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from civitai import MANIFEST                     # noqa: E402
from schutz import SCHWELLE, gesperrt_wd         # noqa: E402


def main():
    from imgutils.tagging import get_wd14_tags
    E = [json.loads(z) for z in open(MANIFEST, encoding="utf-8")]
    zu_pruefen = [e for e in E if e.get("datei")]
    print(f"{len(zu_pruefen)} gespeicherte Bilder pruefen", flush=True)

    def pruefe(e):
        try:
            _, alle, _ = get_wd14_tags(e["datei"], model_name="EVA02_Large", general_threshold=SCHWELLE)
            return e["id"], bool(gesperrt_wd(alle))
        except Exception:
            return e["id"], False

    with ThreadPoolExecutor(8) as ex:
        treffer = {i for i, g in ex.map(pruefe, zu_pruefen) if g}
    neu = []
    for e in E:
        if e["id"] in treffer:
            try:
                os.remove(e["datei"])
            except OSError:
                pass
            e = {"id": e["id"], "basis": e["basis"], "user": e.get("user"), "grund": "gesperrt"}
        neu.append(e)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for e in neu:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, MANIFEST)
    print(f"gesperrt und geloescht: {len(treffer)} von {len(zu_pruefen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
