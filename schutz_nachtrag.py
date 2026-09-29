r"""schutz_nachtrag.py — apply the extended protection list (characters + franchises) retroactively.

  Telegram channels   caption from the manifest (characters and franchises)
  AIBooru repost bot  character/copyright tags of the post, fetched again from the AIBooru API
  Danbooru sets       for ratings s/q/e: character/copyright tags, fetched again from the Danbooru API
Hits: lab file deleted, manifest line reduced to "gesperrt", source path appended to daten/gesperrt_quellen.txt —
lesen.Pack excludes these files from EVERY pack. The image-based variant is schutz_wd_figuren.py.

    python schutz_nachtrag.py [--nur danbooru_nichtanime]
"""
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from lesen import DATEN                          # noqa: E402
from schutz import figurenregel_gilt, gesperrt_text   # noqa: E402

H = {"User-Agent": "ai-art-detector-lab/0.1 (research)"}
LISTE = os.path.join(DATEN, "gesperrt_quellen.txt")


def api_tags(basis, ids):
    """{id: "figuren serien"} fuer Booru-Posts, in Hunderter-Bloecken."""
    out = {}
    s = requests.Session()
    s.headers.update(H)
    ids = sorted(set(ids))
    for a in range(0, len(ids), 100):
        teil = ids[a:a + 100]
        for versuch in range(6):
            try:
                r = s.get(f"{basis}/posts.json", params={"tags": "id:" + ",".join(map(str, teil)) + " status:any",
                                                       "limit": 200}, timeout=90)
                if r.status_code == 200:
                    for p in r.json():
                        out[p["id"]] = (p.get("tag_string_character", "") + " " + p.get("tag_string_copyright", ""),
                                        p.get("rating"))
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(15 * (versuch + 1))
        time.sleep(0.6)
    return out


def bereinige(mf, treffer, pfad_feld, id_feld):
    """Manifest umschreiben: Treffer auf {id, grund: gesperrt}; Dateien loeschen; Pfade merken."""
    E = [json.loads(z) for z in open(mf, encoding="utf-8")]
    pfade, neu = [], []
    for e in E:
        if e.get(id_feld) in treffer and not e.get("grund"):
            for f in pfad_feld:
                p = e.get(f)
                if p:
                    pfade.append(p)
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            e = {id_feld: e.get(id_feld), "grund": "gesperrt (nachtrag 2026-09-26)"}
        neu.append(e)
    tmp = mf + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for e in neu:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, mf)
    return pfade


def main():
    nur = set(sys.argv[sys.argv.index("--nur") + 1].split(",")) if "--nur" in sys.argv else None
    will = lambda q: nur is None or q in nur
    alle_pfade = []
    # 1 Telegram-Kanaele mit Beitragstext
    wurzel = f"{DATASETS}/telegram_ki"
    for name in sorted(os.listdir(wurzel)):
        mf = os.path.join(wurzel, name, "manifest.jsonl")
        if not os.path.exists(mf) or name == "aibooru_repost_bot" or not will("telegram"):
            continue
        E = [json.loads(z) for z in open(mf, encoding="utf-8")]
        # ohne gespeichertes Rating (Import vor 2026-09-26) vorsichtshalber wie nicht jugendfrei
        treffer = {e["msg"] for e in E if e.get("msg") is not None and figurenregel_gilt(e.get("rating", "s"))
                   and gesperrt_text(e.get("text"))}
        p = bereinige(mf, treffer, ("datei",), "msg") if treffer else []
        alle_pfade += p
        print(f"  {name:<22} {len(p):>5} Bilder gesperrt", flush=True)

    # 2 AIBooru-Bot: Figuren-/Serien-Tags von der API
    if will("aibooru"):
        mf = f"{DATASETS}/telegram_ki/aibooru_repost_bot/manifest.jsonl"
        E = [json.loads(z) for z in open(mf, encoding="utf-8")]
        tags = api_tags("https://aibooru.online", [e["id"] for e in E if e.get("telegram")])
        treffer = {i for i, (t, r) in tags.items() if figurenregel_gilt(r) and gesperrt_text(t)}
        p = bereinige(mf, treffer, ("telegram", "original"), "id")
        alle_pfade += p
        # kanal2.json (train3.py, Test K2) zeigt auf dieselben Dateien
        k2 = os.path.join(DATEN, "kanal2.json")
        Z = [z for z in json.load(open(k2, encoding="utf-8")) if z["id"] not in treffer]
        json.dump(Z, open(k2, "w", encoding="utf-8"), indent=0, ensure_ascii=False)
        print(f"  {'aibooru_repost_bot':<22} {len(treffer):>5} Posts gesperrt", flush=True)

    # 3 Danbooru-Saetze, nur q/e
    for name in ("danbooru_ab2023", "danbooru_train", "danbooru_nichtanime"):
        mf = os.path.join(f"{DATASETS}", name, "manifest.jsonl")
        if not os.path.exists(mf) or not will(name):
            continue
        E = [json.loads(z) for z in open(mf, encoding="utf-8")]
        qe = [e["id"] for e in E if e.get("datei") and e.get("rating") in ("s", "q", "e")]
        tags = api_tags("https://danbooru.donmai.us", qe)
        treffer = {i for i, (t, r) in tags.items() if figurenregel_gilt(r) and gesperrt_text(t)}
        p = bereinige(mf, treffer, ("datei",), "id")
        alle_pfade += p
        print(f"  {name:<22} {len(treffer):>5} von {len(qe)} s/q/e-Posts gesperrt", flush=True)

    with open(LISTE, "a", encoding="utf-8") as f:
        for p in alle_pfade:
            f.write(p + "\n")
    print(f"{len(alle_pfade)} Dateien entfernt und in {LISTE} eingetragen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
