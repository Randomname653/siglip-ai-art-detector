r"""einsortieren_json.py — copy a Telegram AI channel (JSON export) into the lab, in order.

    python einsortieren_json.py "<export folder>" <short name> [result_repariert.json]

Target: <DATASETS>/telegram_ki/<short name>/
    telegram/<msg-id>.jpg      Telegram-compressed photo
    original/<msg-id>.<ext>    image sent as a file (if present)
    manifest.jsonl             per image: date, caption (prompt), Midjourney version, WD rating, forwarded-from
Rules: posts "via @...bot" are dropped (ads); preview thumbnails are never copied; schutz.py on every image
(WD tags, and the character/franchise list on captions for non-general images); an album's caption is only on
its first image and is passed on to the following images of the same second. Only copies — nothing in the
export folder is changed. A truncated export JSON can be repaired first (close after the last full message).
"""
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from schutz import SCHWELLE, figurenregel_gilt, gesperrt_text, gesperrt_wd   # noqa: E402

BILD = (".jpg", ".jpeg", ".png", ".webp")


VERSION = re.compile(r"--(v|version|niji)\s*([\d.]+|test\w*)|--(testp?)\b|"
                     r"(?:version\s*midjourney|midjourney\s*(?:version|v))\s*:?\s*([\d.]+)", re.I)


def version(t):
    """Midjourney-Parameter aus dem Prompt: '--v 5.2' -> 'v5.2', '--niji 6' -> 'niji6',
    '--test' -> 'test'. Leer, wenn der Prompt keine nennt."""
    for m in VERSION.finditer(t or ""):
        if m.group(3):
            return m.group(3).lower()
        if m.group(4):
            return "v" + m.group(4)
        return ("niji" if m.group(1).lower() == "niji" else "v") + m.group(2).lower()
    return ""


def text(m):
    t = m.get("text", "")
    return "".join(x if isinstance(x, str) else x.get("text", "") for x in t) if isinstance(t, list) else t


def main():
    quelle, name = sys.argv[1], sys.argv[2]
    # --nur-general: fuer Kanaele mit Schwerpunkt auf expliziten Inhalten. Dort tauchen Figuren auf,
    # die im Original minderjaehrig sind (z. B. Marin Kitagawa), erwachsen gezeichnet —
    # das faengt die Tag-Sperre nicht. Dann nur Bilder mit WD-Rating "general".
    nur_general = "--nur-general" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--nur-general"]
    ziel = os.path.join(f"{DATASETS}/telegram_ki", name)
    # optional: anderer JSON-Name, z. B. result_repariert.json bei abgebrochenem Export
    d = json.load(open(os.path.join(quelle, sys.argv[3] if len(sys.argv) > 3 else "result.json"), encoding="utf-8"))
    M = [m for m in d["messages"] if m.get("type") == "message"]

    posten, letzte = [], {"date": None, "text": ""}
    werbung = 0
    for m in M:
        t = text(m)
        if t:
            letzte = {"date": m["date"], "text": t}
        elif m["date"] == letzte["date"]:
            t = letzte["text"]                   # Album: Beschriftung vom ersten Bild
        if m.get("via_bot"):
            werbung += 1
            continue
        for art, feld in (("telegram", "photo"), ("original", "file")):
            p = m.get(feld)
            if not p or "_thumb" in p or not p.lower().endswith(BILD) or p.startswith("("):
                continue
            posten.append({"msg": m["id"], "art": art, "quelle": os.path.join(quelle, p),
                           "date": m["date"], "text": t, "fwd": m.get("forwarded_from")})
    posten = [p for p in posten if os.path.exists(p["quelle"])]
    print(f"{d['name']}: {len(M)} Nachrichten, {werbung} per Bot (raus), {len(posten)} Bilddateien", flush=True)

    from imgutils.tagging import get_wd14_tags

    def pruefe(p):
        try:
            rating, alle, _ = get_wd14_tags(p["quelle"], model_name="EVA02_Large", general_threshold=SCHWELLE)
            p["rating"] = max(rating, key=rating.get)
            if nur_general and p["rating"] != "general":
                return p, False
            if figurenregel_gilt(p["rating"]) and gesperrt_text(p["text"]):
                return p, False           # Figur/Serie, im Original minderjaehrig, Bild nicht jugendfrei
            return p, not gesperrt_wd(alle)
        except Exception:
            return p, False
    with ThreadPoolExecutor(6) as ex:
        erg = list(ex.map(pruefe, posten))
    gut = [p for p, ok in erg if ok]
    print(f"WD-Sperre/Fehler verwerfen {len(posten) - len(gut)}, kopiere {len(gut)}", flush=True)

    for u in ("telegram", "original"):
        os.makedirs(os.path.join(ziel, u), exist_ok=True)
    with open(os.path.join(ziel, "manifest.jsonl"), "w", encoding="utf-8") as mf:
        for p in gut:
            dst = os.path.join(ziel, p["art"], f"{p['msg']}{os.path.splitext(p['quelle'])[1].lower()}")
            if not os.path.exists(dst):
                shutil.copy2(p["quelle"], dst)
            mf.write(json.dumps({"msg": p["msg"], "art": p["art"], "datei": dst, "date": p["date"],
                                 "text": p["text"], "version": version(p["text"]), "rating": p.get("rating"),
                                 "fwd": p["fwd"], "kanal": d["name"],
                                 "herkunft": quelle}, ensure_ascii=False) + "\n")
    print(f"-> {ziel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
