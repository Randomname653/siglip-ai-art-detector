r"""einsortieren_kanal2.py — den AIBooru-Repost-Kanal geordnet ins Lab kopieren.

Aus den Telegram-Exporten in Downloads (zwei Ordner, der Download wurde einmal
neu gestartet) nach
    <DATASETS>\telegram_ki\aibooru_repost_bot\
        telegram\<post-id>.jpg     Telegram-komprimiertes Foto
        original\<post-id>.<ext>  Originaldatei aus der Folgenachricht
        manifest.jsonl            je Post: Rating, Modell-/Meta-Tags, Herkunftsordner
Nur Posts aus daten/kanal2.json (Booru-Tag-Sperre schon angewandt) und nur,
wenn auch die WD-Sperre auf dem Foto nichts findet. Thumbnails und gesperrte
Posts werden nicht kopiert. Danach zeigt kanal2.json auf die Lab-Kopien.

Es wird nur kopiert, nichts in Downloads geloescht.

    python lab/detektor/einsortieren_kanal2.py
"""
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import DATASETS                       # noqa: E402
from lesen import DATEN                          # noqa: E402
from schutz import SCHWELLE, gesperrt_wd         # noqa: E402

ZIEL = f"{DATASETS}/telegram_ki/aibooru_repost_bot"


def main():
    from imgutils.tagging import get_wd14_tags
    liste = os.path.join(DATEN, "kanal2.json")
    Z = json.load(open(liste, encoding="utf-8"))
    for u in ("telegram", "original"):
        os.makedirs(os.path.join(ZIEL, u), exist_ok=True)

    def pruefe(z):
        try:
            _, alle, _ = get_wd14_tags(z["foto"], model_name="EVA02_Large", general_threshold=SCHWELLE)
            return z, not gesperrt_wd(alle)
        except Exception:
            return z, False
    with ThreadPoolExecutor(6) as ex:
        erg = list(ex.map(pruefe, Z))
    gut = [z for z, ok in erg if ok]
    print(f"{len(Z)} Posts, WD-Sperre verwirft {len(Z) - len(gut)}, kopiere {len(gut)}", flush=True)

    neu = []
    with open(os.path.join(ZIEL, "manifest.jsonl"), "w", encoding="utf-8") as mf:
        for z in gut:
            f_ziel = os.path.join(ZIEL, "telegram", f"{z['id']}.jpg")
            o_ziel = os.path.join(ZIEL, "original", f"{z['id']}{os.path.splitext(z['datei'])[1].lower()}")
            if not os.path.exists(f_ziel):
                shutil.copy2(z["foto"], f_ziel)
            if not os.path.exists(o_ziel):
                shutil.copy2(z["datei"], o_ziel)
            e = {"id": z["id"], "rating": z["rating"], "model": z["model"], "meta": z["meta"],
                 "telegram": f_ziel, "original": o_ziel, "herkunft": os.path.dirname(os.path.dirname(z["foto"])),
                 "aibooru": f"https://aibooru.online/posts/{z['id']}"}
            mf.write(json.dumps(e, ensure_ascii=False) + "\n")
            neu.append(dict(z, foto=f_ziel, datei=o_ziel))
    with open(liste, "w", encoding="utf-8") as f:
        json.dump(neu, f, indent=0, ensure_ascii=False)
    print(f"-> {ZIEL}, kanal2.json zeigt jetzt auf die Lab-Kopien ({len(neu)} Posts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
