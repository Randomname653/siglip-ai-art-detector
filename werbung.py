r"""werbung.py — find ads in a sorted-in Telegram AI channel.

Not every ad comes "via @...bot". Two independent hints:
  text   ad language, links to other bots/sites, prices, very long texts (credited posts are skipped)
  image  WD tags typical of ad graphics (logo, text_focus, qr_code, screenshot, user_interface, ...)
Output: candidate list + contact sheet for review. Nothing is removed automatically — the decision is made
after looking (--raus <msg-ids>).

    python werbung.py <channel>
    python werbung.py <channel> --raus 123,456
"""
import json
import os
import re
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfade import TELEGRAM_KI                    # noqa: E402

WORTE = re.compile(r"subscribe|sponsor|promo|advert|реклам|подпис|скидк|купи|price|\$\d|€\d|\d\s?\$|"
                   r"download the app|app store|google play|wallet|crypto|\bearn\b|bonus|casino|\bbet\b|"
                   r"click|limited|offer|free trial", re.I)
BILD_TAGS = {"logo", "text_focus", "qr_code", "screenshot", "user_interface", "icon", "web_address"}
# Ein Post mit Urheber-Nachweis ist fast nie Werbung — nur Posts OHNE ihn pruefen
# (AI Art Gallery: 814 Kandidaten mit, 72 ohne, alle 72 waren Werbung/Memes/Ansagen)
CREDIT = re.compile(r"#from|✅|#sent by|#made by|created using", re.I)
ANSAGE = re.compile(r"subscriber|welcome|#meme|#aitools|#ainews|released|realsed|announce|giveaway|"
                    r"we reached|happy new|merry|addlist", re.I)


EIGEN = []   # eigene Kanal-Links (Einladung, Namen) zaehlen als Signatur, nicht als Werbung


def main():
    name = sys.argv[1]
    if "--eigen" in sys.argv:
        EIGEN.extend(sys.argv[sys.argv.index("--eigen") + 1].split(","))
    ziel = os.path.join(TELEGRAM_KI, name)
    mf = os.path.join(ziel, "manifest.jsonl")
    E = [json.loads(z) for z in open(mf, encoding="utf-8")]
    if "--raus" in sys.argv:
        raus = {int(x) for x in sys.argv[sys.argv.index("--raus") + 1].split(",")}
        bleibt = [e for e in E if e["msg"] not in raus]
        for e in E:
            if e["msg"] in raus and os.path.exists(e["datei"]):
                os.remove(e["datei"])            # Lab-Kopie; das Original im Export bleibt
        with open(mf, "w", encoding="utf-8") as f:
            for e in bleibt:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{len(E) - len(bleibt)} Werbebilder aus dem Lab entfernt, {len(bleibt)} bleiben")
        return 0

    from imgutils.tagging import get_wd14_tags
    kand = []
    for e in E:
        grund = []
        t = e.get("text") or ""
        if CREDIT.search(t):
            continue
        if WORTE.search(t) or ANSAGE.search(t):
            grund.append("Text")
        links = re.findall(r"(?:https?://|t\.me/)\S+", t)
        fremd = [l for l in links if name.replace("_", "").lower()[:6] not in l.lower().replace("_", "")
                 and not any(x in l for x in EIGEN)]
        if fremd:
            grund.append("Link")
        _, allg, _ = get_wd14_tags(e["datei"], model_name="EVA02_Large", general_threshold=0.5)
        tags = sorted(BILD_TAGS & set(allg))
        if tags:
            grund.append("Bild:" + "+".join(tags))
        if grund:
            kand.append((e, grund))
    print(f"{len(kand)} Kandidaten von {len(E)} Bildern")
    for e, g in kand:
        print(f"  msg {e['msg']:>6}  {' | '.join(g):<45} {(e.get('text') or '')[:70]!r}")
    kacheln = []
    for e, _ in kand[:96]:
        im = Image.open(e["datei"]).convert("RGB")
        im.thumbnail((180, 180))
        k = Image.new("RGB", (180, 200), (24, 24, 28))
        k.paste(im, ((180 - im.width) // 2, 0))
        from PIL import ImageDraw
        ImageDraw.Draw(k).text((4, 184), str(e["msg"]), fill=(230, 230, 230))
        kacheln.append(k)
    if kacheln:
        sp = 8
        c = Image.new("RGB", (180 * sp, 200 * ((len(kacheln) + sp - 1) // sp)), (24, 24, 28))
        for j, k in enumerate(kacheln):
            c.paste(k, ((j % sp) * 180, (j // sp) * 200))
        aus = os.path.join(ziel, "werbung_kandidaten.jpg")
        c.save(aus, quality=85)
        print(f"-> {aus}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
