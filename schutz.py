r"""schutz.py — no sexualized depictions of minors in any dataset.

Hard rule for every source (web, boorus, Telegram, Hugging Face datasets). Items that match are never
stored, embedded, trained on or displayed. Details and numbers: docs/SAFETY.md.

  SPERRE     tags indicating minors (loli, shota, child, ...) — blocked at EVERY rating, checked against the
             source's own tags and the WD tagger's predictions (threshold SCHWELLE = 0.2, more sensitive than
             the usual 0.35: better one image too many rejected)
  FIGUREN    characters who are canonically minors (e.g. school students) — blocked for non-general images
  FRANCHISEN franchises whose casts are predominantly minors — blocked for non-general images
             (figurenregel_gilt: WD sensitive/questionable/explicit, booru s/q/e)

The character/franchise lists are matched against captions, booru character/copyright tags, prompts and the
characters the WD tagger recognizes in the image (schutz_wd_figuren.py).
"""
SPERRE = {
    "loli", "shota", "child", "female_child", "male_child", "aged_down", "toddler", "baby",
    "infant", "young", "oppai_loli", "onee-loli", "onee-shota", "onii-shota", "child_on_child",
    "lolidom", "kindergarten_uniform", "randoseru", "elementary_school", "lolicon", "shotacon",
    "kono_lolicon_domome", "child_carry",
}
SCHWELLE = 0.2


# Figuren, die im Original eindeutig minderjaehrig sind, aber oft erwachsen gezeichnet
# werden — das faengt der Tag-Filter nicht. Greift auf Beitragstexte (Telegram-
# Beschriftung, Hashtags). Bewusst klein und eindeutig; eine Ergaenzung, keine Garantie.
# Anlass: ein Telegram-Testkanal (Anime-Kanal C) mit Posts zu einer solchen Figur, 2026-09-26.
FIGUREN = ("marin kitagawa", "kitagawa marin", "marinkitagawa",
           "nezuko", "kanao tsuyuri", "tsuyuri kanao", "anya forger", "shouko komi", "komi shouko",
           "chika fujiwara", "fujiwara chika", "kaguya shinomiya", "shinomiya kaguya",
           "hayasaka ai", "ai hayasaka", "hayasak ai", "miko iino", "iino miko",
           "bocchi", "hitori gotou", "gotou hitori", "nijika ijichi", "ryo yamada", "ikuyo kita",
           # Higurashi (Mittelschule), Genshin/Honkai-Kinderfiguren, Fate-Kinderfiguren u. a.
           "sonozaki", "furude rika", "rika furude", "houjou satoko", "satoko houjou", "ryuuguu rena",
           "klee", "qiqi", "nahida", "sayu (genshin", "diona (genshin", "yaoyao", "bronya zaychik",
           "illyasviel", "illya ", "chloe von einzbern", "miyu edelfelt", "jack the ripper (fate",
           "abigail williams", "nursery rhyme (fate", "megumin", "kanna kamui", "shinobu oshino",
           "hoshino ruby", "ruby hoshino", "arima kana", "kana arima", "madoka kaname", "kaname madoka",
           "homura akemi", "akemi homura", "sayaka miki", "kyoko sakura", "mami tomoe",
           # 2026-09-27 (Anlass: Anime-Kanal F, Figuren per WD-Figurentag erkannt, Posts ohne Text):
           # Schueler bzw. im Original eindeutig unter 18 — Evangelion, Persona 3-5, Jujutsu Kaisen,
           # My Hero Academia, Nagatoro, Re:Zero-Zwillinge (17), Kill la Kill (Oberschule)
           "ayanami rei", "rei ayanami", "souryuu asuka langley", "asuka langley", "ikari shinji",
           "takeba yukari", "yamagishi fuuka", "kujikawa rise", "satonaka chie", "amagi yukiko",
           "shirogane naoto", "takamaki ann", "niijima makoto", "sakura futaba", "okumura haru",
           "yoshizawa kasumi", "kugisaki nobara", "zen'in maki", "zenin maki", "uraraka ochako",
           "asui tsuyu", "jirou kyouka", "yaoyorozu momo", "ashido mina", "hagakure tooru",
           "nagatoro", "rem (re:zero)", "ram (re:zero)", "matoi ryuuko")
# Franchises, deren Hauptfiguren ueberwiegend minderjaehrig angelegt sind. Nur fuer
# anzuegliche/explizite Bilder gedacht (Aufrufer entscheidet, siehe gesperrt_text).
FRANCHISEN = ("higurashi", "umamusume", "uma musume", "blue archive", "love live", "lovelive",
              "touhou", "mahou shoujo madoka", "madoka magica", "precure", "pretty cure", "k-on",
              "lucky star", "bang dream", "oshi no ko", "kaguya-sama", "kaguya sama",
              "bocchi the rock", "spy x family", "spy family", "idolmaster cinderella",
              "prisma illya", "made in abyss", "yuru camp", "non non biyori", "gochuumon")


def figurenregel_gilt(rating):
    """Die Figuren-/Franchise-Liste gilt nur fuer Bilder, die nicht jugendfrei sind
    (WD: sensitive/questionable/explicit; Booru: s/q/e). Jugendfreie Darstellungen
    minderjaehriger Figuren sind unproblematisch (Entscheidung User, 2026-09-26).
    Die Tag-Sperre (SPERRE) gilt dagegen immer, fuer jedes Rating."""
    return str(rating).lower() not in ("general", "g")


def _norm(text):
    return " " + (text or "").lower().replace("#", " ").replace("_", " ").replace("  ", " ") + " "


def gesperrt_text(text, franchisen=True):
    """Beitragstext / Booru-Figuren- und Serien-Tags -> Treffer aus FIGUREN (und FRANCHISEN,
    wenn franchisen=True). Leer, wenn keine."""
    t = _norm(text)
    treffer = [f for f in FIGUREN if f in t]
    if franchisen:
        treffer += [f for f in FRANCHISEN if f in t]
    return sorted(treffer)


def gesperrt_tags(tags):
    """Booru-Tags (Liste oder Leerzeichen-String) -> gesperrte Treffer."""
    if isinstance(tags, str):
        tags = tags.split()
    return sorted(SPERRE & set(tags))


def gesperrt_wd(allg):
    """WD-Tagger-Ausgabe {tag: wahrscheinlichkeit} (mit general_threshold <= SCHWELLE
    abgefragt) -> gesperrte Treffer."""
    return sorted(t for t, p in allg.items() if t in SPERRE and p >= SCHWELLE)
