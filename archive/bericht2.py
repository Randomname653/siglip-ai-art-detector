r"""bericht2.py — der Laborbericht zu v2, Zahlen direkt aus den Ergebnisdateien.

v2 = der Detektor, der allein stehen soll (Entscheidung des Users, 24.09.):
Wallhaven + 14 Generatoren + Telegram-Kanaele vor 2022 + "Ai Only". Die Seite
ersetzt bericht.html (v1); was v1 gelernt hat, steht unten im Abschnitt
"Der Weg dorthin".

Quellen: ergebnis_v2_*.json (train2.py), vergleich_b.json (vergleich_b.py),
kontrolle_v2_best.log (kontrolle.py), pruefe_test.log (pruefe.py), Pack v2.

    python lab/detektor/bericht2.py
"""
import html
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Kernmodule
from bericht import CSS, JS, OUT, grafik, komma, kontrolle, lies, log, prozent   # noqa: E402
from lesen import Pack                                                         # noqa: E402

TAG = "quelle_clip_mid_dino_mid"
NAMEN = {"clip_mid": "CLIP-Mitte", "dino_mid": "DINO-Mitte", "clip_l14": "CLIP-Ende",
         "dino": "DINO-Ende"}
MESSDATUM = "25. September 2026"


def rueckgrate(tag):
    """'quelle_clip_mid_dino_mid' -> ['clip_mid', 'dino_mid']; laengste Namen zuerst."""
    rest, out = tag.split("_", 1)[1], []
    while rest:
        k = next(k for k in sorted(NAMEN, key=len, reverse=True) if rest.startswith(k))
        out.append(k)
        rest = rest[len(k):].lstrip("_")
    return out


def kopfname(tag):
    return (" + ".join(NAMEN[r] for r in rueckgrate(tag))
            + (" (Klassengewicht)" if tag.startswith("klasse_") else ""))


def balken(zeilen, schluessel_lab):
    """Waagerechte Balken, ein Wert je Zeile, Farbe nach Herkunft."""
    lx, rechts, zh, bh = 190, 54, 24, 12
    breite = 640
    pb = breite - lx - rechts
    oben = 16
    hoehe = oben + zh * len(zeilen) + 26
    t = [f'<svg viewBox="0 0 {breite} {hoehe}" width="100%" role="img" '
         f'aria-label="{html.escape(schluessel_lab)}">']
    for v in (0, .25, .5, .75, 1):
        x = lx + v * pb
        t.append(f'<line class="raster" x1="{x:.1f}" y1="{oben-6}" x2="{x:.1f}" y2="{hoehe-24}"/>')
        t.append(f'<text x="{x:.1f}" y="{hoehe-8}" text-anchor="middle">{int(v*100)} %</text>')
    for k, (name, wert, neu) in enumerate(zeilen):
        y = oben + k * zh
        farbe = "var(--s1)" if neu else "var(--s3)"
        w = max(wert * pb, 1.5)
        t.append(f'<text class="gen" x="{lx-10}" y="{y+bh-1}" text-anchor="end">{html.escape(name)}</text>')
        t.append(f'<rect class="bar" x="{lx}" y="{y}" width="{w:.1f}" height="{bh}" rx="2" fill="{farbe}" '
                 f'data-tipp="{html.escape(name)} · {schluessel_lab}: {prozent(wert)}"/>')
        t.append(f'<text class="wert" x="{lx+w+5:.1f}" y="{y+bh-1.5}">{prozent(wert, 1)}</text>')
    t.append(f'<line class="achse" x1="{lx}" y1="{oben-6}" x2="{lx}" y2="{hoehe-24}"/></svg>')
    leg = ('<div class="legende"><span><i style="background:var(--s1)"></i>neue Köpfe (Labor)</span>'
           '<span><i style="background:var(--s3)"></i>Produktion (imagesort.py)</span></div>')
    return leg + "".join(t)


def pruefung(text):
    m = re.search(r"(\d+) bewertet in ([\d.]+) min \(([\d.]+)/s\), (\d+) Fotos, (\d+) zu klein, (\d+) Fehler", text)
    n = re.search(r"KI bei 1-%-Schwelle: (\d+) \(([\d.]+)%\), zusaetzlich verdaechtig bei 5 %: (\d+)", text)
    o = re.search(r"(\d+) Bilder in (.+)", text)
    if not (m and n and o):
        return None
    return {"bewertet": int(m.group(1)), "min": float(m.group(2)), "rate": float(m.group(3)),
            "fotos": int(m.group(4)), "klein": int(m.group(5)), "fehler": int(m.group(6)), "ki": int(n.group(1)),
            "ki_anteil": float(n.group(2)) / 100, "verd": int(n.group(3)),
            "n": int(o.group(1)), "ordner": o.group(2).strip()}


def main():
    erg = lies(f"ergebnis_v2_{TAG}.json")
    ohne = lies(f"ergebnis_v2_{TAG}_ohne.json")
    vb = lies("vergleich_b.json")
    kt, kr = kontrolle(log("kontrolle_v2_best.log"))
    pr = pruefung(log("pruefe_test.log"))

    pack = Pack("v2")
    tab_daten = {}
    for q in ("wallhaven", "generiert", "telegram", "ai_only"):
        for s in ("train", "val", "test"):
            tab_daten[(q, s)] = int(((pack.quelle == q) & (pack.split == s) & pack.ok).sum())
    kanaele = {s: len(set(pack.grp[(pack.quelle == "telegram") & (pack.split == s)]))
               for s in ("train", "val", "test")}

    neu = {k[5:]: v for k, v in vb.items() if k.startswith("neu: ")}
    prod = {k[6:]: v for k, v in vb.items() if k.startswith("Prod: ")}
    best = neu[TAG]["gesamt"]
    pbest_name, pbest = max(prod.items(), key=lambda x: x[1]["gesamt"][0])
    pbest1_name, pbest1 = max(prod.items(), key=lambda x: x[1]["gesamt"][1])
    b = erg["B"]

    h = []
    a = h.append
    a('<title>Der Wallhaven-Detektor</title>')
    a('<link rel="preconnect" href="https://fonts.googleapis.com">'
      '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
      '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700'
      '&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">')
    a(f"<style>{CSS}</style>")
    a('<div id="tipp" aria-hidden="true"></div><main class="seite">')
    a(f'<div class="kicker">Laborbericht v2 · {MESSDATUM} · Produktion unangetastet</div>')
    a('<h1>Der Wallhaven-Detektor, zweite Fassung</h1>')
    a('<p class="unter">Ein KI-Bild-Detektor für Anime-Kunst, der allein stehen soll. '
      'Gelernt aus Wallhaven-Bildern vor 2022, Nachbildungen derselben Motive aus vierzehn '
      'Generatoren, menschlichen Telegram-Posts vor 2022 und einem KI-Kanal. Geprüft auf '
      'Telegram-Kanälen, die er nie gesehen hat.</p>')

    # --- Antwort
    a('<div class="antwort">')
    a(f'<div class="kachel neu"><div class="wert">{prozent(best[1])}</div><div class="was">'
      f'der KI-Bilder erkannt bei 1&nbsp;% Fehlalarm, auf nie gesehenen Telegram-Kanälen</div></div>')
    a(f'<div class="kachel"><div class="wert">{prozent(pbest1["gesamt"][1])}</div><div class="was">'
      f'bester Produktionsdetektor dort ({html.escape(pbest1_name)})</div></div>')
    a(f'<div class="kachel"><div class="wert">{komma(best[0])}</div><div class="was">'
      f'AUC des neuen Kopfs, gegen {komma(pbest["gesamt"][0])} für {html.escape(pbest_name)}</div></div>')
    a('</div>')
    a('<h2 style="border-top:0;margin-top:1.5rem;padding-top:0">Die Antwort in drei Sätzen</h2>')
    a('<ul>'
      f'<li>Auf vier Telegram-Kanälen, die in keinem Schritt vorkamen, gegen einen ungesehenen '
      f'KI-Kanal findet der neue Kopf <b>{prozent(best[1])}</b> der KI-Bilder bei 1&nbsp;% '
      f'Fehlalarm, der beste der sieben Produktionsdetektoren {prozent(pbest1["gesamt"][1])}. '
      f'Er schlägt damit jeden einzelnen, allein und ohne Rangmittel.</li>'
      f'<li>Der größte Sprung kam aus <b>anderen Schichten</b>: die mittleren Schichten von '
      f'CLIP und DINOv2 tragen die Herkunftsspuren, die letzten vor allem, was auf dem Bild '
      f'zu sehen ist. Mit denselben Daten hob das die AUC von {komma(neu["quelle_clip_l14"]["gesamt"][0])} '
      f'(CLIP-Ende) auf {komma(best[0])}.</li>'
      f'<li>Alle Kontrollen auf Verarbeitungsspuren bestanden. Schwach bleibt er bei Generatoren, '
      f'die er nie gesehen hat und die anders aussehen als der Rest: hdm_xut '
      f'{prozent(ohne["hdm_xut"]["tpr@0.01"], 0)}, animepro_flux '
      f'{prozent(ohne["animepro_flux"]["tpr@0.01"], 0)}.</li></ul>')

    # --- Telegram-Test
    a('<h2>Der Telegram-Test</h2>')
    tsd = lambda n: f"{n:,}".replace(",", ".")
    a(f'<p>{tsd(b["n_mensch"])} menschliche Bilder aus vier Kanälen (nur Posts vor 2022) gegen '
      f'{tsd(b["n_ki"])} KI-Bilder: den kompletten kleineren „Ai Only“-Export plus 300 lose Dateien. '
      'Kein Bild dieser Kanäle war im Training oder in der Validierung. Die Produktion sah die '
      'Rohdateien, der neue Kopf dieselben Bilder nach der Zurichtung auf 512&nbsp;px, '
      'ungestört, so wie sie im Einsatz ankämen.</p>')
    zeilen = [(("neu: " if k in neu else "") + (kopfname(k) if k in neu else k),
               (neu.get(k) or prod.get(k))["gesamt"][1], k in neu)
              for k in sorted(list(neu) + list(prod),
                              key=lambda k: -(neu.get(k) or prod.get(k))["gesamt"][1])]
    zeilen = [(n.replace("neu: ", ""), w, nn) for n, w, nn in zeilen]
    a('<figure>' + balken(zeilen, "Treffer bei 1 % Fehlalarm") +
      '<figcaption>Trefferquote bei 1&nbsp;% Fehlalarm auf dem Telegram-Test, alle Köpfe und '
      'alle Produktionsdetektoren auf denselben 23.007 Bildern. Die Schwelle sitzt für jeden '
      'Detektor auf den menschlichen Testbildern selbst, das ist für alle gleich.</figcaption></figure>')
    a('<div class="tabelle"><table><thead><tr><th>Detektor</th><th class="n">AUC</th>'
      '<th class="n">@1 %</th><th class="n">@5 %</th><th class="n">KI-Kanal @1 %</th>'
      '<th class="n">300 lose AUC</th></tr></thead><tbody>')
    for k, v in sorted(list(neu.items()) + list(prod.items()), key=lambda x: -x[1]["gesamt"][0]):
        name = kopfname(k) if k in neu else k
        stil = ' style="font-weight:600"' if k == TAG else ""
        a(f'<tr{stil}><td>{"" if k in neu else "Prod: "}{html.escape(name)}</td>'
          f'<td class="n">{komma(v["gesamt"][0])}</td><td class="n">{prozent(v["gesamt"][1])}</td>'
          f'<td class="n">{prozent(v["gesamt"][2])}</td><td class="n">{prozent(v["kanal"][1])}</td>'
          f'<td class="n">{komma(v["lose"][0])}</td></tr>')
    a('</tbody></table></div>')
    a('<p>Die 300 losen Dateien erkennt fast jeder, auch die Produktion. Schwer ist der Kanal: '
      'er postet kleine Bilder (lange Kante im Median gut 520&nbsp;px), die Upload-Kompression '
      'hat dort viel von dem gelöscht, woran ein Detektor sich hält. Genau dort liegt der '
      'Abstand zur Produktion.</p>')

    # --- Zwischenschichten
    a('<h2>Was den Sprung brachte: mittlere Schichten</h2>')
    a('<p>Alle Köpfe unten sind gleich trainiert (MLP, gestörte Bilder, jede Quelle gleich '
      'gewichtet); sie unterscheiden sich nur in den eingefrorenen Merkmalen. „Ende“ ist die '
      'letzte Schicht eines Modells, „Mitte“ das Klassen-Token aus fünf Schichten über die '
      'Tiefe verteilt (CLIP ViT-L/14: Blöcke 7, 11, 15, 19, 23; DINOv2-L: 8, 12, 16, 20, 24).</p>')
    a('<div class="tabelle"><table><thead><tr><th>Merkmale</th><th class="n">Dimensionen</th>'
      '<th class="n">AUC</th><th class="n">@1 %</th></tr></thead><tbody>')
    dims = {"clip_l14": 768, "clip_mid": 5120, "dino": 2048, "dino_mid": 5120}
    for k, v in sorted(neu.items(), key=lambda x: -x[1]["gesamt"][0]):
        d = sum(dims[r] for r in rueckgrate(k))
        stil = ' style="font-weight:600"' if k == TAG else ""
        a(f'<tr{stil}><td>{html.escape(kopfname(k))}</td><td class="n">{d:,}</td>'
          f'<td class="n">{komma(v["gesamt"][0])}</td><td class="n">{prozent(v["gesamt"][1])}</td>'
          f'</tr>'.replace(f"{d:,}", f"{d:,}".replace(",", ".")))
    a('</tbody></table></div>')
    a('<p>Die Tagger-Merkmale von WD-EVA02, in v1 noch Teil des besten Kopfs, fehlen in v2: '
      'ihre Extraktion wurde zugunsten der mittleren Schichten abgebrochen. Als Tagger ist '
      'WD auf Inhalt trainiert, also eher eine Endschicht; gemessen ist das in v2 nicht. '
      'Welche Variante am besten ist, wurde auf diesem '
      'Test entschieden; die Zahl des Siegers ist deshalb etwas zu gut. Die Reihenfolge ist '
      f'aber deutlich: Mitte schlägt Ende bei beiden Modellen (CLIP {komma(neu["quelle_clip_mid"]["gesamt"][0])} '
      f'gegen {komma(neu["quelle_clip_l14"]["gesamt"][0])}, DINO {komma(neu["quelle_dino_mid"]["gesamt"][0])} gegen '
      f'{komma(neu["quelle_dino"]["gesamt"][0])}), und die beiden Mitten ergänzen sich.</p>')

    # --- Generatoren
    a('<h2>Die vierzehn eigenen Generatoren</h2>')
    a(f'<p>Wallhaven gegen die Generatoren, alle Testbilder gestört (verkleinert, neu '
      f'komprimiert). Zusammen {prozent(erg["A"]["gesamt"]["tpr@0.01"])} Treffer bei 1&nbsp;% '
      f'Fehlalarm, AUC {komma(erg["A"]["gesamt"]["auc"])}. Der zweite Balken zeigt die härtere '
      'Frage: wie gut erkennt ein Kopf einen Generator, den er im Training nie gesehen hat?</p>')
    gz = [{"name": "alle zusammen", "neu": erg["A"]["gesamt"]["tpr@0.01"], "summe": True}]
    for g in sorted(ohne, key=lambda g: -ohne[g]["tpr@0.01"]):
        gz.append({"name": g, "neu": erg["A"][g]["tpr@0.01"], "ohne": ohne[g]["tpr@0.01"]})
    svg = grafik(gz)
    # die dritte Serie (Produktion) gibt es in v2 nicht; ihren Legendeneintrag entfernen
    svg = re.sub(r'<span><i style="background:var\(--s3\)"></i>[^<]*</span>', "", svg)
    a('<figure>' + svg + '<figcaption>Trefferquote bei 1&nbsp;% Fehlalarm je Generator. Blau: '
      'Kopf mit allen Generatoren trainiert. Orange: je ein Kopf, dem dieser eine Generator '
      'fehlte.</figcaption></figure>')
    a(f'<p>Ungesehene Generatoren, die einem gesehenen ähneln, bleiben über 85&nbsp;%. Einbrüche '
      f'gibt es bei den Außenseitern: hdm_xut ({prozent(ohne["hdm_xut"]["tpr@0.01"])}) hat eine '
      f'eigene Architektur, sd15 ({prozent(ohne["sd15"]["tpr@0.01"])}) ist die älteste und '
      f'kleinste, animepro_flux ({prozent(ohne["animepro_flux"]["tpr@0.01"])}) ist das einzige '
      'Flux-Modell. Ein neuer Generator mit wirklich neuer Bauart wird also zunächst schlechter '
      'erkannt; nachtrainieren mit ein paar tausend Bildern von ihm genügt, der Kopf ist in '
      'Minuten neu gelernt.</p>')

    # --- Schwelle
    a('<h2>Schwelle und Fehlalarm im Einsatz</h2>')
    a(f'<p>Für den Einsatz muss die Schwelle vorher feststehen. Sie wurde auf den drei '
      f'Telegram-Validierungskanälen so gesetzt, dass dort 1&nbsp;% der menschlichen Bilder '
      f'darüber liegen. Auf den vier Testkanälen ergibt sie <b>{prozent(b["fa_bei_val_1"], 2)} '
      f'Fehlalarm</b> bei {prozent(b["tr_bei_val_1"])} Treffern: vorsichtiger als geplant, weil '
      'drei Kanäle für eine Schwelle wenig sind. Wer mehr finden will, nimmt die zweite '
      'gespeicherte Schwelle (5&nbsp;% auf val).</p>')
    a('<div class="tabelle"><table><thead><tr><th>Testkanal (menschlich)</th><th class="n">Bilder</th>'
      '<th class="n">Fehlalarm</th></tr></thead><tbody>')
    for k, v in erg["C"].items():
        a(f'<tr><td>{html.escape(k.split("|", 1)[1])}</td><td class="n">{v["n"]:,}</td>'
          f'<td class="n">{prozent(v["fa_bei_val_1"], 2)}</td></tr>'.replace(f'{v["n"]:,}',
                                                                              f'{v["n"]:,}'.replace(",", ".")))
    a('</tbody></table></div>')
    a('<p>Die vier Kanäle liegen eng beieinander; die Schwelle hängt nicht an einem einzelnen.</p>')

    # --- Kontrollen
    a('<h2>Kontrollen auf Verarbeitungsspuren</h2>')
    a('<p>In v1 hatte ein Kopf gelernt, wie ein Bild verkleinert wurde, statt wer es gemacht hat. '
      'Zwei Prüfungen schließen das für den neuen Kopf aus.</p>')
    if kt:
        a('<h3>Eingriff: nur den Maßstab ändern</h3>')
        a('<p>Menschliche Testbilder werden um genau den Faktor vergrößert, der sie der Größe der '
          'KI-Bilder angleicht, KI-Bilder entsprechend verkleinert. Hinge der Kopf am Maßstab, '
          'wanderten die menschlichen nach oben und die KI-Bilder nach unten.</p>')
        a('<div class="tabelle"><table><thead><tr><th></th><th class="n">Logit vorher</th>'
          '<th class="n">nachher</th><th class="n">Verschiebung (Median)</th></tr></thead><tbody>')
        for lab, nm in (("Negative", "menschlich"), ("Positive", "KI")):
            v = kt.get(lab)
            if v:
                a(f'<tr><td>{nm}</td><td class="n">{v[0]:+.2f}</td><td class="n">{v[1]:+.2f}</td>'
                  f'<td class="n">{v[2]:+.3f}</td></tr>'.replace(".", ","))
        a('</tbody></table></div>')
        a(f'<p>Die Verschiebungen sind winzig gegen den Abstand der Klassen (rund '
          f'{kt["Positive"][0] - kt["Negative"][0]:.0f} Logit-Punkte), '
          'und die KI-Bilder wandern sogar leicht nach oben statt nach unten. Kein Maßstab-Leck.</p>')
    if kr:
        a('<h3>Robustheit: beide Klassen gleich stören</h3>')
        a('<div class="tabelle"><table><thead><tr><th>Störung (beide Klassen)</th>'
          '<th class="n">AUC</th><th class="n">Treffer @1 %</th></tr></thead><tbody>')
        for lab, au, tr in kr:
            a(f'<tr><td>{html.escape(lab)}</td><td class="n">{komma(au)}</td>'
              f'<td class="n">{prozent(tr)}</td></tr>')
        a('</tbody></table></div>')
        a(f'<p>Je 1.200 Testbilder beider Klassen aus allen Quellen, also auch Telegram. Die AUC '
          f'fällt von {komma(kr[0][1])} auf {komma(kr[-1][1])}, wenn beide Seiten auf 75&nbsp;% '
          'verkleinert und mit q85 neu komprimiert werden. Der v1-Kopf, der Verarbeitungsspuren '
          'gelernt hatte, fiel dabei auf 0,643.</p>')

    # --- Probelauf
    a('<h2>Das Werkzeug: pruefe.py</h2>')
    a('<p>Ein Ordner-Scanner, der <b>nur liest</b>: nichts wird verschoben, umbenannt oder '
      'gelöscht. Er filtert echte Fotos heraus wie <code>imagesort.py</code>, bewertet den Rest '
      'mit dem neuen Kopf und schreibt eine CSV (alle Bilder, nach Score sortiert) und eine '
      'HTML-Seite mit den verdächtigsten Bildern als Vorschau.</p>')
    a('<pre>.venv\\Scripts\\python.exe lab\\detektor\\pruefe.py "<Ordner>\\'
      'ChatExport_2026-09-22" --max 2000</pre>')
    if pr:
        a(f'<p>Probelauf auf den ersten {pr["n"]:,} Bildern von '.replace(",", ".") +
          f'<code>{html.escape(os.path.basename(pr["ordner"]))}</code>: {pr["bewertet"]:,} bewertet, '
          f'{pr["fotos"]} Fotos und {pr["klein"]} Bilder unter 512&nbsp;px übersprungen, {pr["fehler"]} Fehler, {komma(pr["rate"], 1)} Bilder '
          f'pro Sekunde. <b>{pr["ki"]}</b> Bilder ({prozent(pr["ki_anteil"])}) liegen über der '
          f'1-%-Schwelle, {pr["verd"]} weitere über der 5-%-Schwelle. Ergebnis in '
          '<code>lab\\detektor\\pruefung\\</code>.</p>'.replace(f'{pr["bewertet"]:,}',
                                                              f'{pr["bewertet"]:,}'.replace(",", ".")))

        a('<p>Von Auge durchgesehen: unter den sechs höchsten Scores sind zwei erkennbar alte '
          'offizielle Illustrationen (eine Visual-Novel-Grafik mit Wasserzeichen, ein Magazinscan), '
          'wahrscheinlich Fehlalarme oder hochgerechnete Reposts. Für diesen Ordner gibt es keine '
          'Wahrheit; die Quote ist ein Probelauf des Werkzeugs, keine Messung. Beim ersten Lauf '
          'bewertete der Scanner noch Telegrams Vorschaubilder (<code>*_thumb.jpg</code>) mit, '
          'die im Training nie vorkamen; 24 der damals 31 Treffer waren solche. Sie werden jetzt '
          'übersprungen, ebenso alles unter 512&nbsp;px.</p>')

    # --- Daten
    a('<h2>Datensatz v2</h2>')
    a('<div class="tabelle"><table><thead><tr><th>Quelle</th><th>Klasse</th><th class="n">train</th>'
      '<th class="n">val</th><th class="n">test</th></tr></thead><tbody>')
    for q, lab, kl in (("wallhaven", "Wallhaven vor 2022", "Mensch"),
                       ("telegram", f"Telegram vor 2022, {sum(kanaele.values())} Kanäle", "Mensch"),
                       ("generiert", "14 Generatoren, Wallhaven-Motive", "KI"),
                       ("ai_only", "„Ai Only“-Exporte", "KI")):
        z = [f'{tab_daten[(q, s)]:,}'.replace(",", ".") for s in ("train", "val", "test")]
        a(f'<tr><td>{lab}</td><td>{kl}</td>' + "".join(f'<td class="n">{x}</td>' for x in z) + '</tr>')
    ges = [f'{sum(tab_daten[(q, s)] for q in ("wallhaven", "generiert", "telegram", "ai_only")):,}'
           .replace(",", ".") for s in ("train", "val", "test")]
    a('<tr class="summe"><td>zusammen</td><td></td>' + "".join(f'<td class="n">{x}</td>' for x in ges)
      + '</tr></tbody></table></div>')
    a(f'<ul><li><b>Getrennt nach Kanal, nicht nach Bild.</b> Telegram: {kanaele["train"]} Kanäle '
      f'train, {kanaele["val"]} val, {kanaele["test"]} test, höchstens 5.000 Bilder je Kanal. '
      'Mehrere Exporte desselben Kanals wurden über gleiche Dateinamen zusammengefasst '
      '(88 Exporte, 77 Kanäle), damit kein Kanal auf beiden Seiten landet.</li>'
      '<li><b>„Ai Only“ sind zwei Telegram-Exporte</b>, keine Downloads (in v1 falsch '
      'beschrieben). Gewünscht war halb/halb; weil ein Kanal nicht geteilt werden darf, wurde '
      'es 74/26: der größere Export zum Lernen, der kleinere plus die 300 losen Dateien zum '
      'Testen.</li>'
      '<li><b>Upload-Simulation.</b> Wallhaven- und Generatorbilder wurden vor allem anderen so '
      'verarbeitet wie ein Telegram-Upload (lange Kante höchstens 1280, JPEG q87 4:2:0). Sonst '
      'hieße „sieht nach Telegram aus“ für den Kopf „menschlich“.</li>'
      '<li><b>Nicht verwendet:</b> <code>_ai_quarantine</code>, <code>review_log</code> und '
      'alles, was die Produktion selbst sortiert hat.</li></ul>')

    # --- Grenzen
    a('<h2>Grenzen</h2>')
    a('<ul><li>Die KI-Seite des Telegram-Tests ist ein einziger Kanal. Wie der Kopf auf anderen '
      'KI-Kanälen abschneidet, ist geschätzt, nicht gemessen.</li>'
      '<li>Die Variante wurde auf dem Test gewählt; der Abstand zur Produktion ist so groß, dass '
      'das am Urteil nichts ändert, die genaue Zahl ist aber etwas zu hoch.</li>'
      '<li>Bilder unter 512&nbsp;px kamen im Training nicht vor; der Scanner lässt sie aus. '
      'Kanäle, die nur so kleine Bilder posten, kann er deshalb nicht prüfen.</li>'
      '<li>Menschliche Kunst nach 2022 ist nicht belegbar menschlich und fehlt deshalb ganz. '
      'Neue Zeichenstile können falsch anschlagen.</li></ul>')

    # --- Weg
    a('<h2>Der Weg dorthin</h2>')
    a('<details><summary>Was v1 gelehrt hat</summary><ul>'
      '<li>Rohbildgröße trennte die Klassen allein mit AUC 1,000; deshalb wird alles auf '
      '512&nbsp;px zugerichtet.</li>'
      '<li>Nach der Zurichtung verriet das Seitenverhältnis die Generator-Buckets (AUC 0,997); '
      'deshalb schneiden die Rückgrate quadratisch aus der Mitte.</li>'
      '<li>Ein Kopf auf ungestörten Bildern lernte das Verkleinerungsmuster (menschlich exakt '
      'halbiert): AUC 0,987, bei gleicher Störung beider Seiten 0,643. Seitdem wird nur auf '
      'gestörten Bildern trainiert.</li>'
      '<li>v1 kannte als „menschlich“ nur Wallhaven-Wallpaper und war auf Telegram schwächer als '
      'die Produktion. Die menschlichen Telegram-Kanäle vor 2022 und die mittleren Schichten '
      'haben das umgedreht.</li></ul></details>')
    a('</main>')
    a(f"<script>{JS}</script>")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(h))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
