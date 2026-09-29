r"""bericht3.py — Laborbericht v3, Zahlen direkt aus den Ergebnisdateien.

Quellen: ergebnis_v3.json (train3.py), kanaele_v3*.json (kanaele_test.py),
v3k_danbooru.log (danbooru_test.py v3 v3r), mj_versionen.json,
v3k_kontrolle.log (kontrolle3.py), <DATASETS>\QUELLEN.md fuer die Datenlage.

    python lab/detektor/bericht3.py
"""
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bericht import CSS, JS, OUT, komma, lies, log, prozent   # noqa: E402
from kanaele import ANIME, NAME                  # noqa: E402
from lesen import DATEN                                        # noqa: E402

KANAL_NAME = NAME        # echte Kanalnamen aus kanaele_lokal.py (nur lokal)
NEU = "p3"


def zusammen(name, alt):
    """Aktuelle Ergebnisdatei plus Koepfe aus einem frueheren Lauf (gleiche Methode, je Kopf eigene
    Schwelle -> Laeufe sind kombinierbar)."""
    d = lies(name)
    if os.path.exists(os.path.join(DATEN, alt)):
        a = lies(alt)
        if "ergebnis" in d:
            for t, v in a["ergebnis"].items():
                d["ergebnis"].setdefault(t, v)
        else:
            for t, v in a.items():
                d.setdefault(t, v)
    return d


def phase2(a):
    """Abschnitt universeller Detektor: p3/p2 gegen v3/v3s, Zahlen aus gleich_fa*.json und p2_test*.json."""
    gf = zusammen("gleich_fa.json", "gleich_fa_p2stand.json")
    pt = zusammen("p2_test.json", "p2_test_p2stand.json")
    koepfe = [t for t in ("v3", "v3s", "p2", NEU) if t in gf["ergebnis"] and t in pt]
    n3 = pt[NEU]
    kan = [k for k in n3 if k.startswith("K ") and k[2:] in KANAL_NAME]
    mittel = sum(n3[k][1] for k in kan) / len(kan)
    wh, dn, da = (n3[g][1] for g in ("M Wallhaven general <2022", "M Danbooru Nicht-Anime (alle)",
                                      "M Danbooru 2026 (Anime)"))
    a('<div class="antwort">')
    a(f'<div class="kachel neu"><div class="wert">{prozent(mittel)}</div><div class="was">der KI-Bilder erkannt, '
      f'Mittel über {len(kan)} Telegram-KI-Kanäle (Scanner-Kopf {NEU}, eigene Schwelle)</div></div>')
    a(f'<div class="kachel neu"><div class="wert">{prozent(da)}</div><div class="was">Fehlalarm auf menschlicher '
      f'Anime-Kunst von 2026 (Danbooru, Künstler nie im Training)</div></div>')
    a(f'<div class="kachel neu"><div class="wert">{prozent(wh)} · {prozent(dn)}</div><div class="was">Fehlalarm auf '
      f'menschlicher Nicht-Anime-Kunst: Wallhaven vor 2022 · Danbooru-Nicht-Anime 2022–26 '
      f'(v3: {prozent(pt["v3"]["M Wallhaven general <2022"][1])} · '
      f'{prozent(pt["v3"]["M Danbooru Nicht-Anime (alle)"][1])})</div></div>')
    a('</div>')

    a('<h2 style="border-top:0;margin-top:1.5rem;padding-top:0">Was seit v3 neu ist</h2>')
    a('<ul>'
      '<li><b>p2 — menschliche Nicht-Anime-Kunst im Training.</b> 56.845 Danbooru-Bilder 2022–2026 mit '
      'Stil-Tags wie 3d, realistic, western_comics, card_(medium), character_sheet, landscape, pixel_art '
      '(nach Künstlern geteilt) und 67.969 Wallhaven-Wallpaper „general“ von vor 2022 vom NAS (Foto-Filter). '
      'Dazu 34.117 halbrealistische und 3D-KI-Bilder von CivitAI, die der Anime-Stilfilter von v3 '
      'aussortiert hatte.</li>'
      '<li><b>p3 — frühe KI von 2022.</b> 17.040 Bilder aus DiffusionDB (Stable-Diffusion-Discord, August '
      '2022, Lizenz CC0), höchstens 25 je Nutzer, Split nach Nutzer. Pixel-Abgleich: keines davon steckt in '
      'einem Testkanal.</li>'
      '<li><b>15 Telegram-KI-Kanäle</b> als Test statt 9 — alle nur Test, keiner im Training; Werbung '
      'und Vorschaubilder entfernt.</li>'
      '<li><b>Schutzregel geschärft</b> und rückwirkend auf alle Quellen angewandt (siehe Datenlage). '
      '<b>v3s</b> ist v3 neu trainiert ohne die nachträglich gesperrten Bilder — als sauberer Vergleich.</li>'
      '<li><b>Schwelle neu kalibriert.</b> Auf ungestörten val-Bildern der drei menschlichen Gruppen '
      '(Danbooru-Anime 2026, Wallhaven, Danbooru-Nicht-Anime), jede gleich gewichtet, im Mittel 1&nbsp;%.</li>'
      '</ul>')

    a('<h2>Bei gleichem Fehlalarm</h2>')
    a(f'<p>Jeder Kopf ist auf etwas anderes kalibriert; ein Vergleich bei der jeweils eigenen Schwelle misst '
      f'deshalb auch die Kalibrierung. Hier bekommt jeder Kopf die Schwelle, bei der er auf den drei menschlichen '
      f'Testgruppen im Mittel {komma(gf["fa"], 0)}&nbsp;% Fehlalarm hat. Die Schwelle ist dafür auf dem Test '
      f'gesetzt; sie dient nur dem Vergleich. Die AUC braucht gar keine Schwelle. Fett: bester Kopf je Kanal.</p>')
    e = gf["ergebnis"]
    kopf = "".join(f'<th class="n">{t}</th>' for t in koepfe)
    a(f'<div class="tabelle"><table><thead><tr><th>KI-Kanal</th>{kopf}<th class="n">AUC v3</th>'
      f'<th class="n">AUC {NEU}</th></tr></thead><tbody>')
    reihen = sorted(e[NEU]["treffer"], key=lambda k: (k not in ANIME, -e[NEU]["treffer"][k]))
    for k in reihen:
        best = max(e[t]["treffer"][k] for t in koepfe)
        zellen = "".join(f'<td class="n">{"<b>" if e[t]["treffer"][k] == best else ""}{prozent(e[t]["treffer"][k])}'
                         f'{"</b>" if e[t]["treffer"][k] == best else ""}</td>' for t in koepfe)
        a(f'<tr><td>{html.escape(KANAL_NAME.get(k, k))}</td>{zellen}'
          f'<td class="n">{komma(e["v3"]["auc"][k], 4)}</td><td class="n">{komma(e[NEU]["auc"][k], 4)}</td></tr>')
    mt = {t: sum(e[t]["treffer"].values()) / len(e[t]["treffer"]) for t in koepfe}
    a('<tr class="summe"><td>Mittel der Kanäle</td>' + "".join(f'<td class="n">{prozent(mt[t])}</td>' for t in koepfe)
      + '<td></td><td></td></tr>')
    for g in e[NEU]["fa"]:
        a(f'<tr class="summe"><td>Fehlalarm {html.escape(g)}</td>'
          + "".join(f'<td class="n">{prozent(e[t]["fa"][g])}</td>' for t in koepfe) + '<td></td><td></td></tr>')
    a('</tbody></table></div>')
    a('<p><b>p2 gegen v3:</b> auf allen 15 Kanälen vorn, bei den Anime-Kanälen am deutlichsten (Anime-Kanal A und '
      'Anime-Kanal B fast 20 Punkte). Die breitere menschliche Seite hat den Kopf nicht stumpfer gemacht, sondern '
      'genauer: er muss KI jetzt an etwas anderem erkennen als am Zeichenstil. v3s (ohne die gesperrten Bilder) '
      'liegt gleichauf mit v3 — das Sperren hat nichts gekostet.</p>')
    a('<p><b>p3 gegen p2:</b> kleiner Schritt im Mittel, gezielt dort, wo er hin sollte — Mischkanal B, '
      'Midjourney-Kanal A, Mischkanal G und Anime-Kanal A legen zu; Mischkanal E, Midjourney-Kanal B und der Leonardo-Kanal '
      'geben 1–2,5 Punkte ab. Unterschiede dieser Größe liegen nahe an der Streuung zwischen Trainingsläufen.</p>')

    a('<h2>Fehlalarm und Treffer bei der eigenen Schwelle</h2>')
    a('<p>So, wie jeder Kopf im Scanner laufen würde. M = menschlich (Fehlalarm, soll um 1&nbsp;% liegen), '
      'K = KI. Die Stilzeilen sind Teilmengen von Danbooru-Nicht-Anime (kleine Gruppen, ±1 Punkt ist Rauschen).</p>')
    a(f'<div class="tabelle"><table><thead><tr><th>Gruppe</th><th class="n">Bilder</th>{kopf}</tr></thead><tbody>')
    for g in n3:
        if not g.startswith("M"):
            continue
        a(f'<tr><td>{html.escape(g[2:].replace("   ", "· "))}</td><td class="n">{n3[g][0]:,}</td>'.replace(",", ".")
          + "".join(f'<td class="n">{prozent(pt[t][g][1])}</td>' for t in koepfe) + '</tr>')
    a('<tr class="summe"><td>K CivitAI halbrealistisch/3D</td>'
      f'<td class="n">{n3["K CivitAI halbrealistisch/3D"][0]:,}</td>'.replace(",", ".")
      + "".join(f'<td class="n">{prozent(pt[t]["K CivitAI halbrealistisch/3D"][1])}</td>' for t in koepfe) + '</tr>')
    a('</tbody></table></div>')
    a('<p>Bei der eigenen Schwelle tauschen p2 und p3 Treffer gegen Ruhe: auf den Anime-Kanälen liegen sie '
      'gleichauf mit v3, auf Kanälen mit viel Nicht-Anime-KI erkennen sie weniger — aber v3 kauft dort seine '
      'Treffer mit rund 11&nbsp;% Fehlalarm auf menschlichen Wallpapern und 7–8&nbsp;% auf 3D-Kunst.</p>')
    a('<div class="hinweis"><b>Frühe KI von 2022.</b> Gut die Hälfte von Mischkanal B stammt aus der zweiten '
      'Jahreshälfte 2022 — frühe Stable-Diffusion- und Midjourney-Bilder: Landschaften, Sci-Fi, Konzeptkunst im '
      'Renderlook, dieselben Motive wie bei den menschlichen Wallhaven-Wallpapern. p2 erkannte davon 67&nbsp;%, '
      'p3 mit DiffusionDB 76&nbsp;% (die Bilder von 2023: beide rund 90&nbsp;%). v3 kommt auf 83&nbsp;%, aber '
      'nur mit zehnfachem Fehlalarm auf Wallpapern.</div>')
    a(f'<p><b>Entscheidung:</b> Der Scanner <code>pruefe.py</code> nutzt ab jetzt <b>{NEU}</b> (v3 → p2 → p3). '
      'Kriterium: Anime höchstens 2 Punkte hinter v3, Fehlalarm auf Nicht-Anime höchstens 1,5&nbsp;%, '
      f'Kontrollen bestanden. {NEU}: Anime erfüllt (Anime-Kanäle A–D gleichauf '
      'oder vorn, Anime-Kanal E 1,8 Punkte dahinter), Wallhaven 1,5&nbsp;% und Danbooru-Nicht-Anime 0,5&nbsp;% erfüllt, '
      'Kontrollen bestanden (weiter unten). Die erste Kalibrierung von p2 hatte gestörte val-Bilder aller Quellen '
      'in einen Topf geworfen und ergab auf dem Test 2,0&nbsp;% (Anime) und 3,4&nbsp;% (Wallhaven); seitdem '
      'setzt <code>kalibriere.py</code> die Schwelle.</p>')


def danbooru(text):
    """{tag: {"schwellen": (a, b), "fa": {gruppe: (n, a, b)}, "treffer": {kanal: (n, a, b)}}}"""
    out = {}
    for block in re.split(r"\n(?=v3\w*: )", text):
        m = re.match(r"(v3\w*): ", block.strip())
        if not m:
            continue
        tag = m.group(1)
        s = re.search(r"Schwelle A .*?: ([+-][\d.]+)\s+Schwelle B .*?: ([+-][\d.]+)", block)
        d = {"schwellen": (float(s.group(1)), float(s.group(2))), "fa": {}, "treffer": {}}
        teil = "fa"
        for z in block.splitlines():
            if "KI-Kanal" in z:
                teil = "treffer"
            m2 = re.match(r"\s+(\S[\w\- ]*?)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%", z)
            if m2:
                d[teil][m2.group(1).strip()] = (int(m2.group(2)), float(m2.group(3)) / 100,
                                                float(m2.group(4)) / 100)
        out[tag] = d
    return out


def kontrolle(text):
    e = {}
    for lab in ("Menschen", "KI"):
        m = re.search(lab + r"\s+Logit vorher ([+-][\d.]+)\s+nachher ([+-][\d.]+)\s+Verschiebung Median ([+-][\d.]+)", text)
        if m:
            e[lab] = tuple(float(x) for x in m.groups())
    r = [(m.group(1).strip(), float(m.group(2)), float(m.group(3)) / 100)
         for m in re.finditer(r"  (\S.*?)\s+AUC ([\d.]+)\s+Treffer bei 1 % FA ([\d.]+)%", text)]
    return e, r


def balken_paar(zeilen, breite=640):
    """Je Zeile zwei Balken: vorher (grau) und v3 (blau), 0-100 %."""
    lx, rechts, zh, bh = 200, 50, 30, 9
    pb = breite - lx - rechts
    oben = 14
    hoehe = oben + zh * len(zeilen) + 24
    t = [f'<svg viewBox="0 0 {breite} {hoehe}" width="100%" role="img" aria-label="Treffer je Kanal">']
    for v in (0, .25, .5, .75, 1):
        x = lx + v * pb
        t.append(f'<line class="raster" x1="{x:.1f}" y1="{oben-6}" x2="{x:.1f}" y2="{hoehe-22}"/>')
        t.append(f'<text x="{x:.1f}" y="{hoehe-6}" text-anchor="middle">{int(v*100)} %</text>')
    for k, (name, alt, neu) in enumerate(zeilen):
        y = oben + k * zh
        t.append(f'<text class="gen" x="{lx-10}" y="{y+13}" text-anchor="end">{html.escape(name)}</text>')
        for j, (v, farbe, lab) in enumerate(((alt, "var(--s2)", "v3r"), (neu, "var(--s1)", "v3"))):
            w = max(v * pb, 1.5)
            yy = y + j * (bh + 2)
            t.append(f'<rect class="bar" x="{lx}" y="{yy}" width="{w:.1f}" height="{bh}" rx="2" fill="{farbe}" '
                     f'data-tipp="{html.escape(name)} · {lab}: {prozent(v)}"/>')
            if j == 1:
                t.append(f'<text class="wert" x="{lx+w+5:.1f}" y="{yy+bh-1}">{prozent(v, 0)}</text>')
    t.append(f'<line class="achse" x1="{lx}" y1="{oben-6}" x2="{lx}" y2="{hoehe-22}"/></svg>')
    leg = ('<div class="legende"><span><i style="background:var(--s2)"></i>v3r (gestern, Mensch nur vor 2022)</span>'
           '<span><i style="background:var(--s1)"></i>v3 (mit aktueller menschlicher Kunst)</span></div>')
    return leg + "".join(t)


def main():
    erg = lies("ergebnis_v3.json")
    db = danbooru(log("v3k_danbooru.log"))
    mj = lies("mj_versionen.json")
    kt, kr = kontrolle(log("v3k_kontrolle.log"))
    v3, v3r = db["v3"], db["v3r"]

    h = []
    a = h.append
    a('<title>Der Wallhaven-Detektor</title>')
    a('<link rel="preconnect" href="https://fonts.googleapis.com">'
      '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
      '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700'
      '&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">')
    a(f"<style>{CSS}</style>")
    a('<div id="tipp" aria-hidden="true"></div><main class="seite">')
    a('<div class="kicker">Laborbericht · 27. September 2026 · Produktion unangetastet</div>')
    a('<h1>Der KI-Kunst-Detektor, jetzt für digitale Kunst allgemein</h1>')
    a('<p class="unter">Ein Detektor für KI-generierte Anime- und Digitalkunst (keine Fotos). Gemessen auf '
      '15 echten Telegram-KI-Kanälen, die nie im Training waren, und auf menschlicher Kunst, die ebenfalls '
      'nie im Training war.</p>')
    phase2(a)
    a('<h2>Der Weg zu v3 (Stand 26. September)</h2>')
    a('<p>Alles ab hier ist der Bericht zur Anime-Fassung v3, auf der p2 aufbaut. Die Zahlen gelten für v3 '
      'bei seiner Schwelle (1&nbsp;% Fehlalarm auf Danbooru-Anime 2026).</p>')

    fa_alle = v3["fa"]["alle"]
    tr = v3["treffer"]
    mitte = sorted(x[2] for x in tr.values())
    a('<div class="antwort">')
    a(f'<div class="kachel neu"><div class="wert">{prozent(fa_alle[2])}</div><div class="was">Fehlalarm auf '
      f'{fa_alle[0]:,} menschlichen Bildern von 2026 (nie im Training)</div></div>'.replace(",", "."))
    a(f'<div class="kachel neu"><div class="wert">{prozent(min(mitte), 0)}–{prozent(max(mitte), 1)}</div>'
      f'<div class="was">der KI-Bilder erkannt, je nach Kanal ({len(tr)} echte Telegram-KI-Kanäle)</div></div>')
    a(f'<div class="kachel"><div class="wert">{prozent(v3r["fa"]["alle"][1])} → {prozent(v3["fa"]["alle"][1])}</div>'
      '<div class="was">Fehlalarm auf Kunst von 2026 bei der alten Schwelle: v3r gegen v3</div></div>')
    a('</div>')

    a('<h3>Was sich seit v2 geändert hat</h3>')
    a('<ul>'
      '<li><b>Drei Fehler gefunden und behoben.</b> (1) Die Hälfte des KI-Trainingskanals waren Telegram-'
      'Vorschaubilder, die es auf der menschlichen Seite nicht gab. (2) Der alte KI-Testkanal enthielt auch '
      'menschliche Fan-Wallpaper und taugt nicht als Wahrheit. (3) Alle menschlichen Bilder waren von vor '
      '2022, alle KI-Bilder danach: der Kopf hielt teils <i>neue Zeichenstile</i> für KI.</li>'
      '<li><b>Viel mehr echte KI.</b> 114.611 Bilder aus rund 30 CivitAI-Basismodellen (inkl. OpenAI, '
      'Nano Banana, Seedream, Grok, Imagen4) und 31.662 von AIBooru (inkl. Midjourney/Niji, DALL-E, NovelAI).</li>'
      '<li><b>Aktuelle menschliche Kunst im Training.</b> 69.713 Danbooru-Bilder von 2022 bis März 2026 '
      '(Danbooru verbietet KI); getestet wird mit anderen Künstlern aus April–Juni 2026.</li>'
      '<li><b>Fünf Köpfe gemittelt</b> statt einem: zwischen Seeds schwankte das Ergebnis um bis zu 0,04 AUC.</li>'
      '<li><b>Ehrliche Schwelle.</b> Kalibriert auf 3.638 menschlichen Danbooru-Bildern von 2026 statt auf drei '
      'alten Telegram-Kanälen.</li></ul>')

    a('<h2>Echte KI-Kanäle von Telegram</h2>')
    a('<p>Jeder Kanal war nie im Training. Trefferquote bei der ehrlichen Schwelle (1&nbsp;% Fehlalarm auf '
      'menschlicher Kunst von 2026). Orange: der Vortagskopf ohne aktuelle menschliche Kunst und ohne die '
      'meisten Web-Modelle, bei seiner eigenen ehrlichen Schwelle.</p>')
    zeilen = sorted(((KANAL_NAME.get(k, k), v3r["treffer"].get(k, (0, 0, 0))[2], v[2])
                     for k, v in tr.items()), key=lambda z: -z[2])
    a('<figure>' + balken_paar(zeilen) + '<figcaption>Anime-Kanäle: A, B, C. Alle '
      'anderen zeigen digitale Kunst allgemein — der Detektor ist inzwischen deutlich mehr als ein '
      'Anime-Detektor.</figcaption></figure>')

    pv = lies("prod_vergleich.json")
    a('<h2>Gegen die Produktion</h2>')
    a(f'<p>Dieselben Bilder für alle (je Kanal bis zu 3.000), dieselbe Regel: jeder Detektor bekommt die '
      f'Schwelle, bei der er auf menschlicher Danbooru-Kunst von 2026 (val) 1&nbsp;% Fehlalarm hat. '
      f'Die Produktion sieht dieselben Dateien wie v3.</p>')
    kan = [k for k in pv["n"] if not k.startswith("danbooru")]
    prod = {k: v for k, v in pv["ergebnis"].items() if k != "v3" and max(v[x] for x in kan) > 0}
    a('<div class="tabelle"><table><thead><tr><th>KI-Kanal</th><th class="n">v3</th>'
      '<th class="n">beste Produktion</th><th>welcher</th></tr></thead><tbody>')
    for k in sorted(kan, key=lambda k: -pv["ergebnis"]["v3"][k]):
        bn, bv = max(((d, v[k]) for d, v in prod.items()), key=lambda x: x[1])
        a(f'<tr><td>{html.escape(KANAL_NAME.get(k, k))}</td><td class="n"><b>{prozent(pv["ergebnis"]["v3"][k])}</b></td>'
          f'<td class="n">{prozent(bv)}</td><td>{html.escape(bn.split(":")[1])}</td></tr>')
    a('</tbody></table></div>')
    a('<p>v3 liegt auf acht von neun Kanälen deutlich vorn; nur bei Anime-Kanal C zieht '
      'caformer_plus gleich. Ein Allrounder fehlt der Produktion: je Kanal ist ein anderer Detektor ihr '
      'bester. mobilenetv3_sce ist hier nicht messbar — er gibt vielen Bildern denselben Score, eine '
      '1-%-Schwelle lässt sich nicht setzen (0&nbsp;% in allen Kanälen, deshalb nicht aufgeführt).</p>')

    a('<h2>Fehlalarm auf menschlicher Kunst von 2026</h2>')
    a(f'<p>8.438 Danbooru-Bilder aus April bis Juni 2026, von Künstlern, die in keinem Training vorkommen. '
      f'„Alte Schwelle“ ist die, die auf den Telegram-Menschen vor 2022 1&nbsp;% ergibt — so wurde bis v3r '
      f'kalibriert.</p>')
    a('<div class="tabelle"><table><thead><tr><th>Danbooru 2026</th><th class="n">Bilder</th>'
      '<th class="n">v3r alte Schwelle</th><th class="n">v3 alte Schwelle</th><th class="n">v3 neue Schwelle</th>'
      '</tr></thead><tbody>')
    for g in ("alle", "general", "sensitive", "questionable", "explicit"):
        n, _, _ = v3["fa"][g]
        a(f'<tr{" class=summe" if g == "alle" else ""}><td>{g}</td><td class="n">{n:,}</td>'
          f'<td class="n">{prozent(v3r["fa"][g][1])}</td><td class="n">{prozent(v3["fa"][g][1])}</td>'
          f'<td class="n">{prozent(v3["fa"][g][2])}</td></tr>'.replace(f"{n:,}", f"{n:,}".replace(",", ".")))
    a('</tbody></table></div>')
    a(f'<p>Auf den alten Telegram-Menschen gibt die neue Schwelle {prozent(v3["fa"]["Telegram-Menschen"][2], 2)} '
      'Fehlalarm. Explizite menschliche Kunst schlägt nicht häufiger an als jugendfreie — der Kopf hat '
      '„explizit = KI“ nicht gelernt, obwohl ein Teil der KI-Trainingsbilder sehr explizit ist.</p>')

    a('<h2>Midjourney, nie im Training</h2>')
    a('<div class="tabelle"><table><thead><tr><th>Version (aus dem Prompt)</th><th class="n">Bilder</th>'
      '<th class="n">v3r</th><th class="n">v3</th></tr></thead><tbody>')
    ordnung = ["test", "testp", "v4", "v5", "v6", "v7", "v8"] + sorted(k for k in mj["v3"] if k.startswith("ohne"))
    for g in ordnung:
        if g in mj["v3"]:
            a(f'<tr><td>{html.escape(g)}</td><td class="n">{mj["v3"][g][0]}</td>'
              f'<td class="n">{prozent(mj["v3r"][g][1])}</td><td class="n">{prozent(mj["v3"][g][1])}</td></tr>')
    a('</tbody></table></div>')

    a('<h2>Basismodelle: ungesehene Ersteller</h2>')
    a('<p>CivitAI-Bilder von Erstellern, die nie im Training waren, bei der ehrlichen Schwelle. '
      'Schwächster Wert: Lumina — mit gut 800 Trainingsbildern das dünnste Modell; wird aufgestockt.</p>')
    a('<div class="tabelle"><table><thead><tr><th>Basismodell</th><th class="n">Bilder</th>'
      '<th class="n">erkannt</th></tr></thead><tbody>')
    for k, v in sorted(((k.strip(), v) for k, v in erg.items() if k.startswith("   ")),
                       key=lambda x: x[1]["tr_val_schwelle"]):
        k = "AIBooru (alle Modelle)" if k == "aibooru" else k
        a(f'<tr><td>{html.escape(k)}</td><td class="n">{v["n"]}</td>'
          f'<td class="n">{prozent(v["tr_val_schwelle"])}</td></tr>')
    a('</tbody></table></div>')
    a(f'<p>Die 14 selbst erzeugten Generatoren (gestört): {prozent(erg["A"]["tpr1"])} Treffer bei 1&nbsp;% '
      f'Fehlalarm auf Wallhaven.</p>')

    a('<h2>Kontrollen auf Verarbeitungsspuren</h2>')
    if kt:
        a(f'<p><b>Maßstab-Eingriff:</b> menschliche Bilder vergrößert verschieben sich um '
          f'{komma(kt["Menschen"][2], 2)} Logit-Punkte, KI-Bilder verkleinert um {komma(kt["KI"][2], 2)} — bei '
          f'einem Klassenabstand von rund {kt["KI"][0] - kt["Menschen"][0]:.0f}. Ein Maßstab-Leck müsste die '
          'Menschen nach oben schieben; sie bleiben stehen. Die KI-Seite reagiert leicht auf Verkleinern.</p>')
    pk, pr = kontrolle(log("p3k_kontrolle.log"))
    if pk:
        a(f'<p><b>{NEU}:</b> Menschen verschieben sich um {komma(pk["Menschen"][2], 2)}, KI-Bilder um '
          f'{komma(pk["KI"][2], 2)} Logit-Punkte — ebenfalls kein Maßstab-Leck.</p>')
    if kr:
        pr = dict((lab, (au, t1)) for lab, au, t1 in pr)
        a('<div class="tabelle"><table><thead><tr><th>beide Klassen gleich gestört</th><th class="n">AUC v3</th>'
          '<th class="n">@1 % v3</th><th class="n">AUC ' + NEU + '</th><th class="n">@1 % ' + NEU + '</th></tr></thead><tbody>')
        for lab, au, t1 in kr:
            p_au, p_t1 = pr.get(lab, (None, None))
            a(f'<tr><td>{html.escape(lab)}</td><td class="n">{komma(au)}</td><td class="n">{prozent(t1)}</td>'
              f'<td class="n">{komma(p_au) if p_au else "–"}</td><td class="n">{prozent(p_t1) if p_t1 else "–"}</td></tr>')
        a('</tbody></table></div>')
        a('<p>Menschen hier: Telegram-Menschen vor 2022 und Danbooru 2026 gemischt; ' + NEU + ' wurde auf breiteren '
          'Menschen trainiert, deshalb liegen seine Werte auf dieser Mischung etwas unter v3.</p>')

    a('<h2>Datenlage</h2>')
    a('<p>Alle Quellen, ihre Filter und ihre Verwendung stehen in <code>&lt;DATASETS&gt;\\QUELLEN.md</code>, '
      'jedes Bild mit Herkunft im Manifest seiner Quelle. Für alle Web- und Telegram-Quellen gilt eine harte '
      'Sperre gegen Darstellungen Minderjähriger: Booru- und WD-Tags (loli, child …) bei jedem Rating; dazu bei '
      'nicht jugendfreien Bildern eine Liste kanonisch minderjähriger Figuren und Serien. Die Regel wurde '
      'rückwirkend auf alle Quellen angewandt; gesperrte Bilder werden weder verwendet noch angezeigt.</p>')

    a('<h2>Grenzen und nächste Schritte</h2>')
    a('<ul><li>Frühe KI von 2022: DiffusionDB schließt die Stable-Diffusion-Seite teilweise (67 → 76&nbsp;%); '
      'frühes Midjourney (v3/v4) fehlt weiter — eine saubere, lizenzierte Quelle dafür ist offen.</li>'
      '<li>p3 gegen p2 ist ein kleiner Schritt; ein zweiter Trainingslauf mit anderen Seeds würde zeigen, '
      'wie viel davon Streuung ist.</li>'
      '<li>Wallhaven-Wallpaper bleiben die schwerste menschliche Gruppe (Konzeptkunst, 3D-Renderings).</li>'
      '<li>Die Kanäle sind echte KI-Posts, aber nicht jedes Bild ist garantiert KI; Trefferquoten sind '
      'eher untere Schranken.</li>'
      '<li>Der Scanner <code>pruefe.py</code> nutzt jetzt p3 (fünf Köpfe, Schwelle auf drei menschlichen '
      'val-Gruppen). Die Produktion ist unverändert.</li></ul>')
    a('</main>')
    a(f"<script>{JS}</script>")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(h))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
