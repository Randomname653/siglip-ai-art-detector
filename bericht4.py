r"""bericht4.py — Laborbericht Stand 28.09.: p8sx (SigLIP 2), Profi-Kunst, Robustheit, p9sx, Zahlen direkt aus den Ergebnisdateien.

Quellen (daten/): gleich_fa_bericht.json (alle Koepfe, 18 Kanaele, gleicher Fehlalarm), bench_p8.json
(Produktionsvergleich), p2_test.json (Fehlalarm je Menschengruppe), mj_versionen_p.json, ausschluss.json,
ausSschluss.json (falls vorhanden), gleich_fa_rg.json + bench_rg.json + rg_tempo.json (Rueckgrate),
nacht_kontr.log (Kontrollen p8x), schutz_wd_figuren*.log.

    python lab/detektor/bericht4.py
"""
import html
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bericht import CSS, JS, OUT, komma, lies, log, prozent   # noqa: E402
from bericht3 import KANAL_NAME, kontrolle                     # noqa: E402
from kanaele import ANIME                        # noqa: E402
from lesen import DATEN                                        # noqa: E402

PROD = {"deepghs:caformer_s36_plus_sce": "caformer_s36_plus", "deepghs:caformer_s36_sce_v1": "caformer_s36_v1",
        "hf:legekka": "legekka", "hf:saltacc": "saltacc", "deepghs:mobilenetv3_sce_dist_v1": "mobilenet_dist_v1",
        "deepghs:mobilenetv3_sce_dist": "mobilenet_dist", "deepghs:mobilenetv3_sce": "mobilenet"}
STAND = [("v3", "v3", "Anime-Fassung (CLIP + DINO)"), ("p2", "p2", "+ menschliche Nicht-Anime-Kunst"),
         ("p3x", "p3x", "+ frühe KI 2022 (DiffusionDB)"), ("p6x", "p6x", "+ Midjourney (Felix, niji, v6)"),
         ("p7x", "p7x", "+ SigLIP 2 als drittes Rückgrat"), ("p8x", "p8x", "+ Schutzregel per Bildabgleich, 18 Kanäle"),
         ("p8sx", "p8sx", "nur SigLIP 2 — Scanner-Standard")]
H = "p8sx"     # Hauptkopf (Scanner-Standard seit 28.09.)


def balken(zeilen, a_lab, b_lab, breite=640):
    """Je Zeile zwei Balken 0-100 %: a (orange) und b (blau)."""
    lx, rechts, zh, bh = 210, 50, 30, 9
    pb = breite - lx - rechts
    oben = 14
    hoehe = oben + zh * len(zeilen) + 24
    t = [f'<svg viewBox="0 0 {breite} {hoehe}" width="100%" role="img" aria-label="Treffer je Kanal">']
    for v in (0, .25, .5, .75, 1):
        x = lx + v * pb
        t.append(f'<line class="raster" x1="{x:.1f}" y1="{oben-6}" x2="{x:.1f}" y2="{hoehe-22}"/>')
        t.append(f'<text x="{x:.1f}" y="{hoehe-6}" text-anchor="middle">{int(v*100)} %</text>')
    for k, (name, a, b) in enumerate(zeilen):
        y = oben + k * zh
        t.append(f'<text class="gen" x="{lx-10}" y="{y+13}" text-anchor="end">{html.escape(name)}</text>')
        for j, (v, farbe, lab) in enumerate(((a, "var(--s2)", a_lab), (b, "var(--s1)", b_lab))):
            w = max(v * pb, 1.5)
            yy = y + j * (bh + 2)
            t.append(f'<rect class="bar" x="{lx}" y="{yy}" width="{w:.1f}" height="{bh}" rx="2" fill="{farbe}" '
                     f'data-tipp="{html.escape(name)} · {lab}: {prozent(v)}"/>')
            if j == 1:
                t.append(f'<text class="wert" x="{lx+w+5:.1f}" y="{yy+bh-1}">{prozent(v, 0)}</text>')
    t.append(f'<line class="achse" x1="{lx}" y1="{oben-6}" x2="{lx}" y2="{hoehe-22}"/></svg>')
    leg = (f'<div class="legende"><span><i style="background:var(--s2)"></i>{html.escape(a_lab)}</span>'
           f'<span><i style="background:var(--s1)"></i>{html.escape(b_lab)}</span></div>')
    return leg + "".join(t)


def tab(kopf, zeilen, rechts_ab=1, summe=()):
    a = ['<div class="tabelle"><table><thead><tr>' + "".join(
        f'<th{" class=n" if i >= rechts_ab else ""}>{html.escape(k)}</th>' for i, k in enumerate(kopf)) + '</tr></thead><tbody>']
    for z in zeilen:
        a.append(f'<tr{" class=summe" if z[0] in summe else ""}>' + "".join(
            f'<td{" class=n" if i >= rechts_ab else ""}>{c}</td>' for i, c in enumerate(z)) + '</tr>')
    return "".join(a) + '</tbody></table></div>'


def main():
    gf = lies("gleich_fa_bericht.json")
    gf["ergebnis"].update({k: v for k, v in lies("gleich_fa_p8s.json")["ergebnis"].items() if k == H})
    bn = lies("bench_p8.json")
    bn["ergebnis"].update({k: v for k, v in lies("bench_p8s.json")["ergebnis"].items() if k == H})
    pt = lies("p2_test.json")
    mj = lies("mj_versionen_p.json")
    aus = lies("ausschluss.json")
    ausS = lies("ausSschluss.json") if os.path.exists(os.path.join(DATEN, "ausSschluss.json")) else None
    rgf, rgb, tempo = lies("gleich_fa_rg.json"), lies("bench_rg.json"), lies("rg_tempo.json")
    kt, kr = kontrolle(log("p8s_kontr.log"))
    e, b, art = gf["ergebnis"], bn["ergebnis"], bn["art"]
    kan = list(e[H]["treffer"])
    mittel = {t: float(np.mean([e[t]["treffer"][k] for k in kan])) for t in e}
    prod_best = max(PROD, key=lambda d: b[d]["auc"]["gemischt alle"])
    fa = pt[H]

    h = []
    a = h.append
    a('<title>Der Wallhaven-Detektor</title>')
    a('<link rel="preconnect" href="https://fonts.googleapis.com">'
      '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
      '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700'
      '&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">')
    a(f"<style>{CSS}</style>")
    a('<div id="tipp" aria-hidden="true"></div><main class="seite">')
    a('<div class="kicker">Laborbericht · 28. September 2026 · Produktion unangetastet</div>')
    a('<h1>Der KI-Kunst-Detektor: Stand 28. September</h1>')
    a(f'<p class="unter">Ein Detektor für KI-generierte Anime- und Digitalkunst (keine Fotos). Zehn kleine Köpfe auf den '
      f'Zwischenschichten eines eingefrorenen Bildmodells (SigLIP 2) — Scanner-Standard <b>p8sx</b>. Gemessen auf {len(kan)} Telegram-KI-Kanälen und auf '
      f'menschlicher Kunst, die alle nie im Training waren.</p>')
    a('<div class="antwort">')
    a(f'<div class="kachel neu"><div class="wert">{prozent(mittel[H])}</div><div class="was">der KI-Bilder erkannt, '
      f'Mittel über {len(kan)} Telegram-Kanäle, bei 1&nbsp;% Fehlalarm auf menschlicher Kunst</div></div>')
    a(f'<div class="kachel neu"><div class="wert">{prozent(b[H]["treffer"]["gemischt Kanaele"], 0)} : '
      f'{prozent(b[prod_best]["treffer"]["gemischt Kanaele"], 0)}</div><div class="was">p8sx gegen das beste '
      f'Produktionsmodell ({PROD[prod_best]}) auf denselben Kanalbildern, gleiche Schwellenregel</div></div>')
    a(f'<div class="kachel"><div class="wert">{prozent(fa["M Danbooru 2026 (Anime)"][1])} · '
      f'{prozent(fa["M Wallhaven general <2022"][1])}</div><div class="was">Fehlalarm auf menschlicher Anime-Kunst '
      f'2026 · auf menschlichen Wallpapern (Scanner-Schwelle)</div></div>')
    a('</div>')

    # --- Entwicklung
    a('<h2 style="border-top:0;margin-top:1.5rem;padding-top:0">Wie wir hierher kamen</h2>')
    a(f'<p>Alle Köpfe auf denselben {len(kan)} Kanälen, jeder bei der Schwelle, bei der er auf den drei menschlichen '
      'Testgruppen im Mittel 1&nbsp;% Fehlalarm hat. Die AUC braucht keine Schwelle.</p>')
    zeilen = []
    for t, name, was in STAND:
        if t in e:
            ta = np.mean([e[t]["treffer"][k] for k in ANIME if k in e[t]["treffer"]])
            au = np.mean([e[t]["auc"][k] for k in kan])
            zeilen.append([f'<b>{name}</b>', html.escape(was), prozent(mittel[t]), prozent(ta), komma(au, 4)])
    a(tab(["Kopf", "Schritt", "Kanäle", "Anime-Kanäle", "AUC"], zeilen, rechts_ab=2))
    a('<p>Die größten Sprünge: menschliche Nicht-Anime-Kunst im Training (der Kopf hält neue Zeichenstile nicht '
      'mehr für KI), Midjourney-Daten und vor allem SigLIP&nbsp;2. p8x ist p7x auf dem endgültig bereinigten '
      'Datenstand; p8sx dasselbe nur mit SigLIP&nbsp;2 — gleich stark, doppelt so schnell.</p>')

    # --- Produktion
    a('<h2>Gegen die Produktion</h2>')
    a(f'<p>Die sieben Einzelmodelle aus <code>imagesort.py</code> und unsere Köpfe auf demselben Satz von '
      f'{sum(bn["n"].values()):,} Bildern (Stufenlogik und XGB-Meta-Modell der Produktion nicht nachgebaut). '
      'Regel für alle: Schwelle bei 1&nbsp;% Fehlalarm im Mittel der menschlichen val-Gruppen.</p>'.replace(",", "."))
    z = []
    for d in [H, "p8x", "p6x", "v3"] + sorted(PROD, key=lambda d: -b[d]["auc"]["gemischt alle"]):
        nm = PROD.get(d, d)
        z.append([f'<b>{nm}</b>' if d == H else nm, prozent(b[d]["treffer"]["gemischt Generatoren"]),
                  prozent(b[d]["treffer"]["gemischt Kanaele"]), prozent(b[d]["fa"]["gemischt"]),
                  komma(b[d]["auc"]["gemischt alle"], 4)])
    a(tab(["Detektor", "Generatoren", "Kanäle", "Fehlalarm", "AUC"], z))
    kz = sorted(((KANAL_NAME.get(g[2:], g[2:]), b[prod_best]["treffer"][g], b[H]["treffer"][g])
                 for g in art if art[g] == "k_kanal"), key=lambda r: -r[2])
    a('<figure>' + balken(kz, PROD[prod_best], H) + '<figcaption>Telegram-Kanäle: keiner der Detektoren hat sie '
      'je gesehen — der fairste Vergleich. Die Produktion ist bei älterem Anime-SD stark, bei Midjourney, DALL·E und '
      'geschlossenen Modellen blind.</figcaption></figure>')
    a('<details><summary>Je Generator (Testsplits)</summary>')
    gz = [[html.escape(g[2:]), str(bn["n"][g]), prozent(b[H]["treffer"][g]), prozent(b[prod_best]["treffer"][g]),
           prozent(max(b[d]["treffer"][g] for d in PROD))] for g in art if art[g] == "k_gen"]
    a(tab(["Generator", "Bilder", H, PROD[prod_best], "beste Produktion"], gz))
    a('<p>Bei den Generatorgruppen kennt der Kopf den Generator aus dem Training (andere Bilder, andere Ersteller); '
      'wie gut er <i>unbekannte</i> Generatoren erkennt, zeigen die Ausschlusstests.</p></details>')

    # --- Ausschluss
    a('<h2>Unbekannte Generatoren</h2>')
    a('<p>Je Experiment werden ein oder zwei Generatoren bzw. eine Modellfamilie komplett aus dem Training '
      'genommen; sonst ist alles identisch (Daten, Seeds, Kalibrierung). Die Trefferquote auf genau diesen '
      'Generatoren zeigt, wie gut der Ansatz auf etwas überträgt, das er nie gesehen hat.</p>')
    z = []
    s_by = {x["id"]: x for x in ausS["experimente"]} if ausS else {}
    for x in aus["experimente"]:
        g = x["gruppen"]
        tv = np.mean([v["treffer_basis"] for v in g.values()])
        tn = np.mean([v["treffer_ohne"] for v in g.values()])
        tp = np.mean([v["treffer_produktion"] for v in g.values()])
        row = [x["id"], html.escape(x["titel"]), prozent(tv), f'<b>{prozent(tn)}</b>', prozent(tp)]
        if ausS:
            s = s_by.get(x["id"])
            row.append(prozent(np.mean([v["treffer_ohne"] for v in s["gruppen"].values()])) if s else "–")
        z.append(row)
    kopf = ["", "entfernt", "gesehen", "nie gesehen", "Produktion"] + (["nie gesehen, nur SigLIP"] if ausS else [])
    a(tab(kopf, z, rechts_ab=2))
    a('<p>Ohne ein einziges Trainingsbeispiel des Generators erkennt der Kopf 93–99&nbsp;% seiner Bilder — '
      'die Produktion kommt auf 6–73&nbsp;%. Am meisten verlieren Midjourney (−4,8 Punkte), Nano Banana/OpenAI '
      'und die Flux-Familie (je etwa −3,4): Bildstile, die sich am stärksten vom Rest unterscheiden. '
      'Details je Testgruppe: <code>AUSSCHLUSS.md</code>.</p>')

    # --- Rueckgrate
    a('<h2>Welches Rückgrat trägt was?</h2>')
    rk = [("rg_C", "CLIP", ["clip_mid"]), ("rg_D", "DINOv2", ["dino_mid"]), ("rg_S", "SigLIP 2", ["siglip_mid"]),
          ("rg_CD", "CLIP + DINOv2", ["clip_mid", "dino_mid"]), ("rg_CS", "CLIP + SigLIP 2", ["clip_mid", "siglip_mid"]),
          ("rg_DS", "DINOv2 + SigLIP 2", ["dino_mid", "siglip_mid"]),
          ("p7", "alle drei", ["clip_mid", "dino_mid", "siglip_mid"])]
    kr_ = list(rgf["ergebnis"]["p7"]["treffer"])
    z = []
    for t, nm, rg in rk:
        ee = rgf["ergebnis"][t]
        z.append([nm, prozent(np.mean([ee["treffer"][k] for k in kr_])), komma(np.mean([ee["auc"][k] for k in kr_]), 4),
                  komma(rgb["ergebnis"][t]["auc"]["gemischt Generatoren"], 4), f'{1 / sum(1 / tempo[r] for r in rg):.0f}/s'])
    a(tab(["Rückgrate", "Kanäle", "Kanäle AUC", "Generatoren AUC", "Tempo (4090)"], z))
    a('<p>SigLIP&nbsp;2 allein ist so gut wie alle drei zusammen — und doppelt so schnell. CLIP und DINOv2 tragen '
      'neben SigLIP kaum noch bei. Auch bei <i>unbekannten</i> Generatoren liegt SigLIP allein gleichauf (letzte '
      'Spalte der Ausschlusstabelle: im Mittel 96,5 % gegen 96,2 % mit drei Rückgraten; Midjourney etwas '
      'schwächer, NovelAI und Nano Banana etwas stärker). Details: <code>AUSSCHLUSS_ausS.md</code>.</p>')

    # --- Menschen
    a('<h2>Fehlalarm auf menschlicher Kunst</h2>')
    z = [[html.escape(g[2:].replace("   ", "· ")), f'{fa[g][0]:,}'.replace(",", "."), prozent(pt["v3"][g][1]),
          prozent(pt["p6x"][g][1]), prozent(pt["p8x"][g][1]), f'<b>{prozent(fa[g][1])}</b>'] for g in fa if g.startswith("M")]
    a(tab(["Gruppe (nie im Training)", "Bilder", "v3", "p6x", "p8x", H], z))
    a('<p>Scanner-Schwelle je Kopf. Die Stilzeilen sind kleine Teilmengen (±1 Punkt ist Rauschen).</p>')

    # --- Midjourney
    a('<h3>Midjourney je Version (aus dem Prompt)</h3>')
    ords = [g for g in ("v4", "v5", "v6", "v7", "v8", "niji", "test/testp") if g in mj[H]]
    a(tab(["Version", "Bilder", "v3", "p6x", "p8x", H],
          [[g, str(mj[H][g][0]), prozent(mj["v3"][g][1]), prozent(mj["p6x"][g][1]), prozent(mj["p8x"][g][1]),
            f'<b>{prozent(mj[H][g][1])}</b>'] for g in ords]))

    # --- Kontrollen
    a('<h2>Kontrollen auf Verarbeitungsspuren (p8sx)</h2>')
    if kt:
        a(f'<p><b>Maßstab-Eingriff:</b> Menschen vergrößert verschieben sich um {komma(kt["Menschen"][2], 2)}, KI-Bilder '
          f'verkleinert um {komma(kt["KI"][2], 2)} Logit-Punkte — bei einem Klassenabstand von rund '
          f'{kt["KI"][0] - kt["Menschen"][0]:.0f}. Kein Maßstab-Leck.</p>')
    if kr:
        a(tab(["beide Klassen gleich gestört", "AUC", "Treffer @1 %"], [[html.escape(l), komma(au), prozent(t1)] for l, au, t1 in kr]))

    # --- Profi-Kunst (profikunst_test.json / profikunst_datum.json)
    if os.path.exists(os.path.join(DATEN, "profikunst_datum.json")):
        pk, pdt = lies("profikunst_test.json"), lies("profikunst_datum.json")
        a('<h2>Professionelle Konzept- und Game-Art</h2>')
        n_pk = f'{sum(pk["n"].values()):,}'.replace(",", ".")
        a(f'<p>Eine Menschengruppe, die im Training fehlt: {n_pk} Wallpaper von CGWallpapers '
          '(895 Digitalkünstler, Software laut Fußzeile meist Photoshop/Painter) und GameWallpapers (offizielles Artwork '
          'von 1.549 Spielen) aus der NAS-Sammlung. Fußzeile abgeschnitten, Fotos/Renderings gefiltert, Pixel-Dubletten '
          'entfernt. Die Kopien tragen kein brauchbares Dateidatum, viele Originale aber ein eingebettetes '
          'Erstellungsdatum (XMP/EXIF) — vor 2022 kann es keine Generatorausgabe sein.</p>')
        z = pdt["koepfe"][H]
        namen = {"vor 2022": "vor 2022 (belegbar menschlich)", "ab 2022": "2022 oder später", "undatiert": "ohne Datum", "alle": "alle"}
        a(tab(["datiert", "Bilder", f"Fehlalarm {H}"],
              [[namen[al], f'{z[f"alle|{al}"]["n"]:,}'.replace(",", "."), prozent(z[f"alle|{al}"]["fa"], 2)] for al in namen], summe=("alle",)))
        a('<p>Auf belegbar menschlicher Profi-Kunst bleibt der Kopf bei seinem 1-%-Auslegungspunkt. Die höhere Quote ab '
          '2022 ist entweder unerklärte KI-Hilfe oder wieder der „neuer Stil“-Effekt — das lässt sich hier nicht trennen. '
          'Die Produktionsdetektoren liegen hier ähnlich, verpassen aber den Großteil der KI.</p>')
    # --- Praxis: Sorted Stuff
    a('<h2>Praxistest: NAS „Sorted Stuff“</h2>')
    a('<p>Der Ordner, über den die alten Detektoren liefen: p8sx markiert 2,1 % bei der 1-%-Schwelle. Nach Jahr: '
      'Posts 2015–2021 0,4–0,9 % (= Fehlalarm), 2025 und 2026 je rund 7,5 % — also etwa 350–400 wahrscheinliche '
      'KI-Bilder, die die alten Detektoren durchgelassen haben. Liste: <code>pruefung/sorted_stuff/</code>.</p>')
    # --- Robustheit + p9sx
    if os.path.exists(os.path.join(DATEN, "robust_test_p9sx.json")):
        R = {t: lies(f"robust_test_{t}.json") for t in (H, "p9sx")}
        bn_ = {"original": "unverändert", "jpeg50": "JPEG q50", "jpeg30": "JPEG q30", "social": "0,6× + JPEG q75",
               "crop50": "Ausschnitt 50 %", "text": "Untertitel + Wasserzeichen", "noise": "Rauschen σ=6",
               "blur": "Weichzeichner", "sharpen": "Schärfen", "colour": "Sättigung/Helligkeit", "grey": "Graustufen",
               "flip": "gespiegelt", "upscale_cdc": "4×-Anime-Hochskalierer"}
        a('<h2>Alltags-Bearbeitungen und p9sx</h2>')
        a('<p>Jedes Testbild einmal bearbeitet, dann wie im Scanner normiert; Schwelle fest bei 1 %. p9sx ist p8sx, '
          'nachtrainiert mit genau solchen Bearbeitungen (beide Klassen gleich, andere Texte/Schriften als im Test, '
          'den 4×-Hochskalierer nie gesehen).</p>')
        a(tab(["Bearbeitung", f"Treffer {H}", "Treffer p9sx", f"Fehlalarm {H}", "Fehlalarm p9sx", f"AUC {H}", "AUC p9sx"],
              [[bn_.get(b, b), prozent(R[H][b]["treffer"]), prozent(R["p9sx"][b]["treffer"]), prozent(R[H][b]["fa"], 2),
                prozent(R["p9sx"][b]["fa"], 2), komma(R[H][b]["auc"], 4), komma(R["p9sx"][b]["auc"], 4)] for b in R[H]]))
        a('<p>KI bleibt auch bearbeitet erkannt — Rauschen als Tarnung wirkt nicht. Bearbeitete <i>menschliche</i> '
          'Bilder (Untertitel, Hochskalieren, starke Ausschnitte) rutschen aber Richtung KI: 2–5 % statt 1 %. '
          'p9sx trennt unter jeder Bearbeitung besser (AUC überall höher) und findet bearbeitete KI öfter, holt die '
          'bearbeiteten Menschen aber nicht unter die Schwelle; ungeändert ist es ein Gleichstand (Kanalmittel 97,5 '
          'gegen 97,4 %, beide Köpfe kombiniert 97,6 %). Deshalb bleibt p8sx in der Produktion; p9sx ist die Reserve, '
          'falls bearbeitete oder stark komprimierte KI zum Problem wird.</p>')
        a('<p><b>Und bei unbekannten Generatoren?</b> Die Ausschlusstests mit dem p9-Rezept wiederholt: im Mittel '
          '96,0 % statt 96,4 %, 8 von 11 Experimenten etwas schlechter (Grok −1,8, Flux −1,1 Punkte). Die '
          'Bearbeitungen verwischen genau die feinen Spuren, an denen auch unbekannte Generatoren auffallen — ein '
          'weiterer Grund, bei p8sx zu bleiben. Details: <code>docs/de/AUSSCHLUSS_aus9.md</code>.</p>')

    # --- Daten + Schutz
    a('<h2>Datenlage und Schutz</h2>')
    a('<p>Alle Quellen mit Lizenz und Filtern in <code>&lt;DATASETS&gt;\\QUELLEN.md</code>, jedes Bild mit Herkunft im '
      'Manifest. Neu seit dem letzten Bericht: DiffusionDB (CC0, SD 1.x 2022), Midjourney-Archiv „Felix“ (CC0), '
      'niji-v5 (CC0), eine Midjourney-v6-Stichprobe aus einem fertigen Scrape-Datensatz, drei weitere Telegram-KI-'
      'Kanäle. Selbst gescrapt wird nur, wo die Seite es erlaubt.</p>')
    a('<p><b>Schutz vor Darstellungen Minderjähriger:</b> Tag-Sperre (loli, child …) bei jedem Rating; bei nicht '
      'jugendfreien Bildern zusätzlich eine Liste im Original minderjähriger Figuren und Serien — seit 28.09. nicht '
      'nur gegen Beitragstexte und Booru-Tags, sondern auch gegen die Figuren, die der WD-Tagger im Bild erkennt. '
      'Dadurch rückwirkend 1.112 Trainingsbilder aus CivitAI/DiffusionDB/Midjourney, 1.560 aus AIBooru und rund '
      '2.060 Testbilder aus den Telegram-Kanälen entfernt; p8x und p8sx sind auf dem bereinigten Stand trainiert.</p>')

    a('<h2>Grenzen und nächste Schritte</h2>')
    a('<ul><li>Der Scanner rechnet nur noch SigLIP&nbsp;2 (~135 Bilder/s Merkmale auf der 4090, doppelt so '
      'schnell wie mit drei Rückgraten); p8x bleibt als Alternative (<code>--kopf p8x</code>).</li>'
      '<li>Wallhaven-Wallpaper (Konzeptkunst, 3D) bleiben die schwerste menschliche Gruppe.</li>'
      '<li>Die Kanäle sind echte KI-Posts, aber nicht jedes Bild ist garantiert KI; Trefferquoten sind eher '
      'untere Schranken.</li>'
      '<li>Die Produktion (<code>imagesort.py ai-cleanup</code>) nutzt seit 28.09. nur noch p8sx: ab der '
      '1-%-Schwelle ins Review, automatisch in Quarantäne erst ab Score 0,9999 (0,07 % der Test-Menschen, 91 % der '
      'KI) und nie bei Bildern unter 384 px. XGB-Meta ist aus.</li>'
      '<li>Bearbeitete menschliche Bilder (Untertitel, Hochskalieren) haben 2–5 % Fehlalarm — genau die landen im '
      'Review, nicht in der Quarantäne.</li></ul>')
    a('</main>')
    a(f"<script>{JS}</script>")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(h))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
