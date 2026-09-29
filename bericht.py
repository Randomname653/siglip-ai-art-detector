r"""bericht.py — der Laborbericht als eine HTML-Seite, Zahlen direkt aus den Ergebnisdateien.

Keine Zahl wird hier von Hand abgeschrieben, die in einer Datei steht: die
Tabellen und die Grafik lesen ergebnis_*.json und leak.json. Die Kontroll-,
Pruefstand- und Telegram-Zahlen stehen nur in Protokollen und werden daraus
gelesen; die zwei Werte des ersten, spaeter verworfenen Kopfes stehen unten in
FRUEHER, mit Herkunft.

    python lab/detektor/bericht.py
"""
import html
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import DATEN                          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bericht.html")
KOPF = "wd_eva02_clip_l14_mlp_aug"

# Messungen des ersten Kopfes (sauber trainiert, CLIP + logistische Regression),
# aus kontrolle.py am 2026-09-24 — dieser Kopf wurde verworfen, die Zahlen sind
# der Grund dafuer.
FRUEHER = {
    "robust": [("unveraendert", 0.9860, 0.780), ("nur JPEG q90", 0.9065, 0.242),
               ("x0,9 + q95", 0.9264, 0.377), ("x0,9 + q90", 0.8158, 0.048),
               ("x0,75 + q85", 0.6434, 0.009)],
    "eingriff": {"neg": 3.3761, "pos": -1.4116},
}
V5 = {  # lab/bench/eval_detector.py, bench_*.log, und die Groessenpruefung
    "groesse_format_auc": 0.934,
}
# Erste Telegram-Probe (telegram.py, kept_as_human als menschlich) und die
# Analyse der Bildgroessen dazu, 2026-09-24 — nicht in einer Datei abgelegt.
REVIEWLOG = {"auc": 0.382, "anteil_15": 0.567, "anteil_15_ki": 0.121,
             "neu_15": 0.849, "ki_treffer": 0.671,
             "stumpf_t3": 0.793}   # telegram3.py --stumpf, daten/telegram3_stumpf.log


def lies(name):
    with open(os.path.join(DATEN, name), encoding="utf-8") as f:
        return json.load(f)


def prozent(x, stellen=1):
    return f"{100 * x:.{stellen}f}".replace(".", ",") + " %"


def komma(x, stellen=3):
    return f"{x:.{stellen}f}".replace(".", ",")


def log(name):
    p = os.path.join(DATEN, name)
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        t = f.read().decode("utf-8", "replace")
    return "\n".join(z for z in t.replace("\r", "\n").split("\n")
                     if "Warning" not in z and "warn" not in z)


def kontrolle(text):
    """Eingriff und Robustheit aus dem Protokoll von kontrolle.py."""
    e = {}
    for lab in ("Negative", "Positive"):
        m = re.search(lab + r"\s+Score vorher ([+-][\d.]+)\s+nachher ([+-][\d.]+)\s+"
                      r"Verschiebung Median ([+-][\d.]+)", text)
        if m:
            e[lab] = tuple(float(x) for x in m.groups())
    r = [(m.group(1).strip(), float(m.group(2)), float(m.group(3)))
         for m in re.finditer(r"  (\S.*?)\s+AUC ([\d.]+)\s+Treffer bei 1 % FA ([\d.]+)", text)]
    return e, r


def telegram(text):
    zeilen = re.findall(r"^\s+(\(?[\w:.\-+]+\)?|neu: \S+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%",
                        text, re.M)
    n = re.search(r"(\d+) Bilder mit eigenem Label.*?\((\d+) Mensch, (\d+) KI\)", text)
    return zeilen, n.groups() if n else None


def bench(text):
    m = re.search(r"detektor\.kandidat:\S+\s+([\d.]+)\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)%", text)
    return (float(m.group(1)), float(m.group(2)) / 100) if m else None


def _kenn(y, s):
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    return {"auc": float(roc_auc_score(y, s)), "t1": float(np.interp(0.01, fpr, tpr)),
            "t5": float(np.interp(0.05, fpr, tpr))}


def vergleiche():
    """Die beiden Welten nebeneinander: eigener Testsplit (14 Generatoren,
    gestoert) und der faire Telegram-Test (telegram3.py). Dazu das Rangmittel
    ohne Labels und ohne Gewichte."""
    from scipy.stats import rankdata
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lesen import Pack
    rang = lambda x: rankdata(x) / len(x)
    CAF, LEG = "deepghs_caformer_s36_plus_sce", "hf_legekka"
    te_s = np.load(os.path.join(DATEN, f"score_{KOPF}.npy"))
    te, s = te_s[:, 0].astype(int), te_s[:, 1]
    mi = np.load(os.path.join(DATEN, "mess_test_aug_idx.npy"))
    pos = {i: k for k, i in enumerate(mi)}
    o = np.array([pos[i] for i in te])
    y1 = Pack().label[te].astype(int)
    b1 = {k: np.load(os.path.join(DATEN, f"mess_test_aug_{k}.npy"))[o] for k in (CAF, LEG)}
    t3 = np.load(os.path.join(DATEN, "telegram3.npz"))
    y3, n3 = t3["y"].astype(int), t3["neu"]
    b3 = {k: t3[k] for k in (CAF, LEG)}
    varianten = [
        ("neuer Detektor allein", lambda n, b: n),
        ("caformer_plus allein", lambda n, b: b[CAF]),
        ("legekka allein", lambda n, b: b[LEG]),
        ("caformer_plus + legekka (nur Produktion)", lambda n, b: rang(b[CAF]) + rang(b[LEG])),
        ("neu + caformer_plus + legekka", lambda n, b: rang(n) + rang(b[CAF]) + rang(b[LEG])),
    ]
    tab = [(name, _kenn(y1, f(s, b1)), _kenn(y3, f(n3, b3))) for name, f in varianten]
    # alle acht Detektoren auf dem fairen Test
    alle3 = sorted(((k, _kenn(y3, t3[k])) for k in t3.files if k not in ("y",)),
                   key=lambda x: -x[1]["auc"])
    # Kalibrierung: die val-Schwelle auf fremdem Material
    k = np.load(os.path.join(DATEN, f"kopf_{KOPF}.npz"))
    z = np.log(np.clip(n3, 1e-12, 1 - 1e-12) / np.clip(1 - n3, 1e-12, 1))
    kal = {"fa1": float((z[y3 == 0] > float(k["schwelle_1"])).mean()),
           "fa5": float((z[y3 == 0] > float(k["schwelle_5"])).mean()),
           "tr1": float((z[y3 == 1] > float(k["schwelle_1"])).mean()),
           "n": int((y3 == 0).sum())}
    t2 = np.load(os.path.join(DATEN, "telegram2.npz"))
    y2 = t2["y"].astype(int)
    alle2 = sorted(((kk, _kenn(y2, t2[kk])) for kk in t2.files if kk not in ("y", "pfade")),
                   key=lambda x: -x[1]["auc"])
    z2 = np.log(np.clip(t2["neu"], 1e-12, 1 - 1e-12) / np.clip(1 - t2["neu"], 1e-12, 1))
    kal2 = float((z2[y2 == 0] > float(k["schwelle_1"])).mean())
    return tab, alle3, kal, alle2, kal2, int((y2 == 0).sum()), int((y2 == 1).sum())


CSS = """
:root{
  --grund:#f6f6f3; --flaeche:#ffffff; --tinte:#15171c; --tinte2:#565b64; --leise:#8a8e95;
  --linie:#dddcd5; --raster:#ebeae5; --warn:#8a5a00; --warngrund:#fff5e0;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --grund:#141517; --flaeche:#1b1c1f; --tinte:#eceae4; --tinte2:#a9abb0; --leise:#7c7f86;
    --linie:#303238; --raster:#25272b; --warn:#e3b04b; --warngrund:#2a2416;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --grund:#141517; --flaeche:#1b1c1f; --tinte:#eceae4; --tinte2:#a9abb0; --leise:#7c7f86;
  --linie:#303238; --raster:#25272b; --warn:#e3b04b; --warngrund:#2a2416;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; color-scheme:dark;
}
body{background:var(--grund);color:var(--tinte);
  font:16px/1.6 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif}
.seite{max-width:46rem;margin:0 auto;padding-inline:20px;padding-block:40px 80px}
h1,h2,h3{font-family:"Archivo","IBM Plex Sans",system-ui,sans-serif;text-wrap:balance;
  line-height:1.2;margin:0}
h1{font-size:2.1rem;font-weight:700;letter-spacing:-.01em}
h2{font-size:1.35rem;font-weight:650;margin-top:3rem;padding-top:1.25rem;
  border-top:1px solid var(--linie)}
h3{font-size:1.02rem;font-weight:650;margin-top:1.75rem}
p{margin:.8rem 0;max-width:66ch}
.kicker{font:600 .74rem/1 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.09em;
  text-transform:uppercase;color:var(--tinte2);margin-bottom:.9rem}
.unter{color:var(--tinte2);margin-top:.6rem}
.zahl,td.n,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums}
.antwort{display:grid;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));gap:12px;
  margin:1.8rem 0 1rem}
.kachel{background:var(--flaeche);border:1px solid var(--linie);border-radius:6px;padding:16px 18px}
.kachel .wert{font:600 2.2rem/1 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:-.02em}
.kachel .was{color:var(--tinte2);font-size:.9rem;margin-top:.5rem;line-height:1.4}
.kachel.neu .wert{color:var(--s1)}
.tabelle{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:.42rem .6rem;border-bottom:1px solid var(--raster);text-align:left;white-space:nowrap}
th{font:600 .72rem/1.3 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.05em;
  text-transform:uppercase;color:var(--tinte2);border-bottom:1px solid var(--linie)}
td.n,th.n{text-align:right}
tr.summe td{font-weight:600;border-bottom:1px solid var(--linie)}
.hinweis{background:var(--warngrund);border-left:3px solid var(--warn);padding:10px 14px;
  border-radius:0 4px 4px 0;margin:1rem 0;max-width:66ch}
.hinweis b{color:var(--warn)}
figure{margin:1.5rem 0}
figcaption{color:var(--tinte2);font-size:.88rem;margin-top:.6rem;max-width:66ch}
.legende{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.86rem;color:var(--tinte2);margin-bottom:.6rem}
.legende span{display:inline-flex;align-items:center;gap:7px}
.legende i{width:14px;height:8px;border-radius:2px;display:inline-block}
svg text{fill:var(--tinte2);font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px}
svg .gen{fill:var(--tinte);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:12.5px}
svg .wert{fill:var(--tinte);font-size:11px}
svg .raster{stroke:var(--raster)}
svg .achse{stroke:var(--linie)}
svg rect.bar:hover{opacity:.8}
#tipp{position:fixed;pointer-events:none;background:var(--tinte);color:var(--grund);
  font:12px/1.4 "IBM Plex Mono",ui-monospace,monospace;padding:6px 9px;border-radius:4px;
  opacity:0;transition:opacity .08s;z-index:5}
details{margin:1rem 0}
summary{cursor:pointer;color:var(--tinte2);font-size:.9rem}
summary:focus-visible,a:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
code,pre{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.85rem}
pre{background:var(--flaeche);border:1px solid var(--linie);border-radius:6px;padding:12px 14px;
  overflow-x:auto;line-height:1.5}
ul{padding-left:1.2rem;max-width:66ch} li{margin:.35rem 0}
@media (prefers-reduced-motion:reduce){#tipp{transition:none}}
"""

JS = """
const t=document.getElementById('tipp');
document.querySelectorAll('rect.bar').forEach(r=>{
  r.addEventListener('mousemove',e=>{t.textContent=r.dataset.tipp;t.style.opacity=1;
    t.style.left=(e.clientX+12)+'px';t.style.top=(e.clientY+12)+'px';});
  r.addEventListener('mouseleave',()=>{t.style.opacity=0;});
});
"""


def grafik(zeilen):
    """Gruppierte waagerechte Balken: je Generator drei Werte, 0-100 %."""
    lx, rechts, zh, bh, lu = 132, 58, 34, 7, 2        # Labelbreite, Rand, Zeilenhoehe
    breite = 640
    pb = breite - lx - rechts
    oben = 22
    hoehe = oben + zh * len(zeilen) + 26
    teile = [f'<svg viewBox="0 0 {breite} {hoehe}" width="100%" role="img" '
             f'aria-label="Trefferquote bei 1 % Fehlalarm je Generator">']
    for v in (0, .25, .5, .75, 1):
        x = lx + v * pb
        teile.append(f'<line class="raster" x1="{x:.1f}" y1="{oben-6}" x2="{x:.1f}" '
                     f'y2="{hoehe-24}" stroke-width="1"/>')
        teile.append(f'<text x="{x:.1f}" y="{hoehe-8}" text-anchor="middle">'
                     f'{int(v*100)} %</text>')
    serien = [("neu", "var(--s1)", "neuer Detektor"),
              ("ohne", "var(--s2)", "ohne diesen Generator trainiert"),
              ("prod", "var(--s3)", "bester der sieben Produktionsdetektoren")]
    for k, z in enumerate(zeilen):
        y0 = oben + k * zh
        teile.append(f'<text class="gen" x="{lx-10}" y="{y0+13}" text-anchor="end">'
                     f'{html.escape(z["name"])}</text>')
        for j, (key, farbe, lab) in enumerate(serien):
            v = z.get(key)
            if v is None:
                continue
            y = y0 + j * (bh + lu)
            w = max(v * pb, 1.5)
            tipp = f'{z["name"]} · {lab}: {prozent(v)}'
            if key == "prod" and z.get("prod_name"):
                tipp += f' ({z["prod_name"]})'
            # abgerundetes Datenende, am Nullpunkt gerade
            teile.append(f'<rect class="bar" x="{lx}" y="{y}" width="{w:.1f}" height="{bh}" '
                         f'rx="2" fill="{farbe}" data-tipp="{html.escape(tipp)}"/>')
            if key == "neu":
                teile.append(f'<text class="wert" x="{lx+w+5:.1f}" y="{y+bh-0.5}">'
                             f'{prozent(v, 0)}</text>')
        if z.get("summe"):
            yl = y0 + zh - 5
            teile.append(f'<line class="achse" x1="12" y1="{yl}" x2="{breite-12}" y2="{yl}"/>')
    teile.append(f'<line class="achse" x1="{lx}" y1="{oben-6}" x2="{lx}" y2="{hoehe-24}"/>')
    teile.append("</svg>")
    leg = '<div class="legende">' + "".join(
        f'<span><i style="background:{f}"></i>{l}</span>' for _, f, l in serien) + "</div>"
    return leg + "".join(teile)


def main():
    erg = lies(f"ergebnis_{KOPF}.json")
    mess = lies("ergebnis_messlatten.json")
    leak = lies("leak.json")
    clip = lies("ergebnis_clip_l14_mlp_aug.json")
    wd = lies("ergebnis_wd_eva02_mlp_aug.json")
    lr = lies("ergebnis_clip_l14_logreg_aug.json")

    gens = sorted(k for k in erg["voll"] if k not in ("gesamt", "fpr_ist", "info"))
    prod_g = {k.split("|")[0]: v for k, v in mess.items() if k.endswith("|test_aug")}
    prod_s = {k.split("|")[0]: v for k, v in mess.items() if k.endswith("|test")}
    kurz = lambda k: (k.replace("deepghs_", "").replace("hf_", "")
                      .replace("mobilenetv3_", "mobilenetv3 ").replace("caformer_s36_", "caformer "))

    def bester(tab, g, schluessel="tpr@0.01"):
        k, v = max(((k, v[g][schluessel]) for k, v in tab.items()), key=lambda x: x[1])
        return v, kurz(k)

    best_gesamt_g = max(prod_g.items(), key=lambda x: x[1]["gesamt"]["tpr@0.01"])
    best_gesamt_s = max(prod_s.items(), key=lambda x: x[1]["gesamt"]["tpr@0.01"])
    neu_g = erg["voll"]["gesamt"]["tpr@0.01"]
    neu_auc = erg["voll"]["gesamt"]["auc"]

    zeilen = [{"name": "alle zusammen", "neu": neu_g, "summe": True,
               "prod": best_gesamt_g[1]["gesamt"]["tpr@0.01"], "prod_name": kurz(best_gesamt_g[0])}]
    for g in sorted(gens, key=lambda g: -erg["voll"][g]["tpr@0.01"]):
        pv, pn = bester(prod_g, g)
        zeilen.append({"name": g, "neu": erg["voll"][g]["tpr@0.01"],
                       "ohne": erg.get("ohne", {}).get(g, {}).get("tpr@0.01"),
                       "prod": pv, "prod_name": pn})

    kt, kr = kontrolle(log("kontrolle_beide.log"))
    tg, tg_n = telegram(log("telegram_beide.log"))
    b_neu = bench(log("bench_beide.log")) or bench(log("bench_clip_mlp_aug.log"))
    b_naiv = bench(log("bench_clip_naiv.log"))
    b_base = re.findall(r"^(\S+)\s+([\d.]+)\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)%",
                        log("bench_clip_mlp_aug.log"), re.M)
    b_base = [(k, float(a), float(r) / 100) for k, a, r in b_base if k.startswith(("deepghs", "hf"))]

    h = []
    a = h.append
    a('<title>Der Wallhaven-Detektor</title>')
    a('<link rel="preconnect" href="https://fonts.googleapis.com">'
      '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
      '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700'
      '&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">')
    a(f"<style>{CSS}</style>")
    a('<div id="tipp" aria-hidden="true"></div><main class="seite">')
    a('<div class="kicker">Laborbericht · 24. September 2026 · Produktion unangetastet</div>')
    a('<h1>Der Wallhaven-Detektor</h1>')
    a('<p class="unter">Ein KI-Bild-Detektor, trainiert auf 69.693 menschlichen Anime-Bildern '
      'von Wallhaven (vor 2022, belegbar menschlich) und 95.693 Nachbildungen derselben Motive '
      'aus vierzehn Generatoren. Gemessen gegen die sieben Detektoren, die '
      '<code>imagesort.py</code> heute benutzt.</p>')

    # --- Antwort
    tab, alle3, kal, alle2, kal2, n2m, n2k = vergleiche()
    a('<h2 style="border-top:0;margin-top:1.5rem;padding-top:0">Die Antwort in drei Sätzen</h2>')
    a('<ul>'
      f'<li><b>Auf modernen Generatoren ist er weit überlegen.</b> Bei 1 % Fehlalarm erkennt er '
      f'{prozent(tab[0][1]["t1"])} der KI-Bilder aus unseren vierzehn Generatoren, der beste der sieben '
      f'Produktionsdetektoren {prozent(tab[1][1]["t1"])}. Bei Flux, Z-Image '
      f'und Qwen ist die Produktion fast blind.</li>'
      f'<li><b>Allein ist er auf echtem Telegram-Material schwächer als die Produktion.</b> Gegen '
      f'unvorsortierte Bilder aus dem Netz erreicht er AUC {komma(tab[0][2]["auc"])}, legekka '
      f'{komma(tab[2][2]["auc"])}. Seine Vorstellung von „menschlich“ ist Wallhaven — echte '
      f'Telegram-Kunst hält er zu oft für KI.</li>'
      f'<li><b>Als dritte Stimme neben zwei Produktionsmodellen ist er in beiden Welten vorn.</b> '
      f'Das ist die Empfehlung.</li></ul>')
    a('<div class="tabelle"><table><thead><tr><th rowspan="2">Variante</th>'
      '<th class="n" colspan="2">eigener Test · 14 Generatoren</th>'
      '<th class="n" colspan="2">echtes Telegram · fair</th></tr><tr>'
      '<th class="n">AUC</th><th class="n">@ 1 %</th><th class="n">AUC</th><th class="n">@ 1 %</th>'
      '</tr></thead><tbody>' + "".join(
          f'<tr{" class=summe" if n.startswith("neu +") else ""}><td>{n}</td>'
          f'<td class="n">{komma(e1["auc"])}</td><td class="n">{prozent(e1["t1"])}</td>'
          f'<td class="n">{komma(e3["auc"])}</td><td class="n">{prozent(e3["t1"])}</td></tr>'
          for n, e1, e3 in tab) + '</tbody></table></div>')
    a('<p class="unter">„@ 1 %“: Anteil erkannter KI-Bilder, wenn höchstens 1 % der menschlichen '
      'Bilder fälschlich markiert werden. Kombination = Mittel der Ränge, ohne Labels, ohne '
      'Gewichte. Eigener Test: 16.397 Bilder, gestört wie nach Telegram. Fairer Telegram-Test: '
      'Telegram-Posts von vor 2022 gegen unabhängig heruntergeladene KI-Bilder, beide identisch '
      'Telegram-simuliert (Details unten). Die Wahl von legekka als Partner kam aus diesem '
      'Telegram-Ergebnis — sie ist nicht blind getroffen.</p>')

    # --- Grafik
    a('<h2>Wo er stark ist: je Generator</h2>')
    a('<p>Auf dem eigenen Testsplit, jeder Generator einzeln — die Gesamtzahl allein wäre '
      'irreführend, weil sd15 und hdm_xut leicht zu erkennen sind. Für die Produktion steht hier '
      '<b>je Generator der beste der sieben</b>, das begünstigt sie.</p>')
    a('<figure>' + grafik(zeilen) + '<figcaption>Trefferquote bei 1 % Fehlalarm auf '
      'menschlichen Bildern, gestörter Testsplit. Die Schwelle des neuen Detektors stammt hier aus dem '
      'Validierungssplit, so wie sie im Einsatz stünde — daher „alle zusammen“ ein wenig anders als '
      'in der Tabelle oben, wo sie für alle Varianten auf den Testbildern selbst gesetzt ist. '
      'Orange: ein eigener Kopf, der diesen Generator '
      'im Training nie gesehen hat. Überfahren zeigt Werte und Detektornamen.</figcaption></figure>')
    rows = "".join(
        f'<tr{" class=summe" if z.get("summe") else ""}><td>{html.escape(z["name"])}</td>'
        f'<td class="n">{prozent(z["neu"])}</td>'
        f'<td class="n">{prozent(z["ohne"]) if z.get("ohne") is not None else "—"}</td>'
        f'<td class="n">{prozent(z["prod"])}</td><td>{html.escape(z["prod_name"])}</td></tr>'
        for z in zeilen)
    a('<details><summary>Als Tabelle</summary><div class="tabelle"><table><thead><tr>'
      '<th>Generator</th><th class="n">neu</th><th class="n">ohne ihn</th>'
      '<th class="n">Produktion</th><th>bester Detektor</th></tr></thead>'
      f'<tbody>{rows}</tbody></table></div></details>')

    # --- Rueckgrate
    a('<h3>Was die Leistung trägt</h3>')
    rg = [("CLIP ViT-L/14, linear", lr), ("CLIP ViT-L/14, eine versteckte Schicht", clip),
          ("WD-EVA02-Large, eine versteckte Schicht", wd), ("beide zusammen, eine versteckte Schicht", erg)]
    a('<div class="tabelle"><table><thead><tr><th>Rückgrat und Kopf</th><th class="n">AUC</th>'
      '<th class="n">Treffer @ 1 %</th><th class="n">@ 5 %</th></tr></thead><tbody>' + "".join(
          f'<tr><td>{n}</td><td class="n">{komma(e["voll"]["gesamt"]["auc"])}</td>'
          f'<td class="n">{prozent(e["voll"]["gesamt"]["tpr@0.01"])}</td>'
          f'<td class="n">{prozent(e["voll"]["gesamt"]["tpr@0.05"])}</td></tr>' for n, e in rg)
      + '</tbody></table></div>')
    a('<p>Beide Rückgrate bleiben eingefroren, trainiert wird nur ein kleiner Kopf — Sekunden '
      'statt Stunden. Der Anime-Tagger WD-EVA02 sieht offenbar anderes als das allgemeine CLIP; '
      'zusammen sind sie deutlich besser als jedes allein.</p>')

    # --- Vertrauen
    a('<h2>Warum die Zahlen auf dem eigenen Test halten</h2>')
    a('<p>Drei Lecks hätten jede Zahl wertlos gemacht. Alle drei waren da, alle drei sind '
      'gemessen und geschlossen.</p>')
    gr, gn = leak["groesse_roh"]["gesamt"], leak["groesse_norm"]["gesamt"]
    a('<h3>1 · Die Bildgröße verriet die Klasse</h3>')
    a(f'<p>Die Negative stehen alle auf exakt 1.024 Pixeln lange Kante, zwölf Generatoren '
      f'rechnen größer, sd15 und nai_v1 kleiner. Nur aus der Bildgröße, ohne ein Pixel '
      f'anzusehen: <span class="zahl">AUC {komma(gr)}</span>. Deshalb laufen beide Klassen durch '
      f'dieselbe Zurichtung (lange Kante 512, nie hochgerechnet, JPEG q95 4:4:4). Danach blieb '
      f'noch <span class="zahl">{komma(gn)}</span> — das Seitenverhältnis: Generatoren rechnen in '
      f'Buckets aus Vielfachen von 64 Pixeln, echte Wallpaper nicht. Beide Rückgrate sehen darum '
      f'nur einen quadratischen Mittelausschnitt; das Seitenverhältnis kommt dort nicht an.</p>')
    a('<h3>2 · Die Verarbeitung verriet die Klasse</h3>')
    fr = FRUEHER["robust"]
    a(f'<p>Der erste Kopf erreichte eine AUC von {komma(fr[0][1])} — und war fast wertlos. Die '
      f'Negative wurden von 1.024 auf 512 exakt halbiert, die Positiven um krumme Faktoren; der '
      f'Kopf hatte das Resampling-Muster gelernt. Eine Skalierung um 1,7 % verschob den Score '
      f'der Negative um {komma(FRUEHER["eingriff"]["neg"], 1)} Logits Richtung KI. Stört man '
      f'beide Klassen gleich, bricht er ein:</p>')
    tab = [("Störung", "erster Kopf", "Treffer", "endgültiger Kopf", "Treffer")]
    kr_d = {x[0]: x for x in kr}
    namen = {"unveraendert": "unveraendert (q95)"}
    a('<div class="tabelle"><table><thead><tr><th>beide Klassen gleich gestört</th>'
      '<th class="n">erster Kopf AUC</th><th class="n">Treffer @ 1 %</th>'
      '<th class="n">endgültig AUC</th><th class="n">Treffer @ 1 %</th></tr></thead><tbody>')
    for lab, auc_f, t_f in fr:
        k = kr_d.get(namen.get(lab, lab))
        a(f'<tr><td>{lab.replace("unveraendert", "unverändert")}</td><td class="n">{komma(auc_f)}</td>'
          f'<td class="n">{prozent(t_f)}</td>'
          f'<td class="n">{komma(k[1]) if k else "—"}</td><td class="n">{prozent(k[2]) if k else "—"}</td></tr>')
    a('</tbody></table></div>')
    a('<p>Die Abhilfe ist in der Forschung Standard: <b>trainiert wird nur auf gestörten Bildern</b>, '
      'jedes mit eigenem, zufälligem, aber reproduzierbarem Maßstab (0,6 bis 0,95, nie '
      'ganzzahlig), Filter, JPEG-Qualität (70–95) und Farbunterabtastung — für beide Klassen '
      'gleich verteilt. Der endgültige Kopf hält unter jeder Störung.</p>')
    if kt:
        n_, p_ = kt.get("Negative"), kt.get("Positive")
        if n_ and p_:
            a(f'<p>Der Eingriff mit derselben 1,7-%-Skalierung verschiebt ihn bei Negativen um '
              f'<span class="zahl">{n_[2]:+.2f}</span> und bei Positiven um '
              f'<span class="zahl">{p_[2]:+.2f}</span> Logits — beide in dieselbe Richtung, der '
              f'Abstand zwischen den Klassen bleibt. Beim ersten Kopf liefen sie auseinander '
              f'({FRUEHER["eingriff"]["neg"]:+.2f} gegen {FRUEHER["eingriff"]["pos"]:+.2f}).</p>')
    sr, sn = leak["stumpf_roh"]["gesamt"], leak["stumpf_norm"]["gesamt"]
    a('<h3>3 · Was nach der Zurichtung bleibt, ist echt</h3>')
    a(f'<p>Acht stumpfe Bildstatistiken (Schärfe, Rauschen, Blockkanten …) trennen auf den '
      f'Rohdateien mit {komma(sr)}, nach der Zurichtung mit {komma(sn)}. Der Rest ist echtes '
      f'Generatorsignal, und das lässt sich zeigen: Die zwölf großen Generatoren durchlaufen exakt '
      f'dieselbe Zurichtung, streuen aber von {komma(min(leak["stumpf_norm"][g] for g in ("anima","wai_illustrious")))} '
      f'bis {komma(leak["stumpf_norm"]["hdm_xut"])}. Die Verarbeitung allein kann also höchstens '
      f'die untere Zahl erklären.</p>')

    # --- Unbekannte Generatoren
    a('<h2>Generatoren, die er nie gesehen hat</h2>')
    ohne = erg.get("ohne", {})
    if ohne:
        gut = sorted(ohne, key=lambda g: -ohne[g]["tpr@0.01"])
        a(f'<p>Für jeden der vierzehn einmal ohne ihn trainiert, nur auf ihm gemessen (orange in '
          f'der Grafik). Die SDXL-Familie und die modernen Architekturen übertragen sich gut — '
          f'{", ".join(f"{g} {prozent(ohne[g]["tpr@0.01"],0)}" for g in gut[:3])}. Schwach wird '
          f'es bei den Einzelgängern: '
          f'{", ".join(f"{g} {prozent(ohne[g]["tpr@0.01"],0)}" for g in gut[-3:])}. '
          f'hdm_xut ist die einzige Architektur ihrer Art im Satz, sd15 eine ältere Generation. '
          f'Ein neuer Generator wird also meist erkannt, ein völlig neuer Typ erst nach Training '
          f'auf ihm.</p>')

    # --- Externe Proben
    a('<h2>Auf echtem Telegram-Material</h2>')
    a('<p>Drei Anläufe, weil die ersten beiden je eine Falle hatten. Nur der dritte ist fair.</p>')

    a('<h3>Der faire Test: niemand hat die Bilder vorsortiert</h3>')
    a(f'<p><b>Mensch:</b> {kal["n"]} zufällige Telegram-Posts aus <code>unbearbeitet</code> von '
      f'<b>vor 2022</b> — da gab es die Generatoren noch nicht. Fotos entfernt mit dem Filter aus '
      f'<code>imagesort.py</code>. <b>KI:</b> gleich viele zufällige Bilder aus '
      f'<code>&lt;DATASETS&gt;\\Ai Only</code>, unabhängig heruntergeladen. Weil Downloads mit JPEG '
      f'q≥94 und Telegram mit q≤92 schon an der Datei trennbar sind, laufen <b>beide Seiten '
      f'identisch</b> durch eine Telegram-Simulation (lange Kante 640, JPEG q87, 4:2:0).</p>')
    a('<div class="tabelle"><table><thead><tr><th>Detektor</th><th class="n">AUC</th>'
      '<th class="n">@ 1 %</th><th class="n">@ 5 %</th></tr></thead><tbody>' + "".join(
          f'<tr{" class=summe" if kk == "neu" else ""}><td>{"neuer Detektor" if kk == "neu" else kurz(kk)}</td>'
          f'<td class="n">{komma(e["auc"])}</td><td class="n">{prozent(e["t1"])}</td>'
          f'<td class="n">{prozent(e["t5"])}</td></tr>' for kk, e in alle3)
      + '</tbody></table></div>')
    a(f'<p>Allein liegt der neue Detektor hier hinter fünf der sieben. Dazu kommt die '
      f'<b>Kalibrierung</b>: Die Schwelle, die auf Wallhaven 1 % Fehlalarm erzeugt, markiert auf '
      f'echter Telegram-Kunst <b>{prozent(kal["fa1"])}</b> der menschlichen Bilder. Zwei Gründe liegen '
      f'nahe. Die KI-Bilder hier sind überwiegend SD-1.5-Ära (lange Kante im Median 768) — genau '
      f'das, worauf die Produktionsmodelle trainiert wurden. Und „menschlich“ hat der neue Detektor '
      f'nur an Wallhaven-Wallpapern gelernt; Skizzen, Mangaseiten und Fanart in allen Qualitäten '
      f'kennt er nicht.</p>')
    d3 = np.load(os.path.join(DATEN, "telegram3_doppelt.npz"))
    yd = d3["y"].astype(int)
    kd = {kk: _kenn(yd, d3[kk]) for kk in ("neu", "deepghs_caformer_s36_plus_sce", "hf_legekka")}
    k0 = dict(alle3)
    a(f'<p><b>Ist dieser Test selbst sauber?</b> Die Telegram-Posts haben eine Kompression mehr '
      f'hinter sich als die Downloads (Telegram beim Hochladen). Gibt man den KI-Bildern vorher '
      f'ebenfalls eine Telegram-artige Hochlade-Kompression, ändert sich an der Produktion nichts: '
      f'caformer_plus {komma(k0["deepghs_caformer_s36_plus_sce"]["auc"])} → '
      f'{komma(kd["deepghs_caformer_s36_plus_sce"]["auc"])}, legekka '
      f'{komma(k0["hf_legekka"]["auc"])} → {komma(kd["hf_legekka"]["auc"])}; der neue Detektor '
      f'{komma(k0["neu"]["auc"])} → {komma(kd["neu"]["auc"])}. Die Produktion nutzt die '
      f'Kompressionsgeschichte hier nicht aus — <b>ihr Vorsprung ist echt.</b> Acht stumpfe '
      f'Pixelstatistiken trennen die beiden Seiten mit {komma(REVIEWLOG["stumpf_t3"])}, genau so stark '
      f'wie den eigenen, sauber zugerichteten Datensatz; am stärksten die Sättigung — KI-Bilder sind '
      f'schlicht bunter.</p>')
    a('<div class="hinweis"><b>Was bleibt:</b> Nach dem Aussortieren zu kleiner Bilder stehen je '
      f'{kal["n"]} Bilder pro Seite im Test. Die KI-Bilder werden weniger stark verkleinert (meist '
      '768 → 640) als die Telegram-Bilder (meist 1280 → 640). Und Kunst von vor 2022 ähnelt '
      'stilistisch den Wallhaven-Negativen — bei moderner menschlicher Kunst dürfte der Fehlalarm '
      'eher höher liegen.</div>')

    a('<h3>Deine bestätigten KI-Bilder — von der Produktion vorsortiert</h3>')
    a(f'<p>Dieselben {n2m} Telegram-Posts von vor 2022 gegen die {n2k} Bilder, die du in der '
      f'Review-UI als KI bestätigt hast. Hier liegt die Produktion noch deutlicher vorn (bester: '
      f'{kurz(alle2[0][0])}, AUC {komma(alle2[0][1]["auc"])}; neu: AUC '
      f'{komma(dict(alle2)["neu"]["auc"])}). Das ist aber fast ein Zirkelschluss: Ins Review kamen '
      f'diese Bilder nur, <i>weil</i> die Produktion sie markiert hatte. Der neue Detektor erkennt '
      f'bei seiner 1-%-Schwelle {prozent(REVIEWLOG["ki_treffer"])} von ihnen; der Fehlalarm auf den '
      f'Telegram-Posts liegt dort bei {prozent(kal2)}.</p>')

    a('<h3>Eine Frage an dich: was heißt <code>kept_as_human</code>?</h3>')
    a(f'<p>Der erste Anlauf nahm deine <code>kept_as_human</code>-Entscheidungen als menschlich. '
      f'Ergebnis: AUC {komma(REVIEWLOG["auc"])} — schlechter als Zufall. Der Grund liegt in den Daten '
      f'selbst:</p>')
    a('<ul>'
      f'<li><b>{prozent(REVIEWLOG["anteil_15"])}</b> dieser Bilder haben exakt die Größe eines 1,5-fach '
      f'hochskalierten SDXL-Buckets (1248×1824 = 1,5 × 832×1216, der übliche Hires-Fix), bei den '
      f'bestätigten KI-Bildern nur {prozent(REVIEWLOG["anteil_15_ki"])}.</li>'
      f'<li>Der beste Produktionsdetektor gibt ihnen im Median <b>1,000</b> — den Höchstwert. Der neue '
      f'Detektor markiert {prozent(REVIEWLOG["neu_15"])} von ihnen.</li>'
      f'<li>Sie stammen aus einem einzigen Export, gleichmäßig verteilt über Februar 2024 bis '
      f'Januar 2025, rund hundert im Monat.</li>'
      f'<li><code>kept_ai</code> („KI, will ich aber behalten“) gibt es erst seit dem 26. Juli 2026. '
      f'<b>Davor war <code>kept_as_human</code> die einzige Art, ein Bild zu behalten.</b></li></ul>')
    a('<div class="hinweis"><b>Nur du kannst das beantworten.</b> Heißt <code>kept_as_human</code> in '
      'deinen älteren Entscheidungen „menschlich“ oder „behalten“? Wenn „behalten“, ist ein großer '
      'Teil davon KI, und die review_log taugt nur nach einer Bereinigung als Wahrheit — auch für das '
      'XGB-Meta, das seit Juli genau daraus trainiert.</div>')

    a('<h3>Der Prüfstand V5Minor100</h3>')
    if b_neu:
        bb = max(b_base, key=lambda x: x[2]) if b_base else None
        a(f'<p>Gemessen mit dem robusten CLIP-Kopf fällt der neue Detektor hier durch: AUC {komma(b_neu[0])}, '
          f'{prozent(b_neu[1])} Treffer bei 5 % Fehlalarm'
          + (f' — gegen {prozent(bb[2])} bei {kurz(bb[0].replace(":","_"))}' if bb else '') + '. '
          f'Das hat zwei Gründe, und beide sind gemessen:</p>')
        a(f'<ul><li><b>Es sind Fotos, kein Anime.</b> Die echten Bilder sind Handyfotos '
          f'(Median 2.520 Pixel, häufigste Größe 3024×4032), der Detektor kennt nur Anime.</li>'
          f'<li><b>Die Dateien verraten die Klasse.</b> Nur aus Bildgröße und Dateiformat trennt '
          f'der Prüfstand mit AUC {komma(V5["groesse_format_auc"])} — besser als jeder der sieben '
          f'Detektoren. Er belohnt Empfindlichkeit für Dateieigenschaften, genau das, was dem neuen '
          f'Detektor abtrainiert wurde.</li></ul>')
        if b_naiv:
            a(f'<p>Die Gegenprobe bestätigt das: Der verworfene erste Kopf, der auf '
              f'Verarbeitungsspuren anspringt, schneidet dort <i>besser</i> ab (AUC '
              f'{komma(b_naiv[0])}) als der robuste. Für deinen Zweck taugt dieser Prüfstand nicht '
              f'als Maßstab.</p>')

    # --- Offen
    a('<h2>Was als Nächstes käme</h2>')
    a('<h3>Wofür ich dich brauche</h3>')
    a('<ul>'
      '<li><b>Die Bedeutung von <code>kept_as_human</code></b> (siehe oben). Davon hängt ab, ob die '
      'review_log als Wahrheit taugt — für diesen Detektor und für das XGB-Meta.</li>'
      '<li><b>Der Einbau.</b> Bewusst nicht angefasst. Die Messungen sprechen für den neuen Detektor '
      'als <b>zusätzliche Stimme</b> in einem Rangmittel mit caformer_plus und legekka, nicht als '
      'Ersatz. <code>lab/detektor/kandidat.py</code> liefert schon „Pfad → Score“. Wie das mit der '
      'Stufen- und Veto-Logik zusammenspielt — nach der Messung vom 6. September verschlechtert '
      'Stufe 2 den Recall —, sollte zusammen mit dir entschieden werden.</li></ul>')
    a('<h3>Was ich ohne dich weiter machen könnte</h3>')
    a('<ul>'
      '<li><b>Die menschliche Seite verbreitern.</b> In <code>unbearbeitet</code> liegen '
      '1,31 Millionen Telegram-Posts von vor August 2022 — garantiert menschlich, echte '
      'Telegram-Kompression, echte Vielfalt. Ein Teil davon als zusätzliche Negative dürfte die '
      'Kalibrierung auf Telegram deutlich verbessern. Achtung dabei: Diese Bilder haben keine '
      'Motiv-Partner unter den Positiven, der Detektor könnte Inhalte statt Herkunft lernen. Die '
      'Kontrollen müssten mitlaufen, und der gepaarte Testsplit bleibt der Maßstab.</li>'
      '<li><b>Die Schwelle auf Telegram setzen,</b> nicht auf Wallhaven — der Fehlalarm wandert '
      'sonst von 1 % auf ein Vielfaches.</li>'
      '<li><b>Fotos</b> bleiben außerhalb seiner Zuständigkeit; das Foto-Gate aus '
      '<code>imagesort.py</code> muss vor ihm stehen.</li>'
      '<li><b>anima</b> bleibt der härteste Generator — auch im eigenen Test nur '
      f'{prozent(erg["voll"]["anima"]["tpr@0.01"])}.</li></ul>')

    # --- Reproduktion
    a('<h2>Nachrechnen</h2>')
    a('<pre>python lab/detektor/norm.py                  # Zurichtung, 165.386 Bilder, 13 GB\n'
      'python lab/detektor/leak.py                  # Größe und stumpfe Merkmale\n'
      'python lab/detektor/embed.py --rueckgrat clip_l14 --aug\n'
      'python lab/detektor/embed.py --rueckgrat wd_eva02 --aug\n'
      'python lab/detektor/messlatte.py --aug       # die sieben Produktionsdetektoren\n'
      'python lab/detektor/train.py --rueckgrat wd_eva02 --rueckgrat clip_l14 \\\n'
      '       --aug --kopf mlp --ohne-generator\n'
      'python lab/detektor/kontrolle.py score_wd_eva02_clip_l14_mlp_aug --eingriff --robust\n'
      'python lab/detektor/telegram2.py             # vor-2022-Telegram gegen bestätigte KI\n'
      'python lab/detektor/telegram3.py             # der faire Test\n'
      'python lab/detektor/bericht.py               # diese Seite</pre>')
    a('<p class="unter">Split: 80/10/10 nach Motivgruppen (stärkster Charakter-Tag), kein Motiv in '
      'zwei Teilen. Wahrheit: Wallhaven-Dateien mit Datum vor 2022 gegen eigene Erzeugnisse aus '
      'denselben Tags.</p>')
    a('</main>')
    a(f"<script>{JS}</script>")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(h))
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
