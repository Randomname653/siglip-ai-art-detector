#!/bin/sh
# nacht_kette.sh — Nacht 27./28.09. Startet, wenn rg_kette.sh (Rueckgrat-Vergleich) fertig ist.
#   1. Schutz: Figurenregel per WD-Figurentag auch fuer AIBooru (Training) und alle Telegram-Kanaele (Test)
#   2. Merkmale fuer die neuen KI-Kanaele (Anime F, Midjourney C, Mixed A)
#   3. Ausschlusstests (ausschluss.py): Basis aus_0 + 11 Experimente, drei Rueckgrate -> AUSSCHLUSS.md
#   4. Endgueltiger Kopf auf dem endgueltigen Datenstand: p8 + p8b = p8x, alle Tests + Produktionsvergleich
# Jeder Schritt bricht die Kette bei Fehler ab (Status in daten/nacht_status.log).
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until grep -q "FERTIG\|ABBRUCH" $L/rg_status.log 2>/dev/null; do sleep 120; done
echo "$(date) start (rg: $(tail -1 $L/rg_status.log))" > $L/nacht_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/nacht_status.log
    "$@" > $L/nacht_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/nacht_status.log; exit 1; }
}
KAN=$(ls -d ${DETEKTOR_DATASETS:-datasets}/telegram_ki/*/ | xargs -n1 basename | tr '\n' ',' | sed 's/,$//')
schritt schutz   $PY schutz_wd_figuren.py --nur aibooru,$KAN
schritt kan_emb  $PY kanaele_test.py p6
schritt kan_sig  $PY kanal_siglip.py
schritt aus_lauf $PY ausschluss.py lauf --basis aus_0
schritt aus_ausw $PY ausschluss.py auswerten --basis aus_0
T="$PY train3.py --zusatz v3z2,v3z3,p2z,p3z,p4z,p5z --rating-ausgleich --kalib-p2 \
   --rueckgrate clip_mid,dino_mid,siglip_mid --quelle-gruppe midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney"
schritt p8   $T --tag p8
schritt p8b  $T --tag p8b --seed-basis 100
for i in 0 1 2 3 4; do cp $L/kopf_p8_s$i.npz $L/kopf_p8x_s$i.npz; cp $L/kopf_p8b_s$i.npz $L/kopf_p8x_s$((i+5)).npz; done
schritt kalib    $PY kalibriere.py p8x
schritt gleichfa $PY gleich_fa.py v3 p6x p7x p8x --json gleich_fa_p8.json
schritt p2test   $PY p2_test.py v3 p6x p7x p8x
schritt mjver    $PY mj_versionen.py v3 p6x p7x p8x
schritt kontr    $PY kontrolle3.py p8x
schritt bench    $PY bench.py v3 p2 p6x p7x p8x --json bench_p8.json
echo "$(date) FERTIG" >> $L/nacht_status.log
