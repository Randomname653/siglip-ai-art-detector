#!/bin/sh
# rg_kette.sh — Rueckgrat-Vergleich: einzeln, paarweise, alle drei (p6 = CLIP+DINO und p7 = alle drei
# gibt es schon). Startet nach p7_kette.sh. Ergebnis: RUECKGRATE.md.
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until grep -q "FERTIG\|ABBRUCH" $L/p7_status.log 2>/dev/null; do sleep 120; done
grep -q ABBRUCH $L/p7_status.log && { echo "$(date) p7 abgebrochen" > $L/rg_status.log; exit 1; }
echo "$(date) start" > $L/rg_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/rg_status.log
    "$@" > $L/rg_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/rg_status.log; exit 1; }
}
T="$PY train3.py --zusatz v3z2,v3z3,p2z,p3z,p4z,p5z --rating-ausgleich --kalib-p2 \
   --quelle-gruppe midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney"
for K in C:clip_mid D:dino_mid S:siglip_mid CD:clip_mid,dino_mid CS:clip_mid,siglip_mid DS:dino_mid,siglip_mid; do
    tag=rg_${K%%:*}; rg=${K#*:}
    schritt train_$tag $T --tag $tag --rueckgrate $rg
    schritt kalib_$tag $PY kalibriere.py $tag
done
schritt tempo    $PY rg_tempo.py
schritt gleichfa $PY gleich_fa.py rg_C rg_D rg_S rg_CD rg_CS rg_DS p7 --json gleich_fa_rg.json
schritt bench    $PY bench.py rg_C rg_D rg_S rg_CD rg_CS rg_DS p7 --json bench_rg.json
schritt bericht  $PY rg_bericht.py
echo "$(date) FERTIG" >> $L/rg_status.log
