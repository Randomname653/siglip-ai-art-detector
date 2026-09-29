#!/bin/sh
# p7_kette.sh — p6-Konfiguration mit drittem Rueckgrat (SigLIP-2-Mitte), zehn Koepfe (p7 + p7b = p7x).
# Startet, wenn siglip_kette.sh fertig ist. Vergleich gegen p6x auf allem, was es gibt.
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
# wartet ausserdem auf schutz_wd_figuren.py (Figurenregel auf WD-Figurentags), damit p7 auf bereinigten Daten lernt
until grep -q "FERTIG\|ABBRUCH" $L/siglip_status.log 2>/dev/null && grep -q "^FERTIG" $L/schutz_wd_figuren.log 2>/dev/null; do
    sleep 120
done
grep -q ABBRUCH $L/siglip_status.log && { echo "$(date) SigLIP abgebrochen" > $L/p7_status.log; exit 1; }
echo "$(date) start" > $L/p7_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p7_status.log
    "$@" > $L/p7_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p7_status.log; exit 1; }
}
T="$PY train3.py --zusatz v3z2,v3z3,p2z,p3z,p4z,p5z --rating-ausgleich --kalib-p2 \
   --rueckgrate clip_mid,dino_mid,siglip_mid --quelle-gruppe midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney"
schritt train   $T --tag p7
schritt train_b $T --tag p7b --seed-basis 100
for i in 0 1 2 3 4; do cp $L/kopf_p7_s$i.npz $L/kopf_p7x_s$i.npz; cp $L/kopf_p7b_s$i.npz $L/kopf_p7x_s$((i+5)).npz; done
schritt kalib   $PY kalibriere.py p7
schritt kalib_x $PY kalibriere.py p7x
schritt gleichfa $PY gleich_fa.py v3 p3x p6x p7 p7x
schritt p2test  $PY p2_test.py v3 p6x p7x
schritt mjver   $PY mj_versionen.py v3 p6x p7x
schritt kontrolle $PY kontrolle3.py p7x
schritt bench   $PY bench.py v3 p2 p3x p6x p7x
echo "$(date) FERTIG" >> $L/p7_status.log
