#!/bin/sh
# p6b_kette.sh — p6 mit anderen Seeds (100-104); danach p6x = p6 + p6b (zehn Koepfe), kalibriert.
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
echo "$(date) start" > $L/p6b_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p6b_status.log
    "$@" > $L/p6b_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p6b_status.log; exit 1; }
}
schritt train $PY train3.py --zusatz v3z2,v3z3,p2z,p3z,p4z,p5z --tag p6b --rating-ausgleich --kalib-p2 \
                  --seed-basis 100 --quelle-gruppe "midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney"
for i in 0 1 2 3 4; do cp $L/kopf_p6_s$i.npz $L/kopf_p6x_s$i.npz; cp $L/kopf_p6b_s$i.npz $L/kopf_p6x_s$((i+5)).npz; done
schritt kalib $PY kalibriere.py p6x
schritt gleichfa $PY gleich_fa.py v3 p3x p6 p6x
echo "$(date) FERTIG" >> $L/p6b_status.log
