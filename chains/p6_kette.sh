#!/bin/sh
# p6_kette.sh — p5-Daten, aber die drei Midjourney-Quellen zaehlen bei der Gewichtung als EINE Quelle.
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
echo "$(date) start" > $L/p6_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p6_status.log
    "$@" > $L/p6_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p6_status.log; exit 1; }
}
schritt train    $PY train3.py --zusatz v3z2,v3z3,p2z,p3z,p4z,p5z --tag p6 --rating-ausgleich --kalib-p2 \
                     --quelle-gruppe "midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney"
schritt kalib    $PY kalibriere.py p6
schritt gleichfa $PY gleich_fa.py v3 p3x p4 p5 p6
schritt p2test   $PY p2_test.py v3 p3x p4 p5 p6
schritt mjver    $PY mj_versionen.py v3 p3x p4 p5 p6
schritt kontrolle $PY kontrolle3.py p6
schritt bench    $PY bench.py v3 p2 p3x p4 p5 p6
echo "$(date) FERTIG" >> $L/p6_status.log
