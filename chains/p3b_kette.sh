#!/bin/sh
# p3b_kette.sh — p3 mit anderen Seeds (100-104): wie viel vom Abstand p3/p2 ist Streuung?
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
echo "$(date) start" > $L/p3b_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p3b_status.log
    "$@" > $L/p3b_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p3b_status.log; exit 1; }
}
schritt train    $PY train3.py --zusatz v3z2,v3z3,p2z,p3z --tag p3b --rating-ausgleich --kalib-p2 --seed-basis 100
schritt kalib    $PY kalibriere.py p3b
schritt kanaele  $PY kanaele_test.py p3b
schritt gleichfa $PY gleich_fa.py v3 p2 p3 p3b
schritt p2test   $PY p2_test.py v3 p2 p3 p3b
schritt kontrolle $PY kontrolle3.py p3b
echo "$(date) FERTIG" >> $L/p3b_status.log
