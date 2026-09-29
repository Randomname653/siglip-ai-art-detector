#!/bin/sh
# ausschluss_kette.sh — startet die Ausschlusstests (ausschluss.py), sobald p45_kette.sh fertig ist.
# Basis: p5 (alle Daten). 11 Experimente a ~18 min, danach Auswertung -> AUSSCHLUSS.md.
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until grep -q "FERTIG\|ABBRUCH" $L/p45_status.log 2>/dev/null; do sleep 120; done
grep -q ABBRUCH $L/p45_status.log && { echo "$(date) p45 abgebrochen, keine Ausschlusstests" > $L/ausschluss_status.log; exit 1; }
echo "$(date) start" > $L/ausschluss_status.log
$PY ausschluss.py lauf --basis p5 --zusatz v3z2,v3z3,p2z,p3z,p4z,p5z >> $L/ausschluss_status.log 2>&1 \
  && $PY ausschluss.py auswerten --basis p5 >> $L/ausschluss_status.log 2>&1
echo "$(date) FERTIG" >> $L/ausschluss_status.log
