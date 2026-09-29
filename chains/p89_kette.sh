#!/bin/sh
# p89_kette.sh — 20-head combination of p8sx (scale+JPEG augmentation) and p9sx (everyday-edit augmentation).
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
echo "$(date) start" > $L/p89_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p89_status.log
    "$@" > $L/p89_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p89_status.log; exit 1; }
}
schritt kalib    $PY kalibriere.py p89sx
schritt gleichfa $PY gleich_fa.py p8sx p9sx p89sx --json gleich_fa_p89.json
schritt robust   $PY robust_test.py --kopf p89sx
echo "$(date) ENDE ok" >> $L/p89_status.log
