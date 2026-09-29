#!/bin/sh
# v31_kette.sh — v3.1 jetzt (CivitAI-Aufstockung ist fertig), unabhaengig von Phase 2
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
echo "$(date) v3.1 startet" > $L/v31_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/v31_status.log
    "$@" > $L/v31_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/v31_status.log; exit 1; }
}
schritt pack   $PY aufbau3.py --stamm v3z3 --ohne v3z2
schritt tags   $PY tags_pack.py v3z3
schritt emb_c  $PY embed.py --rueckgrat clip_mid --pack v3z3 --aug
schritt emb_ct $PY embed.py --rueckgrat clip_mid --pack v3z3 --nur-split test
schritt emb_d  $PY embed.py --rueckgrat dino_mid --pack v3z3 --aug
schritt emb_dt $PY embed.py --rueckgrat dino_mid --pack v3z3 --nur-split test
schritt train  $PY train3.py --zusatz v3z2,v3z3 --tag v3_1 --rating-ausgleich
schritt kanaele  $PY kanaele_test.py v3_1
schritt danbooru $PY danbooru_test.py v3 v3_1
echo "$(date) FERTIG" >> $L/v31_status.log
