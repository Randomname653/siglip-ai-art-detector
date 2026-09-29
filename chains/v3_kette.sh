#!/bin/sh
# v3_kette.sh — der endgueltige v3-Lauf, startet selbst, wenn danbooru_train.py fertig ist.
#   1 Zusatzpack v3z2 (CivitAI + AIBooru + Danbooru-Training)
#   2 WD-Rating + Schutz-Sperre (tags_pack.py)
#   3 Einbetten: CLIP-Mitte + DINO-Mitte, gestoert (Training) und ungestoert (Test)
#   4 Fuenf Koepfe, Schwelle auf Danbooru-val 2026 (train3.py)
#   5 Tests: alle KI-Kanaele, Danbooru nach Rating
# Jeder Schritt schreibt daten/v3k_<schritt>.log; bricht einer ab, halten alle.
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8

until grep -q "FERTIG" $L/danbooru_train.log 2>/dev/null; do sleep 120; done
echo "$(date) Danbooru-Training fertig, Kette startet" > $L/v3k_status.log

schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/v3k_status.log
    "$@" > $L/v3k_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/v3k_status.log; exit 1; }
}
schritt pack     $PY aufbau3.py --stamm v3z2
schritt tags     $PY tags_pack.py v3z2
schritt emb_c    $PY embed.py --rueckgrat clip_mid --pack v3z2 --aug
schritt emb_ct   $PY embed.py --rueckgrat clip_mid --pack v3z2 --nur-split test
schritt emb_d    $PY embed.py --rueckgrat dino_mid --pack v3z2 --aug
schritt emb_dt   $PY embed.py --rueckgrat dino_mid --pack v3z2 --nur-split test
schritt train    $PY train3.py --zusatz v3z2 --tag v3 --rating-ausgleich
schritt kanaele  $PY kanaele_test.py v3
schritt danbooru $PY danbooru_test.py v3 v3r
echo "$(date) FERTIG" >> $L/v3k_status.log
