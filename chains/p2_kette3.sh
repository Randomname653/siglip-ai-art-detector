#!/bin/sh
# p2_kette3.sh — Phase 2 (universell). Ersetzt p2_kette.sh: v3.1 laeuft vorher separat
# (v31_kette.sh), hier nur noch das Phase-2-Pack und der universelle Kopf.
# Startet, wenn Wallhaven general, Danbooru Nicht-Anime UND v3.1 fertig sind.
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8

until grep -q "FERTIG" $L/wallhaven_general.log 2>/dev/null && grep -q "FERTIG" $L/danbooru_nichtanime.log 2>/dev/null \
      && grep -q "FERTIG\|ABBRUCH" $L/v31_status.log 2>/dev/null; do
    sleep 300
done
echo "$(date) Downloads + v3.1 fertig, Phase 2 startet" > $L/p2k_status.log

schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p2k_status.log
    "$@" > $L/p2k_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p2k_status.log; exit 1; }
}
schritt schutz   $PY schutz_nachtrag.py --nur danbooru_nichtanime
schritt pack     $PY aufbau3.py --p2 --stamm p2z
schritt tags     $PY tags_pack.py p2z
schritt emb_c    $PY embed.py --rueckgrat clip_mid --pack p2z --aug
schritt emb_ct   $PY embed.py --rueckgrat clip_mid --pack p2z --nur-split test
schritt emb_d    $PY embed.py --rueckgrat dino_mid --pack p2z --aug
schritt emb_dt   $PY embed.py --rueckgrat dino_mid --pack p2z --nur-split test
schritt train    $PY train3.py --zusatz v3z2,v3z3,p2z --tag p2 --rating-ausgleich --kalib-p2
schritt kanaele  $PY kanaele_test.py p2
schritt danbooru $PY danbooru_test.py v3 v3_1 p2
schritt p2test   $PY p2_test.py v3 v3_1 p2
schritt kontrolle $PY kontrolle3.py p2
# v3 sauber neu (ohne die nachtraeglich gesperrten Bilder), damit kein genutzter Kopf sie enthaelt
schritt train_v3s $PY train3.py --zusatz v3z2,v3z3 --tag v3s --rating-ausgleich
schritt kanaele_v3s $PY kanaele_test.py v3s
schritt danbooru2 $PY danbooru_test.py v3s p2
echo "$(date) FERTIG" >> $L/p2k_status.log
