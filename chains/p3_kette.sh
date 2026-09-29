#!/bin/sh
# p3_kette.sh — Phase 3: fruehe KI von 2022 (DiffusionDB) als Aufstockungs-Pack p3z, Kopf p3.
# Startet, wenn diffusiondb.py fertig ist. p2 bleibt Scanner-Standard, bis p3 ihn nachweislich
# schlaegt (gleich_fa.py, p2_test.py, Kontrollen).
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8

until grep -q "^FERTIG" $L/diffusiondb.log 2>/dev/null; do
    sleep 120
done
echo "$(date) DiffusionDB fertig, Phase 3 startet" > $L/p3k_status.log

schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p3k_status.log
    "$@" > $L/p3k_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p3k_status.log; exit 1; }
}
schritt dubletten $PY dubletten_kanaele.py diffusiondb
schritt pack     $PY aufbau3.py --p3 --stamm p3z
schritt tags     $PY tags_pack.py p3z
schritt emb_c    $PY embed.py --rueckgrat clip_mid --pack p3z --aug
schritt emb_ct   $PY embed.py --rueckgrat clip_mid --pack p3z --nur-split test
schritt emb_d    $PY embed.py --rueckgrat dino_mid --pack p3z --aug
schritt emb_dt   $PY embed.py --rueckgrat dino_mid --pack p3z --nur-split test
schritt train    $PY train3.py --zusatz v3z2,v3z3,p2z,p3z --tag p3 --rating-ausgleich --kalib-p2
schritt kalib    $PY kalibriere.py p3
schritt kanaele  $PY kanaele_test.py p3
schritt gleichfa $PY gleich_fa.py v3 p2 p3
schritt p2test   $PY p2_test.py v3 p2 p3
schritt danbooru $PY danbooru_test.py p2 p3
schritt kontrolle $PY kontrolle3.py p3
echo "$(date) FERTIG" >> $L/p3k_status.log
