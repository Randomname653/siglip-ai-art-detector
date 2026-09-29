#!/bin/sh
# p2_kette.sh — startet selbst, wenn ALLE Phase-2-Downloads fertig sind:
#   CivitAI-Aufstockung (civitai2.log), CivitAI-Stil (civitai_stil.log),
#   Wallhaven general, Danbooru Nicht-Anime.
#   1 v3z3: nur neue CivitAI-Bilder (ohne v3z2)      -> Tags, Einbetten
#   2 p2z:  civitai_stil + danbooru_nichtanime + wallhaven_general -> Tags, Einbetten
#   3 v3_1: v2 + v3z2 + v3z3 (Anime, aufgestockt), Schwelle Danbooru 2026
#   4 p2:   v2 + v3z2 + v3z3 + p2z (universell), Schwelle Danbooru 2026 + Nicht-Anime-val
#   5 Tests: Kanaele, Danbooru, Phase-2-Test, Kontrollen fuer p2
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8

until grep -q "FERTIG" $L/civitai2.log 2>/dev/null && grep -q "FERTIG" $L/civitai_stil.log 2>/dev/null \
      && grep -q "FERTIG" $L/wallhaven_general.log 2>/dev/null && grep -q "FERTIG" $L/danbooru_nichtanime.log 2>/dev/null; do
    sleep 300
done
echo "$(date) alle Downloads fertig, Kette startet" > $L/p2k_status.log

schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p2k_status.log
    "$@" > $L/p2k_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p2k_status.log; exit 1; }
}
schritt pack_v3z3  $PY aufbau3.py --stamm v3z3 --ohne v3z2
schritt tags_v3z3  $PY tags_pack.py v3z3
schritt pack_p2z   $PY aufbau3.py --p2 --stamm p2z
schritt tags_p2z   $PY tags_pack.py p2z
for st in v3z3 p2z; do
    for r in clip_mid dino_mid; do
        schritt emb_${st}_${r}   $PY embed.py --rueckgrat $r --pack $st --aug
        schritt embt_${st}_${r}  $PY embed.py --rueckgrat $r --pack $st --nur-split test
    done
done
schritt train_v3_1 $PY train3.py --zusatz v3z2,v3z3 --tag v3_1 --rating-ausgleich
schritt train_p2   $PY train3.py --zusatz v3z2,v3z3,p2z --tag p2 --rating-ausgleich --kalib-p2
schritt kanaele    $PY kanaele_test.py p2
schritt kanaele31  $PY kanaele_test.py v3_1
schritt danbooru   $PY danbooru_test.py v3 v3_1 p2
schritt p2test     $PY p2_test.py v3 v3_1 p2
schritt kontrolle  $PY kontrolle3.py p2
echo "$(date) FERTIG" >> $L/p2k_status.log
