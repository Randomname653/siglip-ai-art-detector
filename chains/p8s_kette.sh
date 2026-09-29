#!/bin/sh
# p8s_kette.sh — schlanker Endkopf: nur SigLIP-2-Mitte, zehn Koepfe (p8s + p8sb = p8sx), endgueltiger Datenstand.
# Anlass: Rueckgrat-Vergleich und Ausschlusstests (28.09.) — SigLIP allein ist gleich gut wie alle drei,
# bei doppeltem Scanner-Tempo. Vergleich direkt gegen p8x (drei Rueckgrate).
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
echo "$(date) start" > $L/p8s_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p8s_status.log
    "$@" > $L/p8s_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p8s_status.log; exit 1; }
}
T="$PY train3.py --zusatz v3z2,v3z3,p2z,p3z,p4z,p5z --rating-ausgleich --kalib-p2 \
   --rueckgrate siglip_mid --quelle-gruppe midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney"
schritt train   $T --tag p8s
schritt train_b $T --tag p8sb --seed-basis 100
for i in 0 1 2 3 4; do cp $L/kopf_p8s_s$i.npz $L/kopf_p8sx_s$i.npz; cp $L/kopf_p8sb_s$i.npz $L/kopf_p8sx_s$((i+5)).npz; done
schritt kalib    $PY kalibriere.py p8sx
schritt gleichfa $PY gleich_fa.py v3 p6x p8x p8sx --json gleich_fa_p8s.json
schritt p2test   $PY p2_test.py v3 p6x p8x p8sx
schritt mjver    $PY mj_versionen.py v3 p6x p8x p8sx
schritt kontr    $PY kontrolle3.py p8sx
schritt bench    $PY bench.py v3 p6x p8x p8sx --json bench_p8s.json
echo "$(date) ENDE ok" >> $L/p8s_status.log
