#!/bin/sh
# p10a_kette.sh — "aligned" training against false positives (research 29.09.): human training images passed
# through four generator VAEs (SD1, SDXL, Flux, Qwen) become extra AI examples next to their untouched originals.
# Also a variant with hard-negative weighting (p10h) on the p8s data only, for comparison.
# Compared with p8sx on: channels at equal FA (gleich_fa), all human test groups + FA curve (fa_kurve),
# the owner's own folder "Sorted Stuff" (ordner_bewerten), robustness matrix.
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
echo "$(date) start" > $L/p10a_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p10a_status.log
    "$@" > $L/p10a_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p10a_status.log; exit 1; }
}
until grep -q "^FERTIG" $L/ordner_sorted.log 2>/dev/null; do sleep 60; done
schritt sorted_p8sx $PY ordner_bewerten.py --name sorted --koepfe p8sx
Z="v3z2,v3z3,p2z,p3z,p4z,p5z"
G="midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney"
T="$PY train3.py --rating-ausgleich --kalib-p2 --rueckgrate siglip_mid --quelle-gruppe $G"
# harte Negative (nur Gewichtung, gleiche Daten) — laeuft vor dem langen VAE-Schritt
schritt train_h  $T --zusatz $Z --tag p10h --hart 3:0.02
schritt train_hb $T --zusatz $Z --tag p10hb --seed-basis 100 --hart 3:0.02
for i in 0 1 2 3 4; do cp $L/kopf_p10h_s$i.npz $L/kopf_p10hx_s$i.npz; cp $L/kopf_p10hb_s$i.npz $L/kopf_p10hx_s$((i+5)).npz; done
schritt kalib_h  $PY kalibriere.py p10hx
# VAE-Rekonstruktionen
schritt rekon    $PY vae_rekon.py --je-vae 12000
schritt pack6    $PY aufbau3.py --p6 --stamm p6z
schritt tags6    $PY tags_pack.py p6z
schritt emb6     $PY embed.py --rueckgrat siglip_mid --pack p6z --aug
schritt emb6t    $PY embed.py --rueckgrat siglip_mid --pack p6z --nur-split test
schritt train_a  $T --zusatz $Z,p6z --tag p10a
schritt train_ab $T --zusatz $Z,p6z --tag p10ab --seed-basis 100
for i in 0 1 2 3 4; do cp $L/kopf_p10a_s$i.npz $L/kopf_p10ax_s$i.npz; cp $L/kopf_p10ab_s$i.npz $L/kopf_p10ax_s$((i+5)).npz; done
schritt kalib_a  $PY kalibriere.py p10ax
schritt gleichfa $PY gleich_fa.py v3 p8sx p10hx p10ax --json gleich_fa_p10.json
schritt kurve_8  $PY fa_kurve.py --kopf p8sx
schritt kurve_h  $PY fa_kurve.py --kopf p10hx
schritt kurve_a  $PY fa_kurve.py --kopf p10ax
schritt sorted   $PY ordner_bewerten.py --name sorted --koepfe p8sx,p10hx,p10ax
schritt robust_a $PY robust_test.py --kopf p10ax --nur jpeg50,social,text,noise,upscale_cdc
echo "$(date) ENDE ok" >> $L/p10a_status.log
