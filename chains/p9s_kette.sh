#!/bin/sh
# p9s_kette.sh — p8sx plus everyday-edit augmentation (norm.stoere2): crops, text/watermarks, noise, colour,
# greyscale, blur/sharpen, upscale round trip — equally for both classes. Same data, same head recipe.
# Reason: robust_test.py (28.09.) — p8sx false positives on humans rose to 2.4-4.2 % under crop/noise/text/upscaling.
# Compared against p8sx on channels (gleich_fa), human groups (p2_test), pro art and the robustness matrix.
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
echo "$(date) start" > $L/p9s_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p9s_status.log
    "$@" > $L/p9s_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p9s_status.log; exit 1; }
}
for p in v2 v3z2 v3z3 p2z p3z p4z p5z; do
    [ -f $L/emb_siglip_mid_aug2_$p.npy ] || schritt emb_$p $PY embed.py --rueckgrat siglip_mid --pack $p --aug2
done
T="$PY train3.py --zusatz v3z2,v3z3,p2z,p3z,p4z,p5z --rating-ausgleich --kalib-p2 --aug-name aug2 \
   --rueckgrate siglip_mid --quelle-gruppe midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney"
schritt train   $T --tag p9s
schritt train_b $T --tag p9sb --seed-basis 100
for i in 0 1 2 3 4; do cp $L/kopf_p9s_s$i.npz $L/kopf_p9sx_s$i.npz; cp $L/kopf_p9sb_s$i.npz $L/kopf_p9sx_s$((i+5)).npz; done
schritt kalib    $PY kalibriere.py p9sx
schritt gleichfa $PY gleich_fa.py v3 p8x p8sx p9sx --json gleich_fa_p9s.json
schritt robust   $PY robust_test.py --kopf p9sx
schritt profi    $PY profikunst_test.py --koepfe p9sx,p8sx
schritt profid   $PY profikunst_datum.py --koepfe p9sx,p8sx
schritt p2test   $PY p2_test.py v3 p6x p8x p8sx p9sx
schritt kontr    $PY kontrolle3.py p9sx
schritt klein    $PY klein_test.py --kopf p9sx
echo "$(date) ENDE ok" >> $L/p9s_status.log
