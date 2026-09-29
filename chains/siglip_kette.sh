#!/bin/sh
# siglip_kette.sh — drittes Rueckgrat (SigLIP-2-Mitte) fuer alles einbetten, was Training, Kalibrierung
# und Tests brauchen. Reihenfolge: erst Test/Kalibrierung (klein), dann die grossen gestoerten Packs.
# Fertige Teile (Datei emb_siglip_mid_<...>.npy vorhanden) werden uebersprungen — neu startbar.
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
echo "$(date) start" >> $L/siglip_status.log
schritt() {
    name=$1; ziel=$2; shift 2
    if [ -n "$ziel" ] && [ -f "$L/$ziel" ]; then echo "$(date) $name schon da" >> $L/siglip_status.log; return; fi
    echo "$(date) $name" >> $L/siglip_status.log
    "$@" > $L/siglip_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/siglip_status.log; exit 1; }
}
E="$PY embed.py --rueckgrat siglip_mid"
schritt v3d     emb_siglip_mid_v3d_ok.npy       $E --pack v3d
schritt p2z_val emb_siglip_mid_p2z_val_ok.npy   $E --pack p2z --nur-split val
for P in v2 v3z2 v3z3 p2z p3z p4z p5z; do
    schritt ${P}_test emb_siglip_mid_${P}_test_ok.npy $E --pack $P --nur-split test
done
schritt kanaele "" $PY kanal_siglip.py
for P in p5z p4z p3z v3z3 p2z v3z2 v2; do
    schritt ${P}_aug emb_siglip_mid_aug_${P}_ok.npy $E --pack $P --aug
done
echo "$(date) FERTIG" >> $L/siglip_status.log
