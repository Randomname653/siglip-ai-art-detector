#!/bin/sh
# aus9_kette.sh — leave-generator-out with the p9 everyday-edit augmentation (compare with ausS = p8s recipe).
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
A="--praefix aus9 --rueckgrate siglip_mid --aug-name aug2"
echo "$(date) start" > $L/aus9_status.log
$PY ausschluss.py lauf $A >> $L/aus9_status.log 2>&1 || { echo "$(date) ABBRUCH lauf" >> $L/aus9_status.log; exit 1; }
$PY ausschluss.py auswerten $A >> $L/aus9_status.log 2>&1 || { echo "$(date) ABBRUCH auswerten" >> $L/aus9_status.log; exit 1; }
echo "$(date) ENDE ok" >> $L/aus9_status.log
