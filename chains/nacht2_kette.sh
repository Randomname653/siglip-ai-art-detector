#!/bin/sh
# nacht2_kette.sh — nach nacht_kette.sh: zweiter Satz Ausschlusstests mit SigLIP allein (praefix ausS),
# dazu die Kontrollen fuer den SigLIP-Einzelkopf. Frage: sind drei Rueckgrate bei NEUEN Generatoren
# robuster als SigLIP allein (im Standardtest gleichauf, aber halb so schnell)?
# Wartebedingung: eine Zeile, die auf FERTIG ENDET (die Startzeile der Nachtkette enthaelt das Wort auch).
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until grep -q -e "FERTIG$" -e "ABBRUCH" $L/nacht_status.log 2>/dev/null; do
    sleep 120
done
echo "$(date) start, nacht: $(tail -1 $L/nacht_status.log | tr -d '()')" > $L/nacht2_status.log
schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/nacht2_status.log
    "$@" > $L/nacht2_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/nacht2_status.log; exit 1; }
}
schritt kontr_S   $PY kontrolle3.py rg_S
schritt ausS_lauf $PY ausschluss.py lauf --praefix ausS --rueckgrate siglip_mid
schritt ausS_ausw $PY ausschluss.py auswerten --praefix ausS --rueckgrate siglip_mid
echo "$(date) FERTIG" >> $L/nacht2_status.log
