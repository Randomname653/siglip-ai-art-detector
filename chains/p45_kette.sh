#!/bin/sh
# p45_kette.sh — Midjourney ins Training.
#   p4 = p3 + p4z (Felix-Archiv CC0 + niji-v5 CC0, je ein Ersteller)
#   p5 = p4 + p5z (MJ-v6-Stichprobe aus Scrape-Datensatz, viele Ersteller)
# Getrennt trainiert, damit sichtbar wird, was die Stichprobe mit vielen Erstellern zusaetzlich bringt.
# Startet, wenn midjourney.py und mj_v6.py fertig sind. Die Telegram-MJ-Kanaele bleiben Test;
# echte Dubletten zu ihnen (32x32-Pixelabgleich) fliegen vorher aus den Quellen.
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8

until grep -q "^FERTIG" $L/midjourney.log 2>/dev/null && grep -q "^FERTIG" $L/mj_v6.log 2>/dev/null; do
    sleep 120
done
echo "$(date) Downloads fertig" > $L/p45_status.log

schritt() {
    name=$1; shift
    echo "$(date) $name" >> $L/p45_status.log
    "$@" > $L/p45_$name.log 2>&1 || { echo "$(date) ABBRUCH in $name" >> $L/p45_status.log; exit 1; }
}
schritt dub_felix $PY embed_dubletten.py midjourney_felix --ausschliessen
schritt dub_niji  $PY embed_dubletten.py nijijourney_p1atdev --ausschliessen
schritt dub_v6    $PY embed_dubletten.py midjourney_v6_scrape --ausschliessen
schritt pack4    $PY aufbau3.py --p4 --stamm p4z
schritt pack5    $PY aufbau3.py --p5 --stamm p5z
schritt tags4    $PY tags_pack.py p4z
schritt tags5    $PY tags_pack.py p5z
for P in p4z p5z; do
    schritt emb_c_$P  $PY embed.py --rueckgrat clip_mid --pack $P --aug
    schritt emb_ct_$P $PY embed.py --rueckgrat clip_mid --pack $P --nur-split test
    schritt emb_d_$P  $PY embed.py --rueckgrat dino_mid --pack $P --aug
    schritt emb_dt_$P $PY embed.py --rueckgrat dino_mid --pack $P --nur-split test
done
schritt train4   $PY train3.py --zusatz v3z2,v3z3,p2z,p3z,p4z --tag p4 --rating-ausgleich --kalib-p2
schritt kalib4   $PY kalibriere.py p4
schritt train5   $PY train3.py --zusatz v3z2,v3z3,p2z,p3z,p4z,p5z --tag p5 --rating-ausgleich --kalib-p2
schritt kalib5   $PY kalibriere.py p5
schritt gleichfa $PY gleich_fa.py v3 p2 p3 p3x p4 p5
schritt p2test   $PY p2_test.py v3 p2 p3x p4 p5
schritt mjver    $PY mj_versionen.py v3 p2 p3x p4 p5
schritt kontr4   $PY kontrolle3.py p4
schritt kontr5   $PY kontrolle3.py p5
schritt bench    $PY bench.py v3 p2 p3 p3x p4 p5
echo "$(date) FERTIG" >> $L/p45_status.log
