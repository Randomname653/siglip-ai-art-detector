#!/bin/sh
# fortsetzen.sh — nach dem Neustart am 2026-09-27 alles wieder anwerfen (alles setzt dort auf, wo es stand):
#   midjourney.py  Felix-Archiv ab shard-00004 (niji fertig; .fertig-Marker + SHA-256 im Manifest)
#   mj_v6.py       MJ-v6-Stichprobe ab train_070 (4 von 8 Dateien fertig)
#   bench.py       Produktionsvergleich (mobilenetv3_sce_dist_v1 zwischengespeichert)
#   p45_kette.sh   wartet auf beide Downloads, dann p4/p5 + Tests + bench mit allen Koepfen
cd "$(dirname "$0")/.." || exit 1
L=${DETEKTOR_DATEN:-daten}
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
$PY midjourney.py --nur felix >> $L/midjourney.log 2>&1 && echo "FERTIG: felix" >> $L/midjourney.log &
$PY mj_v6.py >> $L/mj_v6.log 2>&1 &
$PY bench.py v3 p2 p3 p3x > $L/bench.log 2>&1 &
sh p45_kette.sh &
wait
