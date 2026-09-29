r"""wachbleiben.py — haelt Windows wach, bis daten/wach.stop existiert (hoechstens 14 h).

Die Detektor-Skripte (embed.py, train*.py, ...) setzen selbst keine
Schlafsperre — nur generate.py hat eine (seit dem 2026-09-10, nach zwei
Bluescreens beim Einschlafen mitten in der Rechnung). Statt sie in jedes
Skript zu bauen, laeuft fuer lange Ketten dieser eine Waechter daneben.

ES_SYSTEM_REQUIRED haelt nur das System wach, nicht die Bildschirme — die
duerfen ausgehen.

    start /b python lab/detektor/wachbleiben.py
    echo. > lab/detektor/daten/wach.stop      # beenden
"""
import ctypes
import os
import time

STOP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daten", "wach.stop")
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

if os.path.exists(STOP):
    os.remove(STOP)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
t0 = time.time()
try:
    while not os.path.exists(STOP) and time.time() - t0 < 14 * 3600:
        time.sleep(30)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
