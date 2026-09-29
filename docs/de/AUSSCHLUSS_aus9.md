# Ausschlusstests: unbekannte Generatoren

Stand 2026-09-28 18:37. Basiskopf **aus9_0** (fünf Köpfe, Seeds 0–4; Rückgrate: siglip_mid). Je Experiment wird der Kopf mit identischen Daten, Seeds und Kalibrierung neu trainiert — nur ohne die genannten Generatoren (Training und Validierung). Getestet wird auf dem Standard-Testsatz (`bench.py`: Testsplits je Generator, 15 Telegram-KI-Kanäle, drei menschliche Testgruppen). Schwelle je Kopf: 1 % Fehlalarm im Mittel der drei menschlichen val-Gruppen.

Zum Vergleich jeweils das beste Produktionsmodell (`caformer_s36_plus_sce`), das keinen der Generatoren im Training hatte.

| Exp. | entfernt | Bilder raus | Treffer vorher → ohne | AUC vorher → ohne | Produktion | übrige Gruppen Δ | Kanäle Δ | Fehlalarm |
|---|---|---|---|---|---|---|---|---|
| A | Nano Banana + OpenAI (geschlossen, 2025) | 14602 | 99.0% → **96.5%** | 0.9989 → 0.9974 | 6.2% | -0.1% | -0.4% | 0.9% |
| B | Seedream + Imagen4 (geschlossen) | 6496 | 98.8% → **98.6%** | 0.9981 → 0.9971 | 18.1% | +0.3% | +0.0% | 1.3% |
| C | Grok | 3593 | 91.8% → **91.1%** | 0.9932 → 0.9916 | 19.6% | +0.1% | +0.2% | 1.2% |
| D | Flux-Familie (Flux.1/Flux.2, Anime-Flux) | 30024 | 98.9% → **94.2%** | 0.9989 → 0.9957 | 27.7% | +0.0% | -0.2% | 1.1% |
| E | Illustrious-Familie (Illustrious, NoobAI, WAI) | 50902 | 98.5% → **97.1%** | 0.9989 → 0.9979 | 46.7% | -0.1% | -0.2% | 1.0% |
| F | NovelAI (v1, v2, AIBooru) | 18216 | 96.3% → **94.9%** | 0.9976 → 0.9968 | 52.6% | -0.2% | -0.2% | 1.1% |
| G | Pony (Pony, Pony V7) | 19474 | 99.2% → **98.8%** | 0.9992 → 0.9989 | 72.6% | -0.0% | +0.1% | 1.1% |
| H | Midjourney + Nijijourney | 33415 | 98.6% → **91.8%** | 0.9989 → 0.9936 | 23.9% | -0.0% | -1.1% | 1.2% |
| I | Stable Diffusion 1.x (SD 1.5, DiffusionDB 2022) | 33778 | 99.5% → **96.4%** | 0.9996 → 0.9979 | 41.8% | -0.2% | -0.3% | 0.9% |
| J | Qwen-Image (Qwen, Qwen 2) | 13812 | 99.5% → **99.2%** | 0.9995 → 0.9994 | 52.0% | -0.3% | -0.4% | 1.1% |
| K | Z-Image (Base, Turbo, Anime) | 17660 | 98.4% → **97.1%** | 0.9988 → 0.9983 | 27.1% | -0.0% | -0.1% | 1.2% |

Treffer: Anteil erkannter KI-Bilder der entfernten Generatoren (Mittel ihrer Testgruppen). „übrige Gruppen Δ“ und „Kanäle Δ“: mittlere Veränderung auf allem anderen — zeigt, ob das Entfernen auch Nachbarn schwächt. Streuung zwischen zwei Trainingsläufen mit anderen Seeds: bis ±2,6 Punkte je Kanal (p3 gegen p3b).

## Je Gruppe

### A: Nano Banana + OpenAI (geschlossen, 2025)

Entfernt: `Nano Banana`, `OpenAI`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Nano Banana | 99.3% | 98.3% | 0.9986 | 0.9974 | 5.3% |
| CivitAI OpenAI | 98.7% | 94.7% | 0.9992 | 0.9974 | 7.0% |

### B: Seedream + Imagen4 (geschlossen)

Entfernt: `Seedream`, `Imagen4`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Seedream | 99.7% | 99.3% | 0.9997 | 0.9996 | 25.3% |
| CivitAI Imagen4 | 98.0% | 98.0% | 0.9964 | 0.9946 | 10.9% |

### C: Grok

Entfernt: `Grok`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Grok | 91.8% | 91.1% | 0.9932 | 0.9916 | 19.6% |

### D: Flux-Familie (Flux.1/Flux.2, Anime-Flux)

Entfernt: `Flux.1 D`, `Flux.1 S`, `Flux.1 Krea`, `Flux.2 D`, `Flux.2 Klein 9B`, `Flux.2 Klein 4B`, `animepro_flux`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Flux.1 D | 98.7% | 99.0% | 0.9986 | 0.9988 | 37.0% |
| CivitAI Flux.1 S | 98.6% | 98.2% | 0.9978 | 0.9971 | 15.5% |
| CivitAI Flux.1 Krea | 100.0% | 100.0% | 1.0000 | 1.0000 | 52.1% |
| CivitAI Flux.2 D | 99.2% | 98.4% | 0.9987 | 0.9983 | 17.1% |
| CivitAI Flux.2 Klein 9B | 99.3% | 99.0% | 0.9998 | 0.9997 | 32.7% |
| eigen animepro_flux | 97.7% | 70.7% | 0.9984 | 0.9804 | 12.0% |

### E: Illustrious-Familie (Illustrious, NoobAI, WAI)

Entfernt: `Illustrious`, `NoobAI`, `illustrious`, `noobai`, `wai_illustrious`, `aibooru:illustrious`, `aibooru:noobai`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Illustrious | 100.0% | 99.7% | 0.9998 | 0.9998 | 67.7% |
| CivitAI NoobAI | 98.7% | 98.0% | 0.9986 | 0.9982 | 49.3% |
| eigen illustrious | 98.0% | 94.0% | 0.9982 | 0.9955 | 25.3% |
| eigen noobai | 96.0% | 94.0% | 0.9981 | 0.9963 | 21.0% |
| eigen wai_illustrious | 100.0% | 100.0% | 0.9999 | 0.9999 | 70.3% |

### F: NovelAI (v1, v2, AIBooru)

Entfernt: `nai_v1`, `nai_v2`, `aibooru:novelai`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| eigen nai_v1 | 100.0% | 100.0% | 0.9999 | 0.9998 | 77.7% |
| eigen nai_v2 | 97.7% | 93.3% | 0.9984 | 0.9966 | 19.3% |
| AIBooru NovelAI | 91.3% | 91.3% | 0.9946 | 0.9940 | 60.7% |

### G: Pony (Pony, Pony V7)

Entfernt: `Pony`, `Pony V7`, `pony`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Pony | 99.3% | 99.3% | 0.9992 | 0.9990 | 87.3% |
| CivitAI Pony V7 | 98.3% | 97.0% | 0.9986 | 0.9980 | 58.3% |
| eigen pony | 100.0% | 100.0% | 0.9998 | 0.9996 | 72.0% |

### H: Midjourney + Nijijourney

Entfernt: `Midjourney (Felix)`, `Nijijourney v5`, `Midjourney v6`, `aibooru:nijijourney`, `aibooru:midjourney`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| Midjourney (Felix) | 99.7% | 91.0% | 0.9991 | 0.9905 | 19.3% |
| Nijijourney v5 | 98.3% | 87.3% | 0.9992 | 0.9936 | 26.3% |
| Midjourney v6 | 99.3% | 84.7% | 0.9996 | 0.9867 | 1.7% |
| AIBooru Nijijourney | 98.7% | 97.0% | 0.9989 | 0.9982 | 63.7% |
| Midjourney channel A | 98.7% | 96.7% | 0.9993 | 0.9978 | 20.2% |
| Midjourney channel B | 96.8% | 93.8% | 0.9974 | 0.9950 | 12.0% |

### I: Stable Diffusion 1.x (SD 1.5, DiffusionDB 2022)

Entfernt: `SD 1.5`, `sd15`, `SD 1.x 2022 (DiffusionDB)`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI SD 1.5 | 99.7% | 99.3% | 0.9998 | 0.9998 | 58.7% |
| eigen sd15 | 100.0% | 92.7% | 0.9997 | 0.9967 | 57.7% |
| DiffusionDB SD 1.x 2022 | 100.0% | 97.3% | 1.0000 | 0.9986 | 18.0% |
| Mixed channel B | 98.5% | 96.2% | 0.9990 | 0.9963 | 32.7% |

### J: Qwen-Image (Qwen, Qwen 2)

Entfernt: `Qwen`, `Qwen 2`, `Qwen 2.1`, `qwen_image`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Qwen | 99.0% | 98.3% | 0.9991 | 0.9988 | 41.0% |
| eigen qwen_image | 100.0% | 100.0% | 0.9999 | 1.0000 | 63.0% |

### K: Z-Image (Base, Turbo, Anime)

Entfernt: `ZImageBase`, `ZImageTurbo`, `z_anime`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI ZImageBase | 97.3% | 95.3% | 0.9980 | 0.9975 | 35.3% |
| CivitAI ZImageTurbo | 100.0% | 100.0% | 0.9999 | 0.9999 | 32.4% |
| eigen z_anime | 98.0% | 96.0% | 0.9985 | 0.9975 | 13.7% |

