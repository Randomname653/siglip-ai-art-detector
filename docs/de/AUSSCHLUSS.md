# Ausschlusstests: unbekannte Generatoren

Stand 2026-09-28 04:57. Basiskopf **aus_0** (fünf Köpfe, Seeds 0–4; Rückgrate: clip_mid,dino_mid,siglip_mid). Je Experiment wird der Kopf mit identischen Daten, Seeds und Kalibrierung neu trainiert — nur ohne die genannten Generatoren (Training und Validierung). Getestet wird auf dem Standard-Testsatz (`bench.py`: Testsplits je Generator, 15 Telegram-KI-Kanäle, drei menschliche Testgruppen). Schwelle je Kopf: 1 % Fehlalarm im Mittel der drei menschlichen val-Gruppen.

Zum Vergleich jeweils das beste Produktionsmodell (`caformer_s36_plus_sce`), das keinen der Generatoren im Training hatte.

| Exp. | entfernt | Bilder raus | Treffer vorher → ohne | AUC vorher → ohne | Produktion | übrige Gruppen Δ | Kanäle Δ | Fehlalarm |
|---|---|---|---|---|---|---|---|---|
| A | Nano Banana + OpenAI (geschlossen, 2025) | 14602 | 98.7% → **95.3%** | 0.9988 → 0.9969 | 6.2% | +0.3% | +0.0% | 1.0% |
| B | Seedream + Imagen4 (geschlossen) | 6496 | 99.2% → **99.2%** | 0.9994 → 0.9992 | 18.1% | +0.6% | +0.5% | 1.1% |
| C | Grok | 3593 | 96.1% → **94.0%** | 0.9970 → 0.9962 | 19.6% | +0.1% | +0.2% | 1.1% |
| D | Flux-Familie (Flux.1/Flux.2, Anime-Flux) | 30024 | 98.5% → **95.2%** | 0.9992 → 0.9973 | 27.7% | +0.5% | +0.4% | 1.1% |
| E | Illustrious-Familie (Illustrious, NoobAI, WAI) | 50902 | 99.2% → **97.4%** | 0.9992 → 0.9985 | 46.7% | +0.3% | +0.2% | 1.1% |
| F | NovelAI (v1, v2, AIBooru) | 18216 | 95.3% → **93.1%** | 0.9975 → 0.9959 | 52.6% | -0.0% | +0.2% | 0.9% |
| G | Pony (Pony, Pony V7) | 19474 | 99.3% → **98.6%** | 0.9994 → 0.9989 | 72.6% | +0.4% | +0.7% | 1.0% |
| H | Midjourney + Nijijourney | 33415 | 98.5% → **93.7%** | 0.9992 → 0.9962 | 23.9% | +0.6% | -0.8% | 1.2% |
| I | Stable Diffusion 1.x (SD 1.5, DiffusionDB 2022) | 33778 | 98.6% → **96.2%** | 0.9986 → 0.9957 | 41.8% | +0.6% | +0.1% | 1.2% |
| J | Qwen-Image (Qwen, Qwen 2) | 13812 | 99.2% → **99.2%** | 0.9997 → 0.9996 | 52.0% | +0.2% | +0.4% | 1.2% |
| K | Z-Image (Base, Turbo, Anime) | 17660 | 97.4% → **96.7%** | 0.9989 → 0.9980 | 27.1% | +0.1% | +0.0% | 1.0% |

Treffer: Anteil erkannter KI-Bilder der entfernten Generatoren (Mittel ihrer Testgruppen). „übrige Gruppen Δ“ und „Kanäle Δ“: mittlere Veränderung auf allem anderen — zeigt, ob das Entfernen auch Nachbarn schwächt. Streuung zwischen zwei Trainingsläufen mit anderen Seeds: bis ±2,6 Punkte je Kanal (p3 gegen p3b).

## Je Gruppe

### A: Nano Banana + OpenAI (geschlossen, 2025)

Entfernt: `Nano Banana`, `OpenAI`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Nano Banana | 99.0% | 96.3% | 0.9986 | 0.9974 | 5.3% |
| CivitAI OpenAI | 98.3% | 94.3% | 0.9990 | 0.9965 | 7.0% |

### B: Seedream + Imagen4 (geschlossen)

Entfernt: `Seedream`, `Imagen4`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Seedream | 99.7% | 99.7% | 0.9998 | 0.9998 | 25.3% |
| CivitAI Imagen4 | 98.6% | 98.6% | 0.9989 | 0.9987 | 10.9% |

### C: Grok

Entfernt: `Grok`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Grok | 96.1% | 94.0% | 0.9970 | 0.9962 | 19.6% |

### D: Flux-Familie (Flux.1/Flux.2, Anime-Flux)

Entfernt: `Flux.1 D`, `Flux.1 S`, `Flux.1 Krea`, `Flux.2 D`, `Flux.2 Klein 9B`, `Flux.2 Klein 4B`, `animepro_flux`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Flux.1 D | 99.0% | 99.7% | 0.9993 | 0.9993 | 37.0% |
| CivitAI Flux.1 S | 99.3% | 98.6% | 0.9995 | 0.9992 | 15.5% |
| CivitAI Flux.1 Krea | 99.0% | 100.0% | 0.9999 | 0.9999 | 52.1% |
| CivitAI Flux.2 D | 96.9% | 96.1% | 0.9982 | 0.9980 | 17.1% |
| CivitAI Flux.2 Klein 9B | 99.7% | 99.7% | 0.9998 | 0.9997 | 32.7% |
| eigen animepro_flux | 97.0% | 77.0% | 0.9984 | 0.9876 | 12.0% |

### E: Illustrious-Familie (Illustrious, NoobAI, WAI)

Entfernt: `Illustrious`, `NoobAI`, `illustrious`, `noobai`, `wai_illustrious`, `aibooru:illustrious`, `aibooru:noobai`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Illustrious | 99.7% | 99.3% | 0.9998 | 0.9996 | 67.7% |
| CivitAI NoobAI | 99.7% | 98.0% | 0.9994 | 0.9990 | 49.3% |
| eigen illustrious | 98.0% | 94.7% | 0.9988 | 0.9973 | 25.3% |
| eigen noobai | 98.7% | 95.0% | 0.9983 | 0.9966 | 21.0% |
| eigen wai_illustrious | 100.0% | 100.0% | 0.9999 | 0.9999 | 70.3% |

### F: NovelAI (v1, v2, AIBooru)

Entfernt: `nai_v1`, `nai_v2`, `aibooru:novelai`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| eigen nai_v1 | 100.0% | 100.0% | 0.9999 | 0.9996 | 77.7% |
| eigen nai_v2 | 96.3% | 88.7% | 0.9978 | 0.9939 | 19.3% |
| AIBooru NovelAI | 89.7% | 90.7% | 0.9948 | 0.9942 | 60.7% |

### G: Pony (Pony, Pony V7)

Entfernt: `Pony`, `Pony V7`, `pony`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Pony | 99.3% | 99.7% | 0.9996 | 0.9996 | 87.3% |
| CivitAI Pony V7 | 98.7% | 96.3% | 0.9988 | 0.9977 | 58.3% |
| eigen pony | 100.0% | 99.7% | 0.9997 | 0.9995 | 72.0% |

### H: Midjourney + Nijijourney

Entfernt: `Midjourney (Felix)`, `Nijijourney v5`, `Midjourney v6`, `aibooru:nijijourney`, `aibooru:midjourney`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| Midjourney (Felix) | 99.7% | 94.0% | 0.9997 | 0.9957 | 19.3% |
| Nijijourney v5 | 98.7% | 91.3% | 0.9992 | 0.9947 | 26.3% |
| Midjourney v6 | 100.0% | 89.3% | 0.9999 | 0.9951 | 1.7% |
| AIBooru Nijijourney | 96.3% | 95.7% | 0.9984 | 0.9974 | 63.7% |
| Midjourney channel A | 98.8% | 97.5% | 0.9993 | 0.9983 | 20.2% |
| Midjourney channel B | 97.7% | 94.7% | 0.9985 | 0.9957 | 12.0% |

### I: Stable Diffusion 1.x (SD 1.5, DiffusionDB 2022)

Entfernt: `SD 1.5`, `sd15`, `SD 1.x 2022 (DiffusionDB)`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI SD 1.5 | 99.7% | 99.7% | 0.9999 | 0.9998 | 58.7% |
| eigen sd15 | 100.0% | 95.0% | 0.9998 | 0.9967 | 57.7% |
| DiffusionDB SD 1.x 2022 | 100.0% | 99.0% | 1.0000 | 0.9992 | 18.0% |
| Mixed channel B | 94.7% | 91.2% | 0.9946 | 0.9871 | 32.7% |

### J: Qwen-Image (Qwen, Qwen 2)

Entfernt: `Qwen`, `Qwen 2`, `Qwen 2.1`, `qwen_image`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI Qwen | 98.3% | 98.3% | 0.9994 | 0.9993 | 41.0% |
| eigen qwen_image | 100.0% | 100.0% | 1.0000 | 0.9999 | 63.0% |

### K: Z-Image (Base, Turbo, Anime)

Entfernt: `ZImageBase`, `ZImageTurbo`, `z_anime`

| Testgruppe | Treffer vorher | ohne | AUC vorher | ohne | Produktion |
|---|---|---|---|---|---|
| CivitAI ZImageBase | 95.3% | 94.3% | 0.9987 | 0.9974 | 35.3% |
| CivitAI ZImageTurbo | 100.0% | 100.0% | 0.9999 | 0.9999 | 32.4% |
| eigen z_anime | 97.0% | 95.7% | 0.9979 | 0.9967 | 13.7% |

