# Rückgrat-Vergleich

Welche eingefrorenen Bildmodelle („Rückgrate“) tragen wie viel zur Erkennung bei? Sieben Köpfe, identisch trainiert (gleiche Daten, Gewichtung, Seeds 0–4, Kalibrierung); nur die Rückgrate unterscheiden sich. Merkmale je Rückgrat: fünf Zwischenschichten (CLIP ViT-L/14: Blöcke 7/11/15/19/23, CLS; DINOv2-L: Schichten 8/12/16/20/24, CLS; SigLIP 2 so400m/384: Schichten 7/12/17/22/27, Mittel der Patch-Token).

Kanäle: 15 Telegram-KI-Kanäle, nie im Training, Schwelle je Kopf bei 1 % Fehlalarm im Mittel der drei menschlichen Testgruppen. Generatoren: Testsplits je Generator aus `bench.py` (Schwelle 1 % auf den val-Menschen). Tempo: Bilder/s im Scanner auf einer RTX 4090 (nur Merkmale, ohne Dateilesen).

| Rückgrate | Kanäle Treffer | Anime-Kanäle | Kanäle AUC | Generatoren Treffer | Generatoren AUC | Fehlalarm Test | Tempo |
|---|---|---|---|---|---|---|---|
| CLIP | 87.7% | 84.6% | 0.9902 | 90.1% | 0.9932 | 1.1% | 284/s |
| DINOv2 | 91.2% | 89.4% | 0.9941 | 88.9% | 0.9939 | 0.8% | 288/s |
| SigLIP 2 | 97.0% | 97.1% | 0.9974 | 98.5% | 0.9987 | 1.5% | 135/s |
| CLIP + DINOv2 | 93.2% | 92.2% | 0.9949 | 94.4% | 0.9959 | 1.4% | 143/s |
| CLIP + SigLIP 2 | 96.8% | 96.6% | 0.9973 | 98.2% | 0.9987 | 1.0% | 91/s |
| DINOv2 + SigLIP 2 | 96.9% | 96.7% | 0.9972 | 98.3% | 0.9986 | 1.3% | 92/s |
| CLIP + DINOv2 + SigLIP 2 | 97.0% | 96.9% | 0.9971 | 98.1% | 0.9987 | 1.0% | 69/s |

## Je Kanal (Treffer bei gleichem Fehlalarm)

| Kanal | CLIP | DINOv2 | SigLIP 2 | CLIP + DINOv2 | CLIP + SigLIP 2 | DINOv2 + SigLIP 2 | CLIP + DINOv2 + SigLIP 2 |
|---|---|---|---|---|---|---|---|
| Mixed channel C | 88.1% | 86.9% | 94.4% | 91.1% | 94.9% | 95.0% | 95.2% |
| Mixed channel B | 79.2% | 95.3% | 98.2% | 91.1% | 93.2% | 98.1% | 95.3% |
| Mixed channel D | 92.2% | 93.0% | 98.6% | 96.2% | 98.8% | 98.6% | 98.7% |
| Anime channel D | 91.6% | 97.7% | 99.7% | 97.5% | 99.4% | 99.6% | 99.3% |
| Anime channel A (AIBooru reposts) | 74.0% | 71.2% | 91.6% | 80.9% | 90.7% | 90.3% | 91.7% |
| Anime channel E | 89.0% | 94.5% | 97.4% | 95.4% | 97.1% | 97.4% | 97.7% |
| Anime channel C (ChatGPT/Grok) | 95.9% | 99.5% | 99.8% | 99.8% | 99.9% | 99.9% | 99.9% |
| Mixed channel G | 85.5% | 87.1% | 96.0% | 90.6% | 95.3% | 95.7% | 96.0% |
| Leonardo channel | 90.9% | 92.8% | 96.7% | 93.8% | 97.6% | 97.4% | 97.2% |
| Midjourney channel A | 93.9% | 94.3% | 98.4% | 96.3% | 98.5% | 98.7% | 98.7% |
| Midjourney channel B | 91.1% | 86.4% | 96.4% | 92.9% | 96.7% | 96.7% | 96.9% |
| Mixed channel F | 95.7% | 93.4% | 98.3% | 96.5% | 98.9% | 98.5% | 98.8% |
| Stable Diffusion channel | 97.9% | 98.7% | 99.4% | 98.8% | 99.4% | 99.3% | 99.3% |
| Anime channel B | 72.5% | 84.2% | 96.7% | 87.2% | 96.1% | 96.1% | 95.8% |
| Mixed channel E | 77.8% | 92.5% | 93.1% | 89.5% | 95.1% | 92.7% | 94.7% |

## Je Generator (Treffer, bench.py)

| Generator | CLIP | DINOv2 | SigLIP 2 | CLIP + DINOv2 | CLIP + SigLIP 2 | DINOv2 + SigLIP 2 | CLIP + DINOv2 + SigLIP 2 |
|---|---|---|---|---|---|---|---|
| CivitAI halbrealistisch/3D | 98.0% | 98.0% | 99.7% | 100.0% | 99.7% | 99.7% | 99.7% |
| CivitAI Anima | 85.7% | 80.3% | 96.0% | 88.7% | 95.3% | 95.7% | 94.7% |
| CivitAI Chroma | 84.2% | 81.6% | 91.5% | 85.7% | 93.4% | 91.9% | 94.5% |
| CivitAI Flux.1 D | 95.7% | 95.0% | 99.0% | 97.0% | 99.3% | 99.0% | 99.0% |
| CivitAI Flux.1 Krea | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| CivitAI Flux.1 S | 96.4% | 91.7% | 98.6% | 97.5% | 98.6% | 98.9% | 98.6% |
| CivitAI Flux.2 D | 89.1% | 88.4% | 99.2% | 91.5% | 98.4% | 98.4% | 97.7% |
| CivitAI Flux.2 Klein 9B | 97.3% | 97.7% | 99.3% | 99.3% | 99.3% | 99.3% | 99.7% |
| CivitAI Grok | 91.8% | 88.3% | 94.0% | 94.0% | 95.0% | 93.6% | 96.8% |
| CivitAI HiDream | 95.9% | 95.0% | 100.0% | 97.7% | 99.1% | 100.0% | 99.1% |
| CivitAI Illustrious | 95.7% | 96.7% | 100.0% | 98.0% | 100.0% | 99.3% | 99.7% |
| CivitAI Imagen4 | 93.2% | 84.4% | 98.0% | 94.6% | 98.0% | 98.0% | 98.6% |
| CivitAI Krea 2 | 90.3% | 92.7% | 98.0% | 95.3% | 98.3% | 98.0% | 98.3% |
| CivitAI Lumina | 72.0% | 75.8% | 90.9% | 79.5% | 81.8% | 91.7% | 85.6% |
| CivitAI Nano Banana | 90.0% | 85.3% | 99.3% | 92.0% | 98.7% | 99.0% | 99.0% |
| CivitAI NoobAI | 86.7% | 90.0% | 99.0% | 95.0% | 99.7% | 98.3% | 100.0% |
| CivitAI OpenAI | 93.0% | 89.0% | 99.0% | 96.3% | 98.0% | 98.3% | 98.7% |
| CivitAI Pony | 97.3% | 97.0% | 99.7% | 99.0% | 99.3% | 99.0% | 99.3% |
| CivitAI Pony V7 | 87.7% | 86.3% | 98.3% | 93.3% | 97.3% | 99.3% | 99.0% |
| CivitAI Qwen | 97.3% | 95.3% | 98.7% | 97.7% | 99.0% | 98.3% | 98.7% |
| CivitAI SD 1.5 | 96.3% | 95.0% | 100.0% | 98.3% | 99.7% | 99.7% | 99.7% |
| CivitAI SD 3.5 | 99.3% | 98.6% | 99.3% | 99.3% | 99.3% | 99.3% | 99.3% |
| CivitAI SDXL 1.0 | 99.3% | 99.3% | 99.7% | 99.7% | 99.7% | 99.7% | 99.7% |
| CivitAI Seedream | 94.0% | 96.7% | 99.3% | 98.7% | 100.0% | 100.0% | 99.7% |
| CivitAI ZImageBase | 89.0% | 88.0% | 97.0% | 92.3% | 96.7% | 97.3% | 95.3% |
| CivitAI ZImageTurbo | 93.7% | 97.1% | 100.0% | 99.2% | 100.0% | 100.0% | 100.0% |
| AIBooru NovelAI | 77.7% | 68.3% | 94.3% | 86.0% | 94.0% | 91.7% | 91.3% |
| AIBooru Nijijourney | 86.3% | 85.3% | 99.0% | 91.3% | 98.3% | 98.0% | 97.3% |
| AIBooru DALL-E | 90.9% | 88.6% | 97.7% | 95.5% | 97.7% | 97.7% | 97.7% |
| AIBooru andere/unbekannt | 86.7% | 84.3% | 98.0% | 91.7% | 97.3% | 98.0% | 97.0% |
| eigen anima | 53.0% | 46.7% | 95.0% | 65.0% | 92.0% | 92.0% | 89.7% |
| eigen animagine | 98.7% | 99.3% | 100.0% | 99.7% | 100.0% | 100.0% | 100.0% |
| eigen animepro_flux | 77.3% | 74.0% | 98.3% | 86.3% | 98.3% | 97.3% | 97.0% |
| eigen hdm_xut | 97.0% | 97.3% | 99.7% | 99.7% | 100.0% | 99.7% | 100.0% |
| eigen illustrious | 85.0% | 78.7% | 99.3% | 94.0% | 99.3% | 98.7% | 99.3% |
| eigen nai_v1 | 93.3% | 96.7% | 100.0% | 98.0% | 100.0% | 100.0% | 100.0% |
| eigen nai_v2 | 63.3% | 56.0% | 99.0% | 84.7% | 97.7% | 99.0% | 97.0% |
| eigen noobai | 70.3% | 68.7% | 98.7% | 85.0% | 98.7% | 98.0% | 98.3% |
| eigen pony | 94.7% | 97.0% | 100.0% | 99.3% | 100.0% | 100.0% | 100.0% |
| eigen qwen_image | 96.7% | 97.0% | 100.0% | 98.7% | 100.0% | 100.0% | 100.0% |
| eigen sd15 | 91.0% | 93.7% | 100.0% | 97.7% | 100.0% | 100.0% | 100.0% |
| eigen sdxl_base | 89.3% | 95.0% | 99.0% | 95.0% | 99.7% | 99.3% | 99.0% |
| eigen wai_illustrious | 96.7% | 99.0% | 100.0% | 99.7% | 100.0% | 100.0% | 100.0% |
| eigen z_anime | 79.3% | 77.0% | 98.7% | 89.7% | 97.7% | 97.7% | 96.3% |
| DiffusionDB SD 1.x 2022 | 99.3% | 97.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Midjourney (Felix) | 97.3% | 97.0% | 99.7% | 99.0% | 99.7% | 99.7% | 99.7% |
| Nijijourney v5 | 90.0% | 89.7% | 99.3% | 95.0% | 99.3% | 99.0% | 99.3% |
| Midjourney v6 | 99.7% | 98.0% | 99.0% | 100.0% | 100.0% | 98.7% | 100.0% |
