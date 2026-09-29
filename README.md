# Brushproof — AI art & illustration detector (SigLIP 2)

Detects AI-generated **anime, illustration and digital art** (not photographs). A small ensemble of MLP heads on the
intermediate layers of a frozen image model (SigLIP 2). Built and measured with a strict protocol: every
test image is from creators, artists or channels that never appeared in training.

| | this detector (`p8sx`) | best existing detector we compared |
|---|---:|---:|
| AI images caught, 18 real Telegram AI channels (never seen) | **97.7 %** | 47.1 % |
| AI images caught, generators removed from training entirely | **92–99 %** | 6–73 % |
| human art wrongly flagged (Danbooru 2026 · wallpapers · non-anime) | 1.1 % · 1.0 % · 0.9 % | — |
| in practice: 12,594 human images of a real Telegram art collection, genuine false positives on art | **0.46 %** | 3.0 %¹ |
| ROC AUC, mixed test set of 29,019 images | **0.998** | 0.887 |

Threshold: 1 % false positives on held-out human art. In practice about half of the raw flags on a real collection
are photos, memes and reposts, which are out of scope; see section 14 of [docs/RESULTS.md](docs/RESULTS.md) for the
breakdown and for what a stricter threshold costs. ¹ previous production system (seven detectors, majority vote),
same images, after its photo filter.

## How it works

```
image ──normalize──▶ SigLIP 2 so400m (frozen) ──▶ mean patch token of layers 7/12/17/22/27 (5 × 1152)
                                                          │
                                                   10 small MLP heads (512 hidden) ──▶ mean logit ──▶ threshold
```

- **Frozen backbone, intermediate layers.** Middle layers still carry *how* an image was made; the final layer
  mostly describes *what* it shows. SigLIP 2 alone matches CLIP + DINOv2 + SigLIP 2 combined at twice the speed
  ([backbone comparison](docs/RESULTS.md#4-which-backbone-carries-the-signal)).
- **Trained only on perturbed images** (random rescale, crop, JPEG — identical for both classes), so the heads
  cannot learn file size, resampling or compression instead of origin.
- **Diverse, current data on both sides.** ~344,000 AI images from dozens of generators (~30 CivitAI base models,
  AIBooru, DiffusionDB, Midjourney archives, 14 locally run models) against ~373,000 human images, including human art made *after*
  2022 (Danbooru, which bans AI images) and human non-anime art — otherwise a detector learns "new style = AI"
  or "non-anime = AI".
- **Honest calibration and comparison.** Thresholds are set on unperturbed validation art (three human groups
  weighted equally); detectors are compared at the same false-positive rate or by AUC.

What went wrong along the way and how it was caught: [docs/FINDINGS.md](docs/FINDINGS.md).

## Quick start

```bash
git clone https://github.com/Randomname653/siglip-ai-art-detector.git
cd siglip-ai-art-detector
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# download p8sx_heads.zip from the Releases page (github.com/Randomname653/siglip-ai-art-detector/releases)
# and unpack it into ./daten
set DETEKTOR_DATEN=./daten            # PowerShell: $env:DETEKTOR_DATEN="./daten"; bash: export DETEKTOR_DATEN=./daten
python pruefe.py "path/to/images" --kopf p8sx
```

The scanner is **read-only**: it never moves or changes files. It skips photos and images smaller than 512 px
and writes a CSV and an HTML contact sheet of the most suspicious images. SigLIP 2 is downloaded from Hugging Face
on first use.

## Repository layout

| | |
|---|---|
| `pruefe.py`, `kandidat.py` | scanner / scoring |
| `schutz.py`, `schutz_*.py` | child protection ([docs/SAFETY.md](docs/SAFETY.md)) |
| `civitai.py`, `aibooru.py`, `danbooru*.py`, `diffusiondb.py`, `midjourney.py`, `mj_v6.py`, `wallhaven_general.py`, `einsortieren_*.py`, `werbung.py` | data collection |
| `norm.py`, `lesen.py`, `aufbau*.py`, `tags_pack.py`, `embed.py` | normalization, packs, features |
| `train3.py`, `kalibriere.py` | training, calibration |
| `gleich_fa.py`, `bench.py`, `ausschluss.py`, `kontrolle3.py`, `p2_test.py`, `mj_versionen.py`, `vorversuch_*.py` | evaluation |
| `ergebnisse_md.py`, `bericht*.py`, `rg_bericht.py` | reports generated from result files |
| `chains/` | the exact shell pipelines of every run |
| `archive/` | earlier experiments (v1/v2), kept for traceability |
| `docs/` | results, findings, data sources, safety, reproduction; `docs/de/` original German lab notes |

Code comments and log messages are in German (the lab language); documentation is in English.
All machine-specific paths are in `pfade.py` and can be set by environment variables or a git-ignored `pfade_lokal.py`.

## Data

No images are included. Sources, licenses and filters: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).
Collection used official APIs, freely licensed datasets and public channels only; sites whose terms forbid
scraping were not used. How to rebuild everything: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Limitations

- Anime and digital art only; photographs are out of scope and filtered out.
- Hardest human cases are polished concept art, 3D renders and landscapes (2–3 % false positives in small
  subsets). Hardest AI cases are generators unlike anything in training: Midjourney drops to ~92 % when no
  Midjourney image is in training.
- The Telegram channels are real AI channels, but not every single post is guaranteed AI; hit rates there are
  lower bounds.
- New generators will appear; the leave-generator-out results are the best estimate of how the detector copes.

## License

[GNU AGPL-3.0](LICENSE). You may use, study, modify and share this code, also commercially — but anything you
distribute or offer as a network service that is based on it must be released under the same license, with
source code.
