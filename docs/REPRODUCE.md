# Reproducing the detector

The full pipeline as it was run for the published heads. Paths come from `pfade.py` and can be overridden with
environment variables (`DETEKTOR_DATEN`, `DETEKTOR_DATASETS`, `DETEKTOR_NAS`, `DETEKTOR_SCHLUESSEL`) or once in a
git-ignored `pfade_lokal.py`. The pipelines in `chains/` run with `PY=<your python> sh chains/<name>.sh`. Scripts are
run from the repository folder; comments and log messages in the code are German.

The exact sequences of each run are kept in [`chains/`](../chains) (shell scripts with the original paths); the
steps below are the distilled version for the final head `p8sx`.

## 0. Setup

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Disk: sources ~1 TB were budgeted; the data actually used is ~150 GB of images plus ~150 GB of packs and features.
GPU: an RTX 4090 (24 GB) was used; SigLIP 2 so400m feature extraction runs at ~100–135 images/s.

## 1. Collect sources (each writes `DATASETS/<source>/manifest.jsonl`)

| script | source |
|---|---|
| `civitai.py`, `civitai_stil.py` | CivitAI API (key via `CIVITAI_API_KEY` or `<DETEKTOR_SCHLUESSEL>/civitai_key.txt`) |
| `aibooru.py` | AIBooru API, ratings g/s |
| `danbooru_train.py`, `danbooru.py`, `danbooru_nichtanime.py` | Danbooru API (human art 2022-2026; polite rate limit) |
| `wallhaven_general.py` | local wallpaper archive, files dated before 2022 |
| `diffusiondb.py`, `midjourney.py`, `mj_v6.py` | Hugging Face datasets (see DATA_SOURCES.md) |
| `einsortieren_json.py`, `werbung.py`, `kanal_einordnen.py` | Telegram channel exports (JSON), ad removal, AI-vs-human triage |

All downloaders apply `schutz.py` (child protection) before storing anything. After extending the rules, run
`schutz_nachtrag.py` / `schutz_wd_figuren.py` to apply them retroactively. Check training sources against the
test channels with `embed_dubletten.py <source> --ausschliessen`.

## 2. Packs, tags, features

```bash
python aufbau2.py            # v2 pack (own generators, Wallhaven, Telegram)      -> daten/v2.*
python aufbau3.py            # web AI + Danbooru train                           -> --stamm v3z2
python aufbau3.py --mensch   # Danbooru test/calibration                         -> --stamm v3d
python aufbau3.py --p2  --stamm p2z   # non-anime humans + semi-realistic AI
python aufbau3.py --p3  --stamm p3z   # DiffusionDB
python aufbau3.py --p4  --stamm p4z   # Midjourney (Felix) + niji-v5
python aufbau3.py --p5  --stamm p5z   # Midjourney v6 sample
python tags_pack.py <pack>            # WD rating + protection flags per image
python embed.py --rueckgrat siglip_mid --pack <pack> --aug            # perturbed features for training
python embed.py --rueckgrat siglip_mid --pack <pack> --nur-split test # unperturbed features for testing
python embed.py --rueckgrat siglip_mid --pack p2z --nur-split val     # unperturbed, for calibration
python kanal_siglip.py        # features of all Telegram test channels
```

(`clip_mid` and `dino_mid` are needed only for the older multi-backbone heads.)

## 3. Train and calibrate

```bash
python train3.py --zusatz v3z2,v3z3,p2z,p3z,p4z,p5z --rueckgrate siglip_mid --rating-ausgleich --kalib-p2 \
    --quelle-gruppe midjourney_felix,nijijourney_p1atdev,midjourney_v6_scrape=midjourney --tag p8s
python train3.py ... --tag p8sb --seed-basis 100     # second run, other seeds
# p8sx = the ten heads of p8s and p8sb (copied as kopf_p8sx_s0..9.npz)
python kalibriere.py p8sx     # threshold: 1 % false positives on unperturbed human val groups
```

## 4. Evaluate

```bash
python gleich_fa.py v3 p6x p8x p8sx     # all heads at equal false-positive rate, 18 channels
python p2_test.py  v3 p6x p8x p8sx      # false positives per human group, own threshold
python mj_versionen.py v3 p6x p8x p8sx  # Midjourney by version
python kontrolle3.py p8sx               # resampling-leak and robustness controls
python bench.py v3 p6x p8x p8sx         # against the seven production detectors
python ausschluss.py lauf --praefix ausS --rueckgrate siglip_mid && python ausschluss.py auswerten --praefix ausS --rueckgrate siglip_mid
python ergebnisse_md.py                 # docs/RESULTS.md
```

## 5. Use

```bash
python pruefe.py <folder>               # read-only scan with the default head (p8sx); CSV + HTML contact sheet
python export_koepfe.py p8sx            # release zip of the heads
```
