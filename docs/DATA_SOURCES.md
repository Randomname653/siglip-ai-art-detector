# Data sources

**No images are distributed with this repository.** The images were collected for local training and
evaluation of a detector only. Every source folder carries a `manifest.jsonl` with one line per checked
item (post ID or URL, stored file, or the reason it was rejected), so every image can be traced back to its
origin and removed on request.

Collection rules:

- **Only where allowed.** Official APIs (CivitAI, AIBooru, Danbooru, Hugging Face) or files the owner released
  under a free license. No scraping of sites whose terms forbid it (e.g. Cara, ArtStation, Pixiv, Newgrounds).
  Ready-made scrape datasets of *AI images* that are already public are used as small samples and marked as such.
- **Child protection** on every source, see [SAFETY.md](SAFETY.md).
- **No real photos.** The detector is for digital art; a photo filter (`imgutils.validate.anime_real`) and the
  WD tag `photo_(medium)` remove photographs from every source.
- **Same processing for both classes.** Web images are stored as a "Telegram simulation" (longest side ≤ 1280 px,
  JPEG q87 4:2:0), and all images are normalized identically before any model sees them (`norm.py`), so the
  detector cannot learn file format, size or compression instead of origin.

Counts are after all filters and child-protection removals (state 2026-09-28).

## AI-generated (positive class)

| source | origin / license | size used | notes | use |
|---|---|---:|---|---|
| CivitAI | civitai.com API (images posted by their creators) | 121,495 | ~30 base models (SD 1.5, SDXL, Pony, Illustrious, NoobAI, Flux, Qwen, Z-Image, OpenAI, Nano Banana, Seedream, Grok, Imagen 4, …); anime/illustration style | train + test, split by creator |
| CivitAI semi-realistic/3D | same API, the images the anime style filter had rejected | 34,077 | realistic/3D styles, photos still removed | train + test, split by creator |
| AIBooru | aibooru.online API, ratings g/s only | 30,102 | includes NovelAI, Nijijourney, Midjourney, DALL·E | train + test, split by uploader |
| DiffusionDB 2M | huggingface.co/datasets/poloclub/diffusiondb, **CC0-1.0** | 17,016 | Stable Diffusion 1.x, Aug 2022; 40 of 2,000 parts, ≤ 25 images per (hashed) user | train + test, split by user |
| Felix Midjourney Archive | huggingface.co/datasets/wafflefan/felix-midjourney-archive, **CC0-1.0**, the creator's own images | 22,388 | Midjourney v3/v4 (2022) to v7, niji | train + test, split by job/prompt |
| niji-v5 (p1atdev) | huggingface.co/datasets/p1atdev/niji-v5, **CC0-1.0**, the creator's own images | 4,727 | Nijijourney v5 | train + test |
| Midjourney v6 sample | huggingface.co/datasets/brivangl/midjourney-v6-llava (dataset license MIT; scraped from the Midjourney Discord by its authors) | 5,283 | 8 of 124 files, 2,500 random images each, before filtering | train + test |
| own generators | 14 models run locally on Wallhaven motifs (`sd15`, `sdxl_base`, `pony`, `illustrious`, `noobai`, `wai_illustrious`, `animagine`, `nai_v1`, `nai_v2`, `anima`, `z_anime`, `qwen_image`, `hdm_xut`, `animepro_flux`) | 95,693 | the project's first dataset | train + test |
| Telegram AI channel (training) | one public AI-art channel, exported by the project owner | 12,881 | ~90 % explicit anime | train |
| Telegram AI channels | 18 public channels, exported by the project owner with Telegram Desktop | ~133,000 | **test only, never trained on**; ads, screenshots and thumbnails removed | test |

Telegram test channels are referred to by neutral labels: six anime channels (A–F), three Midjourney channels (A–C), one Leonardo and one Stable Diffusion channel, and seven channels mixing several generators (A–G). Images from training sources that also appear in a test channel would inflate
the test; a pixel-level duplicate check (`embed_dubletten.py`, 32×32 thumbnails) found none.

## Human-made (negative class)

| source | origin / license | size used | notes | use |
|---|---|---:|---|---|
| Danbooru 2022-01 … 2026-03 | danbooru.donmai.us API (bans AI images) | 64,954 | official art + scored posts, all ratings | train |
| Danbooru 2023-01 … 2026-06 | same API, **different artists** | 10,904 | calibration (val) and test | test, calibration |
| Danbooru non-anime 2022 … 2026 | same API, style tags realistic, painting, watercolor, 3d, pixel_art, traditional_media, landscape, concept_art, cartoon, western_comics, card, character_sheet, promotional_art | 56,845 | split by artist 80/5/15 | train, calibration, test |
| Wallhaven "general" < 2022 | the project owner's local wallpaper archive; file date before 2022 (before image generators were usable) | 67,969 | split by file 85/5/10 | train, calibration, test |
| Wallhaven anime < 2022 | the owner's wallpaper archive | 69,693 | first dataset (v1/v2) | train + test |
| Telegram art channels < 2022 | 28 public human art channels from the owner's archive, posts before 2022 | 102,350 | v2 | train + test |

## What is *not* used

- Real photographs (filtered everywhere).
- Anything from sites whose terms forbid automated collection.
- Images that fall under the child-protection rules — they are never stored; if found later, the file is
  deleted and its path recorded in a block list that every data loader honours.
