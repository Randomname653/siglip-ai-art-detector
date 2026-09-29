# Findings and pitfalls

What we learned while building the detector — most of it is about ways a detector can look good while
learning the wrong thing. Each item names the check that caught it.

## Leaks: the detector learns *how* an image was processed, not *who* made it

- **Resampling leak (v1).** All human wallpapers were exactly 1024 px, so normalizing to 512 px was an exact
  halving for them and an odd factor for the AI images. A head reached test AUC 0.987 — and fell to 0.643 once
  both classes were rescaled by 75 % and recompressed. *Fix:* train only on randomly perturbed images
  (`norm.stoere`: scale, crop, JPEG), identical for both classes; *check:* `kontrolle3.py` rescales humans up and
  AI down and measures the score shift.
- **Thumbnail leak.** Half of an AI training channel were Telegram preview thumbnails (520 px, q67); the human
  side had none. *Fix:* thumbnails are excluded everywhere (`lesen.Pack.ok`).
- **Aspect ratio / file size.** Checked with `leak.py` and `kontrolle.py`: after normalization, format and size
  alone must not separate the classes.

## Confounds in the data

- **"New style = AI" (temporal confound).** All human images were from before 2022, all AI images after. The head
  flagged 6–7 % of human art from 2026 as AI. *Fix:* current human art (Danbooru 2022–2026, a site that bans AI
  images) in training, with disjoint artists for calibration and test.
- **"Non-anime = AI".** With only anime on the human side, the head flagged 10.8 % of human wallpapers and 8 % of
  human 3D art. *Fix:* human non-anime art (Danbooru style tags, Wallhaven pre-2022) → 1–2 %.
- **Professional digital art is not a blind spot.** On 2,154 concept-art and game wallpapers with an embedded
  creation date before 2022 (a kind of art absent from training) the head flags 0.84 % — its 1 % design point.
  Work dated 2022+ is flagged at 3.1 %: undisclosed AI assistance and the "new style" confound cannot be told
  apart there (`profikunst_datum.py`). File dates of copies are useless; embedded XMP/EXIF dates are not.
- **"Explicit = AI" was *not* learned** (explicit human art is not flagged more often); rating balancing in the
  loss (`--rating-ausgleich`) keeps it that way.
- **A test channel was contaminated** with human fan wallpapers; it was dropped as ground truth. Every new
  channel is checked by sampling before it is used.
- **Early AI (2022) was missing**: 2022 posts of a channel were detected at 67 % vs. 90 % for 2023. DiffusionDB
  (SD 1.x, Aug 2022) closed most of the gap.

## Measuring honestly

- **Calibrate on unperturbed validation data, per group.** Pooling perturbed validation images of all sources
  put the threshold where large, easy groups dominated → 2 % instead of 1 % false positives on anime, 3.4 % on
  wallpapers. *Fix:* `kalibriere.py` — unperturbed val images, three human groups weighted equally.
- **Compare heads at equal false-positive rate, not at their own thresholds.** Heads calibrated differently look
  better or worse only because their thresholds sit elsewhere. A "loss" of 3–5 points on anime channels turned out
  to be a stricter threshold on human anime; at equal anime false-positive rate the heads were even. `gleich_fa.py`
  and AUC are the fair comparisons.
- **Seed variance is real**: up to ±2.6 points per channel between two training runs → only ensembles are
  compared, and a second run with other seeds confirms any small gain.
- **Seen vs. unseen generators.** Test images from generators that are in training (other images, other creators)
  measure familiarity; leave-generator-out experiments (`ausschluss.py`) measure generalization.

## Duplicates between training and test

- **Perceptual hashes (dHash) are not a proof**: 583 "duplicates", none real in a sample — dark images with soft
  gradients collide.
- **The detector's own features are not a duplicate test either**: the same image perturbed vs. unperturbed had
  cosine 0.82, the best *different* image 0.79.
- **Pixel check works**: 32×32 grayscale thumbnails, cosine ≥ 0.97; known identical pairs score 0.9999
  (`embed_dubletten.py`).

## What makes the detector good

- **Intermediate layers, not the final embedding.** CLS tokens of five intermediate layers beat the final layer by a
  wide margin (CLIP: 65 → 95 % on own generators). Final layers describe *what* is shown, middle layers still carry
  *how* it was made.
- **The backbone matters most.** SigLIP 2 (so400m) intermediate layers took the channel average from 93.8 % to
  97.4 % at equal false positives; alone it is as good as CLIP + DINOv2 + SigLIP together, at twice the speed.
- **Diversity of generators beats quantity.** Each source gets equal weight in the loss; three Midjourney sources
  are weighted as one so they do not crowd out anime sources.
- **Frozen backbone + small head** trains in minutes, which made dozens of controlled experiments possible.
- **Bigger is not better here.** SigLIP 2 giant (40 layers, 3× slower) did not beat so400m in a pilot. In both
  models the signal sits in the late-middle layers (so400m 17–22 of 27, giant 28–36 of 40); early layers carry
  little (giant layer 4 alone: 32 % vs. layer 32 alone: 95 %).
- **Spread the layers, don't stack the strongest.** Five layers over the whole depth (7/12/17/22/27) beat five
  layers from the strongest range (16–24: 93.9 vs. 95.4 % on channels); a denser choice of nine layers is within
  noise (95.8 %). Weak early layers still add information the late ones lack.
- **Edited human images drift toward "AI"** (captions, noise, upscaling, strong crops: 2-5 % false positives
  instead of 1 %), while edited AI images stay detected. Training with the same kinds of edits on both classes
  (p9sx) improved AUC under every edit and raised hit rates on edited AI, but did not bring edited humans back
  below the threshold — the drift is not simply a missing augmentation. A 50 % crop of a 512 px image is really a
  256 px image; its false-positive rate is the small-image effect. Worse, the edit augmentation slightly *hurt*
  generalization to unseen generators (leave-generator-out mean 96.4 → 96.0 %, 8 of 11 lower): the edits blur
  the fine traces that also give away unknown generators. More augmentation is not free.
- **Legacy detectors add nothing.** Stacking the head with any of the seven previous production detectors does not
  improve it (AUC 0.9976 alone, 0.9975–0.9977 combined), so the production pipeline now uses the head only.

## The hard human cases

The human images the head is most sure are AI fall into three kinds: pictures that look like modern generator
output (some may be undetected AI in the "human" sets), genuinely hard human art (palette-knife paintings in a
popular style, dense ornamental/psychedelic art, soft watercolours) and junk (flat textures, a photo). For the
Danbooru part this was checked against the site itself: of 298 Danbooru "human" posts above the 1 % threshold,
none has been tagged AI or deleted since (`danbooru_nachpruefen.py`) — they count as genuine false positives.
No threshold reaches zero false positives: at score 0.9999, 0.07 % of 55,497 human test images remain, while
91 % of channel AI images are above it. The production pipeline therefore auto-quarantines only above 0.9999
(reversible, files are moved, never deleted) and sends everything between the 1 % threshold and 0.9999 to review.

## Engineering notes

- SigLIP's Hugging Face image processor runs in one thread and starved the GPU (48 img/s); the same resize and
  normalization in a thread pool gives 73–135 img/s with identical features (cosine 0.999999).
- Status files that quote a previous stage's "DONE" line can trigger a waiting pipeline stage early — wait for a
  line that *ends* with the marker, and ignore the first line.
