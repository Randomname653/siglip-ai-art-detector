# Child protection

Several sources contain explicit adult content; that is intentional (the detector must not learn
"explicit = AI" or "explicit = human"). **Sexualized depictions of minors are never collected, stored,
embedded, trained on or displayed.** The rules live in [`schutz.py`](../schutz.py) and are applied to every
source at download time and again retroactively whenever a rule is extended.

## Rules

1. **Tag block, every rating.** Images tagged with terms indicating minors (`loli`, `shota`, `child`,
   `toddler`, `aged_down`, `kindergarten_uniform`, …) are rejected — using the source's own tags (booru
   tags) *and* the WD tagger's predictions on the image (threshold 0.2). This also applies to general-rated
   images.
2. **Character and franchise list, non-general images.** For images that are not general-rated (WD
   sensitive/questionable/explicit, booru s/q/e), characters who are canonically minors (e.g. school
   students) and franchises whose casts are predominantly minors are blocked. The list is matched against
   post captions, booru character/copyright tags, prompts, and — since 2026-09-28 — against the characters
   the WD tagger recognizes in the image (threshold 0.75). General-rated depictions are allowed.
3. **Prompts.** For prompt-based sources (DiffusionDB, Midjourney), prompts containing blocked terms are
   rejected before the image is even opened.
4. **Conservative on AIBooru.** Only ratings g/s are downloaded there; the share of problematic content among
   explicit posts was too high for tag filtering alone.

## Enforcement

- Rejected items are never written to disk; the manifest keeps only the ID and the reason (`gesperrt`).
- Retroactive checks (`schutz_nachtrag.py`, `schutz_wd_figuren.py`) delete the stored file, reduce the
  manifest line to ID + reason and append the path to a block list. `lesen.Pack` and `lesen.kanal_X` exclude
  every listed file from all training and all tests.
- Contact sheets used for manual review (ads, duplicates) are built only from general/sensitive images.

## Numbers (2026-09-28)

The image-based character check removed 1,112 training images from CivitAI/DiffusionDB/Midjourney, 1,560 from
AIBooru and about 2,060 test images from the Telegram channels. Earlier text/tag-based passes removed several
thousand more. All current heads are trained on the cleaned data.

The lists are a safeguard, not a guarantee; additions are welcome.
