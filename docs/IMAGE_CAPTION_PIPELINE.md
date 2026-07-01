# IMAGE_CAPTION_PIPELINE — Figure ↔ Caption Matching

Step-by-step explanation of how each extracted solar image is matched to its figure
caption. This is **shared logic**, not a pipeline stage: the geometry is captured once by
`extract`, and the matching is performed once by `metadata`.

---

## Table of Contents

1. [Overview](#overview)
2. [Where the pieces run](#where-the-pieces-run)
3. [Prerequisites](#prerequisites)
4. [Module Structure](#module-structure)
5. [Caption Extraction Algorithm](#caption-extraction-algorithm)
6. [Image-to-Caption Matching Algorithm](#image-to-caption-matching-algorithm)
7. [Known Limitations](#known-limitations)

---

## Overview

To read the observational metadata out of a paper, each solar image must be tied to the
figure caption (and, later, the body paragraphs) that describe it. Matching is done by
**vertical proximity** on the PDF page: an image's placement rectangle is compared against
the bounding boxes of the figure captions, and the closest caption wins.

The work is split so it happens exactly once:

- **`extract`** records each image's page-placement `bbox` in `extraction_log.json`
  (via `utils/pdf_extractor.py`). This is the image side of the match.
- **`metadata`** reads those bboxes back from the log and matches them to captions it
  extracts from the PDF (via `utils/caption_extractor.py`). This is the caption side.

```
extract  ──►  output/images/<name>/extraction_log.json   (each image: page + bbox + filename)
                                    │
metadata ──►  extract_all_captions(pdf)      →  captions_by_page
              match_image_to_caption(page, bbox, captions_by_page)  →  Caption
```

Because the bbox is persisted at extract time, `metadata` never re-derives image geometry
from the PDF — it only opens the PDF for caption text and for the body-text paragraphs that
cite each figure.

---

## Where the pieces run

| Piece | Module | Called by |
|-------|--------|-----------|
| Image `bbox` capture | `utils/pdf_extractor.py` (`_largest_image_bbox`) | `extract` |
| Caption extraction | `utils/caption_extractor.py` (`extract_all_captions`) | `metadata` |
| Image ↔ caption match | `utils/caption_extractor.py` (`match_image_to_caption`) | `metadata` |
| Body-text references | `utils/caption_extractor.py` (`extract_figure_body_refs`) | `metadata` |

> Historical note: earlier versions ran matching in a separate `label` stage and also
> classified the structure with a zero-shot BART model. Both were removed — the `metadata`
> LLM's `phenomenon` field supersedes the BART label, and matching now runs a single time
> inside `metadata`.

---

## Prerequisites

### 1. Run `extract` first (the PDF is kept by default)

`metadata` needs the `extraction_log.json` (for the image bboxes) and the original PDF (for
caption and body text). `extract` keeps the PDF at `output/papers/<name>.pdf` **by default**:

```bash
./tools/extract_plots.sh extract --id 2620529
```

If you passed `--no-keep-pdf`, re-run `extract` so the PDF is available.

### 2. PyMuPDF

Both the bbox capture and the caption extraction use **PyMuPDF** (`fitz`):

```bash
pip install pymupdf
```

---

## Module Structure

```
utils/
├── pdf_extractor.py       # captures each image's page-placement bbox (used by extract)
└── caption_extractor.py   # caption extraction + image↔caption matching (used by metadata)
```

`caption_extractor.py` is consumed by `scripts/metadata_extraction.py`; it is not called
directly by the shell wrapper.

---

## Caption Extraction Algorithm

**Implemented in:** `utils/caption_extractor.py`

### Data class: `Caption`

```python
@dataclass
class Caption:
    figure_label: str                           # "Figure 1", "Figure 2a"
    text: str                                   # Full merged text (whitespace-normalized)
    page: int                                   # 1-based page number
    bbox: Tuple[float, float, float, float]     # (x0, y0, x1, y1) in PDF points
```

`__post_init__` normalizes whitespace in `text` via `" ".join(text.split())`, so trailing spaces and newline artifacts from PDF text extraction are always removed.

### Helper: `_assemble_block_text(block)`

PyMuPDF returns each text block as a nested dict: `block → lines → spans → text`. This helper concatenates all span texts across all lines in a block into a single string, joining lines with a space. This flattens the internal structure so that caption text that wraps across lines within the same PDF text block is treated as a single string.

### Helper: `_union_bbox(a, b)`

Returns the smallest axis-aligned bounding rectangle that contains both `a` and `b`: `(min(x0), min(y0), max(x1), max(y1))`. Used when merging continuation blocks into a multi-paragraph caption.

### Main function: `extract_all_captions(pdf_path)`

This function iterates every page in the PDF and uses a **state machine** with a single `current` accumulator variable to handle captions that span multiple text blocks.

**Detection rule:** A text block is the start of a new caption if its assembled text matches the regular expression:

```python
CAPTION_RE = re.compile(r"^(?:Figure|Fig\.)\s*\d+[a-z]?[.:\s]", re.IGNORECASE)
```

This matches `"Figure 1."`, `"Figure 2a:"`, `"Fig. 4."`, `"Fig. 5a "` (note trailing space), etc. It does not match `"Figures 1–3"` (plural) or mid-sentence uses of `"Fig."` that do not start a block.

**Multi-block merging:** In astronomy journals, figure captions are often long enough to span more than one text block in the PDF's internal layout. The state machine handles this by treating any non-caption text block that follows an active caption (on the same page) as a continuation block, merging it by concatenating the text and unioning the bounding boxes.

**State machine transitions:**

| State | New block | Action |
|-------|-----------|--------|
| No active caption | Caption block | Start new `current` |
| No active caption | Non-caption block | Skip |
| Active caption, same page | Caption block | Finalize `current`, start new one |
| Active caption, same page | Non-caption block | Merge into `current` |
| Active caption, new page | Any block | Finalize `current` first, then process the block normally |

After iterating all pages, any remaining `current` caption is finalized (handles captions on the last page of the document).

**Output:** A dict `{page_number: [Caption, ...]}` where each list is sorted top-to-bottom by `bbox[1]` (the y-coordinate of the caption's top edge).

### Body-text references: `extract_figure_body_refs(pdf_path)`

Scans all non-caption text blocks and buckets each paragraph under every figure number it
cites (`"Fig. 2"`, `"Figure 2a"`, `"Figs. 2 and 3"`, …). `metadata` uses this to give the LLM
the richer observational description from the main text, not just the terse caption.

---

## Image-to-Caption Matching Algorithm

**Implemented in:** `match_image_to_caption(img_page, img_rect, captions_by_page)` — `utils/caption_extractor.py`

`img_rect` is a `fitz.Rect` built from the bbox recorded in `extraction_log.json`
(`fitz.Rect(*entry["bbox"])`), so no fresh image-geometry pass over the PDF is needed.

The function uses a **vertical gap** as the distance metric between an image and a caption. The gap is:

```python
gap = min(
    |caption.y0 − image.y1|,   # caption top vs image bottom
    |caption.y1 − image.y0|,   # caption bottom vs image top
)
```

This measures the shortest distance between the closest edges of the two objects, regardless of which one appears above the other on the page. It works correctly for both "caption below image" (the most common layout in astronomy journals) and "caption above image" layouts.

**Search strategy (three tiers):**

1. **Same page** — collect all captions on `img_page`, pick the one with the smallest gap. Returns `confidence = "same_page"`. This is the expected case: caption and image are on the same page.

2. **Adjacent pages** — if there are no captions on the image's page (e.g., a full-page figure with the caption on the next page), check `img_page + 1` first, then `img_page − 1`. Returns the first caption found. Returns `confidence = "adjacent_page"`.

3. **No caption found** — returns `(None, "none")`. This happens when the PDF has no detected captions near the image (e.g., the paper uses "Fig." abbreviations that the regex does not match).

**Note:** Same-page matching picks the globally closest caption, not the closest caption that is specifically below the image. In multi-figure pages, this can occasionally assign the wrong caption if two images are close together and their captions are interleaved.

Once matched, `metadata` pulls the figure number out of `caption.figure_label`
(`_LABEL_NUM_RE`) and looks up the citing paragraphs from `extract_figure_body_refs`, then
sends caption + paragraphs to the LLM. See `docs/METADATA_EXTRACTION.md`.

---

## Known Limitations

1. **Figure label regex matches "Figure N" and "Fig. N":** Both the full word and the `"Fig."` abbreviation are recognized (case-insensitive). Other abbreviations such as bare `"F."` or non-English equivalents (e.g. `"Abbildung"`) are not matched and will yield no caption (`"none"`).

2. **Multi-panel figures assigned a single caption:** When a figure contains a 2×3 grid of panels (e.g., six AIA wavelengths), PyMuPDF extracts it as a single embedded image with a single bbox, so it matches one caption. Downstream, the LLM reads that caption plus the citing paragraphs to describe the observation.

3. **Adjacent-page captions may be wrong:** When the match confidence is `"adjacent_page"`, the first caption found on the adjacent page is used — it may belong to a different figure. This is a heuristic fallback; verify these cases manually.

4. **Captions spanning page boundaries:** If a caption starts at the bottom of one page and continues at the top of the next, the state machine finalizes the caption at the page boundary and discards the continuation. This is an edge case in practice (PDF layouts almost never split a caption across pages) but can result in truncated caption text.

5. **The PDF must be present:** matching needs `output/papers/<name>.pdf`. `extract` keeps it by default; if you ran `--no-keep-pdf`, re-run `extract`.
