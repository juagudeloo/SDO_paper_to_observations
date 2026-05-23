# IMAGE_CAPTION_PIPELINE — Pipeline Documentation

Step-by-step explanation of the image caption extraction and solar structure classification pipeline (Stage 3: `label`).

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pipeline Structure](#pipeline-structure)
4. [Stage 3: Label (`label`)](#stage-3-label)
5. [Caption Extraction Algorithm](#caption-extraction-algorithm)
6. [Image-to-Caption Matching Algorithm](#image-to-caption-matching-algorithm)
7. [Solar Structure Classification Algorithm](#solar-structure-classification-algorithm)
8. [Output Folder Layout](#output-folder-layout)
9. [End-to-End Example](#end-to-end-example)
10. [Known Limitations](#known-limitations)

---

## Overview

The `label` stage is the third and final stage of the pipeline. It takes the output folder produced by the `extract` stage and enriches each solar image with two pieces of information:

1. **Caption** — the figure caption from the PDF that is closest to the image on the page.
2. **Structure label** — a controlled-vocabulary classification of the solar structure described in that caption, produced by a zero-shot NLP model.

```
output/papers/YYYY-MM - Author, I/         (produced by extract stage)
    ├── extraction_log.json
    ├── paper_XXXXXX.pdf
    ├── solar_001_p2_aia_false_color.png
    └── solar_002_p3_hmi_grayscale.png
                  │
            (label stage)
                  │
                  ▼
output/
    ├── papers/YYYY-MM - Author, I.pdf       ← flat PDF copy
    ├── images/YYYY-MM - Author, I/          ← solar image copies
    │       ├── solar_001_p2_aia_false_color.png
    │       └── solar_002_p3_hmi_grayscale.png
    └── labels/YYYY-MM - Author, I.csv       ← one row per solar image
```

The CSV is the primary deliverable: it links each solar image to its caption text and the predicted solar structure label.

---

## Prerequisites

### 1. Run the extract stage first with `--keep-pdf`

The label stage needs the `extraction_log.json` and the original PDF in the paper folder. The PDF is not kept by default — you must pass `--keep-pdf` to the extract stage:

```bash
./tools/extract_plots.sh extract --id 2620529 --keep-pdf
```

If the paper folder already exists but has no PDF, re-run the extract stage with `--keep-pdf`.

### 2. HuggingFace Transformers

The NLP model (`facebook/bart-large-mnli`) is loaded via the `transformers` library:

```bash
pip install transformers>=4.30.0
```

The model weights are downloaded automatically from HuggingFace Hub on the first run and cached locally (`~/.cache/huggingface/hub/`). Subsequent runs use the cache.

### 3. PyMuPDF (already required by the extract stage)

Caption extraction uses **PyMuPDF** (`fitz`), which is the same dependency used in Stage 2:

```bash
pip install pymupdf
```

### 4. Python environment (pytorch_jupyter conda env)

Same as for the extract stage. The shell script activates this environment automatically.

---

## Pipeline Structure

```
tools/
└── extract_plots.sh         # Shell entry point (dispatches to scripts/)

scripts/                     # Main entry points (run directly)
└── label_plots.py           # Stage 3 orchestrator

utils/                       # Utility modules (imported by scripts/)
├── caption_extractor.py     # PDF caption and image bounding-box extraction
└── structure_classifier_nlp.py  # HuggingFace zero-shot structure classifier
```

The two utility modules (`utils/caption_extractor.py` and `utils/structure_classifier_nlp.py`) are consumed entirely by `scripts/label_plots.py` — they are not called directly by the shell script.

---

## Stage 3: Label

**Command:**
```bash
./tools/extract_plots.sh label --paper-dir output/papers/2012-01\ -\ Labrosse,\ N
```

**Implemented in:** `scripts/label_plots.py` (orchestrator) + `utils/caption_extractor.py` + `utils/structure_classifier_nlp.py`.

**What it does, step by step:**

### Step 1: Load the extraction log
**Function:** `_load_extraction_log(paper_dir)` — `scripts/label_plots.py`

Reads `extraction_log.json` from the paper folder. This JSON file was written by Stage 2 and contains the paper metadata at the top plus one record per extracted image (both solar and non-solar), each with fields: `index`, `page`, `is_solar`, `image_type`, `score`.

The full list of image records is sorted by `index` to ensure global positional order is preserved — this is critical for correctly correlating images to their PDF bounding boxes in Step 4.

### Step 2: Find the PDF
**Function:** `_find_pdf(paper_dir)` — `scripts/label_plots.py`

Searches for a file matching `paper_*.pdf` in the paper folder using `glob`. Raises a clear `FileNotFoundError` with re-run instructions if no PDF is present (i.e., if the extract stage was run without `--keep-pdf`).

### Step 3: Create output directories
**Implemented in:** `main()` — `scripts/label_plots.py`

Creates (if not already present):
- `output/images/<folder_name>/` — destination for solar image copies
- `output/labels/` — destination for the CSV
- `output/papers/` — destination for the flat PDF copy

### Step 4: Copy files
**Functions:** `_copy_idempotent(src, dst)`, `_solar_image_filename(entry, solar_rank)` — `scripts/label_plots.py`

`_copy_idempotent` copies a file only if the destination does not already exist with the same size (avoids redundant I/O on re-runs).

`_solar_image_filename` reconstructs the filename of each solar image as `solar_NNN_pP_<image_type>.png` — the same naming convention used by the extract stage. The solar rank (`NNN`) is a 1-based counter that increments only over images with `is_solar=True`, mirroring how Stage 2 numbered the files on disk.

### Step 5: Extract captions and image bounding boxes from the PDF
**Functions:** `extract_all_captions(pdf_path)`, `get_all_image_bboxes(pdf_path)` — `utils/caption_extractor.py`

`extract_all_captions` scans every page of the PDF and returns a dict mapping page number → list of `Caption` objects (see [Caption Extraction Algorithm](#caption-extraction-algorithm)).

`get_all_image_bboxes` scans every page and returns a dict mapping page number → list of `(xref, fitz.Rect)` pairs, one per unique image (using the same xref deduplication as Stage 2's `extract_pdf_images`). This is then flattened into a global list by `_build_global_bbox_list()`, which iterates pages in ascending order and extends the flat list with each page's entries. The Nth item in this global list corresponds to the log entry with `index=N`, because Stage 2 iterated pages and images in the same order with the same deduplication.

### Step 6: Match each solar image to its nearest caption
**Function:** `match_image_to_caption(img_page, img_rect, captions_by_page)` — `utils/caption_extractor.py`

For each solar image, its page number and bounding rectangle are looked up from the global bbox list using the image's `index`. The matching function then finds the closest caption by vertical gap (see [Image-to-Caption Matching Algorithm](#image-to-caption-matching-algorithm)).

If no bounding rectangle was found for an image (e.g., the PDF embeds the image via an unusual mechanism that `page.get_image_rects()` cannot resolve), the caption match is skipped and `match_confidence` is set to `"none"`.

### Step 7: Classify the solar structure from the caption text
**Function:** `classify_structure(caption_text)` — `utils/structure_classifier_nlp.py`

The matched caption text is passed to the NLP classifier. If no caption was found, `caption_text` is an empty string and the function returns `("Other", 0.0)`. See [Solar Structure Classification Algorithm](#solar-structure-classification-algorithm) for details.

### Step 8: Write the CSV
**Implemented in:** `main()` — `scripts/label_plots.py`

Writes one CSV row per solar image to `output/labels/<folder_name>.csv` using Python's `csv.DictWriter` with UTF-8-BOM encoding (`utf-8-sig`) so the file opens correctly in Excel without character issues.

**CSV columns:**

| Column | Description |
|--------|-------------|
| `paper_filename` | `<folder_name>.pdf` |
| `paper_path` | Relative forward-slash path to the flat PDF copy |
| `image_path` | Relative forward-slash path to the solar image copy |
| `figure_label` | Caption label, e.g. `"Figure 3"` or `"Figure 3a"` |
| `caption_text` | Full normalized caption text (whitespace-collapsed) |
| `structure_label` | Predicted solar structure from the NLP model |
| `structure_confidence` | Model confidence score (0.0–1.0) |
| `caption_match_confidence` | `"same_page"`, `"adjacent_page"`, or `"none"` |
| `image_type` | `"aia_false_color"`, `"hmi_grayscale"`, or `"unknown"` |
| `classifier_score` | Solar classifier score from Stage 2 (0.0–1.0) |

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

### Image bbox function: `get_all_image_bboxes(pdf_path)`

Mirrors the xref-deduplication logic of Stage 2's `extract_pdf_images()` exactly. For each page, it calls `page.get_images(full=True)` to get image records, skips any xref already seen (to deduplicate shared image objects), and calls `page.get_image_rects(xref)` to get the position of the image on the page. Only the first rect is used (images appear at one location per page). Empty rects are skipped.

**Why mirror Stage 2's logic exactly?** The global index in `extraction_log.json` is assigned by Stage 2 in the order that images are encountered (page order, then encounter order within the page, with xref deduplication). `get_all_image_bboxes` must follow the identical traversal so that `global_bboxes[N]` in Stage 3 corresponds to the log entry with `index=N`.

---

## Image-to-Caption Matching Algorithm

**Implemented in:** `match_image_to_caption(img_page, img_rect, captions_by_page)` — `utils/caption_extractor.py`

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

**Note:** Same-page matching picks the globally closest caption, not the closest caption that is specifically below the image. In multi-figure pages, this can occasionally assign the wrong caption if two images are close together and their captions are interleaved. The `caption_match_confidence` column in the CSV lets you audit these cases.

---

## Solar Structure Classification Algorithm

**Implemented in:** `utils/structure_classifier_nlp.py`

### Controlled vocabulary

```python
STRUCTURE_LABELS: List[str] = [
    "Active Region",
    "Flare",
    "Prominence",
    "Coronal Hole",
    "Sunspot",
    "Filament",
    "Plage",
    "Faculae",
    "Granulation",
    "Supergranulation",
    "Polarity Inversion Line",
    "Filament Channel",
    "Coronal Loops",
    "Coronal Cavities",
    "Helmet Streamer",
    "Pseudostreamer",
    "Polar Crown Filament",
    "Sigmoid",
    "Post-Flare Loops",
    "Other", # Fallback never passed to the NLP model
]
```

`"Other"` is reserved as a fallback label for when no candidate scores above the threshold, or when no caption is available. The NLP model only scores the eighteen concrete structure labels.

### Caption preprocessing: `_preprocess_caption(text)`

Before classification, the raw caption text is cleaned:

1. **Strip the figure label prefix** — the regex `r"^Figure\s+\d+[a-z]?[.:\s]+"` removes the leading `"Figure 1."` or `"Figure 2a: "` part, since it carries no information about the solar structure.
2. **Normalize whitespace** — `" ".join(text.split())` collapses all internal whitespace runs.
3. **Truncate to 512 characters** — BART's tokenizer has a practical input limit; captions longer than ~512 characters provide diminishing returns for zero-shot classification.

Example: `"Figure 1. AIA 304 Å image of the prominence..."` → `"AIA 304 Å image of the prominence..."`

### Lazy model singleton: `_get_pipeline()`

The HuggingFace pipeline is loaded **once** per process and stored in the module-level variable `_PIPELINE`. This avoids reloading the ~1.6 GB BART model weights for every image in a multi-image paper.

The pipeline is configured as:
```python
pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0,       # GPU if available, else -1 for CPU
)
```

`_cuda_available()` checks `torch.cuda.is_available()` to decide between GPU (faster, ~2 s/image) and CPU (slower, ~20–30 s/image) execution.

### Main function: `classify_structure(caption_text)`

The function implements a three-step decision process:

**Step 1: Empty caption guard**

If `caption_text` is `None` or blank, immediately return `("Other", 0.0)`. This handles images where no caption was matched.

**Step 2: Zero-shot classification**

The cleaned caption is passed to the BART pipeline with:
- `candidate_labels`: the eighteen concrete structure labels (all entries in `STRUCTURE_LABELS` except `"Other"`)
- `hypothesis_template`: `"This solar image shows a {}."` — this is the natural-language hypothesis that BART's NLI head evaluates. The label is substituted into `{}`. BART classifies the caption as either entailing or contradicting each hypothesis and ranks the candidates by their entailment probability.
- `multi_label=False`: the probabilities are normalized across all candidates (softmax), so they sum to 1.0. The task is single-label classification.

The pipeline returns `result["labels"]` (sorted by score descending) and `result["scores"]`.

**Step 3: Threshold gate**

```python
SCORE_THRESHOLD = 0.10
```

If the top candidate's score is ≥ 0.10, that label and score are returned. If the score is below the threshold, `"Other"` is returned but the raw score is still included in the output — this lets you audit borderline cases in the CSV without losing the model's opinion.

| Top score | Returned label | Returned confidence |
|-----------|---------------|---------------------|
| ≥ 0.10 | Top candidate | Top score (0.10–1.0) |
| < 0.10 | `"Other"` | Top score (below 0.10) |

**Why 0.10?** With nineteen labels (eighteen candidates + `"Other"`) and a softmax output, a score of 1/19 ≈ 0.053 is the random baseline. 0.10 is roughly double the baseline — it accepts cases where BART has a clear preference but is not highly confident, while rejecting cases where all candidates score nearly equally (BART is undecided).

---

## Output Folder Layout

After the label stage completes, the `output/` directory has three sub-trees:

```
output/
├── papers/
│   └── YYYY-MM - LastName, F.pdf           ← flat PDF copy for reference
├── images/
│   └── YYYY-MM - LastName, F/
│       ├── solar_001_p2_aia_false_color.png
│       └── solar_002_p3_hmi_grayscale.png
└── labels/
    └── YYYY-MM - LastName, F.csv           ← one row per solar image
```

The `output/papers/` folder is shared across all processed papers. The `images/` and `labels/` sub-trees use the same `YYYY-MM - LastName, F` folder/file naming as Stage 2's output folder (see `EXTRACT_PLOTS.md` — Output Folder Naming).

**Path encoding in CSV:** Both `paper_path` and `image_path` columns use **forward-slash separators** even on Windows (built via `PurePosixPath`), so the CSV can be opened on any platform without path separator issues.

---

## End-to-End Example

```bash
# 1. Start the API (in a separate terminal)
cd ../NASA_ADS_SDO && ./run_api.sh

# 2. List papers from 2012 to 2013
./tools/extract_plots.sh list --start 2012-01-01 --end 2013-12-31
# → papers_20120101_20131231.csv

# 3. Extract solar images (--keep-pdf is required for the label stage)
./tools/extract_plots.sh extract --id 2620529 --keep-pdf
# Output:
#   Processing paper ID 2620529
#     Title     : White-light flares: a TRACE, RHESSI and SOHO/MDI multi-wavelength stu...
#     Author    : Song, Y
#     Output    : output/papers/2012-01 - Labrosse, N
#   Found 5 embedded images in PDF
#     [SOLAR] img-002 p3 1024x768 rgb  score=0.65 [aia_false_color_22%, ...]
#     ...
#   Extraction complete: 3/5 images classified as solar observations

# 4. Label extracted images
./tools/extract_plots.sh label --paper-dir "output/papers/2012-01 - Labrosse, N"
# Output:
#   Labeling paper: 2012-01 - Labrosse, N
#     Title       : White-light flares: a TRACE, RHESSI and SOHO/MDI...
#     First author: Song, Y
#     Solar images: 3 / 5 extracted
#     PDF copied  : output/papers/2012-01 - Labrosse, N.pdf
#     Images dir  : output/images/2012-01 - Labrosse, N
#     Extracting captions from PDF...
#       Found 6 figure caption(s) across 4 page(s)
#     Loading NLP model for structure classification...
#   Labeling complete: 3 row(s) written
#   CSV: output/labels/2012-01 - Labrosse, N.csv

# 5. Browse the output CSV
cat output/labels/2012-07\ -\ Song,\ Y.csv
```

To see per-image caption matching and structure classification decisions, add `--verbose`:

```bash
./tools/extract_plots.sh label \
    --paper-dir "output/papers/2012-01 - Labrosse, N" \
    --verbose
# solar_001 p3  caption="Figure 3"  [same_page]  → Flare (0.7821)
# solar_002 p4  caption="Figure 4"  [same_page]  → Active Region (0.5130)
# solar_003 p5  caption=(none)      [none]        → Other (0.0000)
```

---

## Known Limitations

1. **Figure label regex matches "Figure N" and "Fig. N":** Both the full word and the `"Fig."` abbreviation are recognized (case-insensitive). Other abbreviations such as bare `"F."` or non-English equivalents (e.g. `"Abbildung"`) are not matched and will produce `caption_match_confidence = "none"`.

2. **Multi-panel figures assigned a single caption:** When a figure contains a 2×3 grid of panels (e.g., six AIA wavelengths), PyMuPDF extracts it as a single embedded image. The label stage assigns one caption row to this image. If the panels show different structures, the NLP model classifies the dominant one mentioned in the caption (or `"Other"` if the caption discusses all panels generically).

3. **Adjacent-page captions may be wrong:** When `caption_match_confidence = "adjacent_page"`, the first caption found on the adjacent page is used — it may belong to a different figure. This is a heuristic fallback; always verify these rows manually.

4. **BART model requires internet on first run:** The weights (~1.6 GB) are downloaded from HuggingFace Hub on the first call to `_get_pipeline()`. Subsequent runs use the local cache (`~/.cache/huggingface/hub/`). In air-gapped environments, pre-download and set `TRANSFORMERS_CACHE` or use `transformers-offline` mode.

5. **GPU strongly recommended:** On CPU, BART inference takes ~20–30 seconds per caption. A paper with 10 solar images takes ~5 minutes on CPU. With a CUDA GPU, this drops to ~2 seconds per caption. The environment already has PyTorch with CUDA in the `pytorch_jupyter` environment.

6. **Captions spanning page boundaries:** If a caption starts at the bottom of one page and continues at the top of the next, the state machine finalizes the caption at the page boundary and discards the continuation. This is an edge case in practice (PDF layouts almost never split a caption across pages) but can result in truncated caption text.

7. **`--keep-pdf` must be passed at extract time:** The label stage requires the paper PDF. If you forgot `--keep-pdf`, re-run the extract stage — it is safe to re-run because it overwrites only the paper folder, and the `--keep-pdf` flag is the only way to retain the PDF.
