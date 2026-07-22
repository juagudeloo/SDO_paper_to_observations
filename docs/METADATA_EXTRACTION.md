# METADATA_EXTRACTION — Pipeline Documentation

Step-by-step explanation of the automated solar observation metadata extraction stage (`metadata`).

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pipeline Structure](#pipeline-structure)
4. [The Metadata Stage (`metadata`)](#the-metadata-stage)
5. [Model Loading and Quantization](#model-loading-and-quantization)
6. [HuggingFace Model Cache](#huggingface-model-cache)
7. [Inference Pipeline](#inference-pipeline)
8. [JSON Parsing and Fallback Strategy](#json-parsing-and-fallback-strategy)
9. [Output Format](#output-format)
10. [End-to-End Example](#end-to-end-example)
11. [Known Limitations](#known-limitations)

---

## Overview

The `metadata` stage consumes the canonical output layout produced by `extract` and uses a
locally-hosted large language model (Qwen2.5-14B-Instruct, 8-bit quantized) to extract
structured observation metadata. It works **per figure**: saved images are matched to their
nearest figure caption and then grouped by figure; for each figure the LLM reads the caption, the
body paragraphs that cite it, and any linked tables, and returns a figure record with a `panels[]`
array — one entry per sub-panel (a, b, c…), each a distinct observation. The result is one JSON
file per paper listing every figure with its panel-level observations.

```
output/images/2012-01 - Labrosse, N/
    ├── extraction_log.json      ← produced by extract (each image: page + bbox + filename)
    ├── solar_001_p3_aia_false_color.png
    └── solar_002_p3_aia_false_color.png
output/papers/2012-01 - Labrosse, N.pdf   ← produced by extract (kept by default)
                    │
              (metadata stage)
                    │
                    ▼
output/metadata/2012-01 - Labrosse, N.json
```

Papers are addressed by their canonical **name** (`--paper-name "2012-01 - Labrosse, N"`) or
processed in bulk (`--all`). The stage is fully resumable: re-running skips any paper whose
JSON already exists.

A key property: because `extract` already recorded each image's page-placement `bbox` in the
log, this stage does **not** re-derive image geometry from the PDF. It reads the bbox from the
log and only opens the PDF for caption text and citing paragraphs. See
`docs/IMAGE_CAPTION_PIPELINE.md` for the matching mechanism.

---

## Prerequisites

### 1. GPU with ≥ 16 GB VRAM

Qwen2.5-14B-Instruct loaded in 8-bit requires approximately 14–16 GB of VRAM. The `pytorch_jupyter` environment targets the NVIDIA RTX A4500 (20 GB). CPU inference is possible but extremely slow (hours per paper).

Verify GPU is accessible:
```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA RTX A4500
```

### 2. bitsandbytes and accelerate

8-bit quantization requires `bitsandbytes` and `accelerate`:

```bash
pip install bitsandbytes accelerate
```

### 3. HuggingFace Transformers

```bash
pip install transformers>=4.30.0
```

### 4. PyMuPDF (already required by `extract`)

Used for caption and body-text extraction:

```bash
pip install pymupdf
```

### 5. HuggingFace Hub access (first run only)

The Qwen2.5-14B-Instruct weights (~14 GB) are downloaded from HuggingFace Hub on the first run and cached in the project's `models/` directory (see [HuggingFace Model Cache](#huggingface-model-cache)). Subsequent runs use the cache.

If the cluster has no internet access, pre-download the model on a machine with internet:
```bash
HF_HOME=/scratchsan/observatorio/juagudeloo/Doctorado/SDO_paper_to_observations/models \
    huggingface-cli download Qwen/Qwen2.5-14B-Instruct
```
Then set `HF_HUB_OFFLINE=1` before running.

### 6. Python environment (pytorch_jupyter conda env)

The shell script activates this environment automatically. It already has `torch 2.6.0+cu124` and `transformers 4.57.6`.

---

## Pipeline Structure

```
tools/
└── extract_plots.sh              # Shell entry point (validates env, dispatches)

scripts/
└── metadata_extraction.py        # Stage orchestrator

utils/
├── folder_naming.py              # canonical paths: log_path(), pdf_path(),
│                                 # metadata_json(), iter_paper_names()
└── caption_extractor.py          # extract_all_captions(), match_image_to_caption(),
                                  # extract_figure_body_refs(), extract_all_tables(),
                                  # extract_figure_table_links()

models/                           # HuggingFace model cache (HF_HOME)
└── hub/
    └── models--Qwen--Qwen2.5-14B-Instruct/
```

The shell script validates that `bitsandbytes` and `transformers` are importable before dispatching to `scripts/metadata_extraction.py`. All PDF text extraction is handled by `utils/caption_extractor.py`.

---

## The Metadata Stage

**Command (single paper):**
```bash
./tools/extract_plots.sh metadata --paper-name "2012-01 - Labrosse, N"
```

**Command (all papers):**
```bash
./tools/extract_plots.sh metadata --all
```

Both accept `--output-dir DIR` (the canonical output root, default `output`) and `--model MODEL`.

**Implemented in:** `scripts/metadata_extraction.py`

**What it does, step by step:**

### Step 1: Collect paper names
**Implemented in:** `main()` — `scripts/metadata_extraction.py`

- With `--paper-name`: the single given name is used.
- With `--all`: `folder_naming.iter_paper_names(root)` lists every paper under `<root>/images/` (the presence of an image directory is the source of truth for "a paper exists").

### Step 2: Check for existing output (resumability)
**Implemented in:** `process_paper()` — `scripts/metadata_extraction.py`

The output path is `folder_naming.metadata_json(root, name)` → `<root>/metadata/<name>.json`. `main()` pre-filters the paper list against existing output **before** loading the model: any paper whose JSON already exists is printed as `[skip]` and the (~8 min) model load is skipped entirely if nothing is pending. This makes the stage safe to re-run after a crash or partial batch, and cheap to re-run when everything is already done.

Pass `--verbose` / `-v` to raise logging to `DEBUG` and see step-by-step progress (caption/table extraction, per-image `[i/N] … querying LLM` lines) instead of a silent wait during model load and inference.

### Step 3: Load the extraction log and locate the PDF
**Functions:** `_load_log()` — `scripts/metadata_extraction.py`; `folder_naming.pdf_path()`

Reads `<root>/images/<name>/extraction_log.json` for the solar image list (each entry carrying `page`, `bbox`, `image_type`, `filename`, …), paper title, and first author. The PDF is looked up at its canonical location `<root>/papers/<name>.pdf`; if it is missing (e.g. `extract` was run with `--no-keep-pdf`), the paper is written out with `status="failed"`.

The author string is resolved in order of priority:
1. PDF document metadata (`fitz.open(pdf_path).metadata["author"]`)
2. `first_author` field from `extraction_log.json`

### Step 4: Extract captions, body references, and tables from the PDF
**Functions:** `extract_all_captions()`, `extract_figure_body_refs()`, `extract_all_tables()`, `extract_figure_table_links()` — `utils/caption_extractor.py`

Several passes over the PDF are run once per paper (not once per image):

1. **`extract_all_captions(pdf_path)`** — scans every text block across all pages and identifies figure captions by the pattern `^(Figure|Fig\.)\s*\d+[a-z]?[.:\s]`. Returns a page-keyed dict of `Caption` objects (each with `figure_label`, full `text`, 1-based page number, and bounding box). Multi-block captions (captions split across consecutive text blocks) are merged automatically.

2. **`extract_figure_body_refs(pdf_path)`** — scans all non-caption text blocks and finds paragraphs that contain an inline figure citation such as `"Fig. 2"`, `"Figure 2a"`, or `"Figs. 2 and 3"` (matched by `\b(?:Figs?\.?\s*|Figures?\s+)(\d+[a-zA-Z]?)`). Returns a dict mapping figure number string (e.g. `"2"`, `"2a"`) to a deduplicated list of paragraph texts in document order.

3. **`extract_all_tables(pdf_path)`** — finds table captions (`^(Table|Tab\.)\s*\d+`) and captures each table's caption plus its body text (header, data rows, footnotes) by merging the following same-page blocks until the next caption, a running-prose block, or a character cap. `find_tables()` reconstructs poorly on vector-drawn A&A tables (often returns 0), but `get_text` still streams the grid as text in reading order, so values like NOAA AR number and per-event heliographic locations (`S17W08`) survive. Returns a dict mapping table number → `Table`.

4. **`extract_figure_table_links(pdf_path)`** — maps each figure number to the table numbers it is linked to, using a **hybrid** strategy: primary is reference-driven (a table cited in a paragraph that also cites the figure is linked), with a positional fallback (for a figure with no co-cited table, scan a ±4-paragraph window around each figure mention). Returns a dict mapping figure number → list of table numbers.

Image bounding boxes are **not** re-derived here — they come from the log (Step 5).

### Step 5: Match saved images to captions, then group by figure
**Functions:** `match_image_to_caption()` — `utils/caption_extractor.py`; `_group_images_by_figure()` — `scripts/metadata_extraction.py`

The stage processes **every image actually saved to disk** — i.e. every log entry with a non-null `filename` — not only those the classifier flagged `is_solar`. Each saved image is matched to a caption, then images are **grouped by figure number** so the observation unit is the figure, not the raster (a multi-panel figure is stored as an unpredictable number of rasters, so raster count never matches panel count).

For every saved image in `extraction_log.json`:

1. Build a `fitz.Rect` from the image's logged `bbox` (`fitz.Rect(*entry["bbox"])`). Images without a recorded bbox are left without a caption.
2. Call `match_image_to_caption(page, img_rect, captions_by_page)` to find the nearest caption by vertical gap (same page first, then adjacent pages, else none).
3. `_group_images_by_figure()` buckets images by the trailing figure number of their caption label; images that match no caption each form their own single-image group (keyed on filename) so nothing is dropped.

Each group yields: `figure_label`, `caption_text`, the citing `paragraphs`, the linked tables, and `source_images` (the group's rasters, each with `is_solar` + `classifier_score`).

### Step 6: Build prompt and query LLM (one call per figure)
**Functions:** `_query_model()`, `SYSTEM_PROMPT`, `USER_TEMPLATE` — `scripts/metadata_extraction.py`

For each figure (skipped only if caption, paragraphs, and referenced tables are all empty), the LLM is called once with a focused two-message prompt:

**System message** (`SYSTEM_PROMPT`): instructs the model to return one JSON object with figure-level fields and a `panels[]` array — one entry per sub-panel, enumerating and **expanding ranges/lists** (`"(b)-(e)"` → b, c, d, e). It defines the accepted values (including `image_kind` and the three `confidence` levels), asks the model to mine referenced tables for `active_region`/`heliographic_location`/timestamps/instrument, and to return a single `"panel": null` entry when the caption has no sub-panels.

**User message** (`USER_TEMPLATE`): embeds the figure label, caption text, body paragraphs, and the caption + raw contents of any linked tables:

```
Extract the figure metadata (with one entry per panel) for the figure described below.
Return ONLY a JSON object.

Figure: Figure 5

Caption:
Fig. 5. A white-light flare in NOAA AR 11515. (a), (b) and (c) are AIA 1600, 171 and 131 Å ...

Body text paragraphs referencing this figure:
- ...

Referenced tables (caption + raw contents):
Table 1: Table 1. Information of the white-light flares detected in NOAA AR 11515 Num Date ... S17W08 ...
```

This is a per-figure call — the LLM sees the whole figure's context at once and enumerates its panels.

Generation uses `max_new_tokens=2048` (room for many panel objects), `temperature=0.1` (near-deterministic for JSON stability), and `do_sample=True`.

### Step 7: Parse LLM output
**Function:** `_parse_llm_output(raw)` — `scripts/metadata_extraction.py`

See [JSON Parsing and Fallback Strategy](#json-parsing-and-fallback-strategy).

The `panels` array is normalised with `_normalize_panel()` (each panel coerced to the full key set, missing keys → `null`, `confidence` → `"low"`). If the model returns no usable panels, a single `"panel": null` entry is written so the figure is never dropped.

### Step 8: Assemble and write the result JSON
**Function:** `_write_result()` — `scripts/metadata_extraction.py`

After all figures are processed, writes one JSON file with the structure described in [Output Format](#output-format). The `status` field is `"success"` if at least one figure record was produced, `"failed"` otherwise.

### Step 9: Print progress and summary
**Implemented in:** `main()` — `scripts/metadata_extraction.py`

After each paper:
```
  [ok]    2012-01 - Labrosse, N  (6 observation(s))
  [skip]  2012-01 - Didkovsky, L
  [fail]  2012-01 - Zhao, J
```

At the end:
```
Summary: 8 processed, 3 skipped, 1 failed  (total: 12)
```

---

## Model Loading and Quantization

**Function:** `load_model(model_name)` — `scripts/metadata_extraction.py`

The model is loaded once before the paper loop and kept in GPU memory for the entire batch. Loading takes approximately 1–2 minutes after the weights are cached locally.

**8-bit quantization via BitsAndBytes:**

```python
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
```

`load_in_8bit=True` uses LLM.int8() quantization: weights are quantized to 8-bit integers while activations remain in float16. This halves VRAM usage compared to full float16 loading (14B × 2 bytes → ~7 GB instead of ~28 GB), with a negligible accuracy penalty for structured extraction tasks.

`device_map="auto"` lets `accelerate` distribute the model layers across available GPU(s) and CPU RAM. On a single 20 GB GPU this places all layers on the GPU.

`model.eval()` disables dropout and batch normalization training modes.

---

## HuggingFace Model Cache

`metadata_extraction.py` redirects the HuggingFace cache to the project-local `models/` directory:

```python
os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parent.parent / "models"),
)
```

This line is placed at module level **before** any `from transformers import ...` statement. `setdefault` is used so a value already set in the environment (e.g. a SLURM job script) takes precedence.

**On-disk layout after first run:**
```
models/
└── hub/
    └── models--Qwen--Qwen2.5-14B-Instruct/
        ├── blobs/           ← actual weight files (~14 GB)
        └── snapshots/
```

**Disk space:** Qwen2.5-14B-Instruct requires ~14 GB on `scratchsan`.

**Offline mode:**
```bash
HF_HUB_OFFLINE=1 ./tools/extract_plots.sh metadata --paper-name "2012-01 - Labrosse, N"
```

---

## Inference Pipeline

### Prompt design

The LLM receives one call per figure, not one call per image or per paper. Each call contains the figure's full context: the figure label, its caption, the body paragraphs that cite it, and the caption + raw contents of any linked tables. The model enumerates the figure's sub-panels and returns a `panels[]` array, so distinct observations (different instrument/wavelength/time) that share one caption are separated into per-panel entries.

The system prompt instructs the model to:
1. Return **only** a valid JSON object (not an array) with no surrounding text.
2. Use `null` for absent fields (not `"N/A"` or empty strings).
3. Emit the figure-level fields plus a `panels[]` array, expanding panel ranges/lists.

The per-panel `confidence` field is defined as:
- `"high"` — explicit Heliprojective Tx/Ty coordinates are stated in the paper
- `"medium"` — a limb position and/or field of view are known, but no explicit coordinates
- `"low"` — only a timestamp or instrument is available

### Token budget

A typical per-figure prompt (figure label + caption + a few body paragraphs) is 200–600 tokens; a linked table's raw contents can add a few hundred more. The `max_new_tokens=2048` cap gives room for a multi-panel figure's `panels[]` array. Both are well within Qwen2.5's 32 768-token context window.

---

## JSON Parsing and Fallback Strategy

**Function:** `_parse_llm_output(raw)` — `scripts/metadata_extraction.py`

The parser expects a single JSON object (not an array). It handles common model deviations in two tiers:

**Tier 1: Strip markdown fences and try direct parse**

```python
cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip("`\n ")
result = json.loads(cleaned)
```

This handles the most common case: the model returns plain JSON or JSON wrapped in ` ```json ... ``` `.

**Tier 2: Regex extraction of the outermost JSON object**

If `json.loads` raises `JSONDecodeError` (e.g. the model added a sentence before the object), the parser searches for the outermost `{...}` block:

```python
match = re.search(r"\{.*\}", cleaned, re.DOTALL)
result = json.loads(match.group())
```

**Tier 3: Graceful empty return**

If both tiers fail, `_parse_llm_output` returns `{}` (an empty dict). The caller fills all LLM-derived fields with `null` and `confidence` defaults to `"low"`. The image record is always written — it is never silently dropped.

---

## Output Format

### Per-paper JSON file

One file per paper, saved as `<root>/metadata/<name>.json`. The **observation unit is the figure**:
each `observations[]` entry is a figure record carrying a `panels[]` array — one entry per sub-panel
(a, b, c…). A figure with no sub-panels yields a single-element `panels[]` with `"panel": null`.

```json
{
  "paper": "2018-06 - Song, Y.pdf",
  "paper_authors": "Song, Y",
  "paper_title": "Observations of white-light flares in NOAA active region 11515 ...",
  "observations": [
    {
      "figure": 5,
      "figure_label": "Fig. 5",
      "caption": "Fig. 5. A white-light flare in NOAA AR 11515. (a), (b) and (c) are AIA 1600, 171 and 131 Å ...",
      "paragraphs": [ "..." ],
      "referenced_tables": [
        { "label": "Table 1", "caption": "Table 1. Information of the white-light flares detected in NOAA AR 11515" }
      ],
      "source_images": [
        { "filename": "img_006_p10_aia_false_color.png", "is_solar": true,  "classifier_score": 0.5 },
        { "filename": "img_005_p10_unknown.png",          "is_solar": false, "classifier_score": 0.1 }
      ],
      "phenomenon": "White-light flare",
      "active_region": "11515",
      "panels": [
        {
          "panel": "a",
          "description": "AIA 1600 Å image at the peak time of the WLF",
          "image_kind": "intensity",
          "instrument": "AIA",
          "wavelength_angstrom": 1600,
          "timestamp_start": null,
          "timestamp_end": null,
          "limb_position": "disk",
          "fov_arcsec": null,
          "center_tx_arcsec": null,
          "center_ty_arcsec": null,
          "heliographic_location": "S18W29",
          "derived_from": null,
          "confidence": "low"
        },
        {
          "panel": "f",
          "description": "difference image (peak - beginning) of HMI continuum",
          "image_kind": "difference",
          "instrument": "HMI",
          "wavelength_angstrom": null,
          "timestamp_start": null,
          "timestamp_end": null,
          "limb_position": "disk",
          "fov_arcsec": null,
          "center_tx_arcsec": null,
          "center_ty_arcsec": null,
          "heliographic_location": "S18W29",
          "derived_from": ["d", "e"],
          "confidence": "low"
        }
      ]
    }
  ],
  "status": "success"
}
```

### Field definitions

**Top-level fields:**

| Field | Description |
|-------|-------------|
| `paper` | PDF filename (e.g. `2018-06 - Song, Y.pdf`) |
| `paper_authors` | Author string from PDF metadata or `first_author` from the extraction log |
| `paper_title` | Paper title from the extraction log |
| `observations` | Array of **figure records** (one per figure grouped from the saved images) |
| `status` | `"success"` if at least one observation was produced; `"failed"` otherwise |

**Figure-record fields:**

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `figure` | integer or `null` | caption label | Leading figure number of the matched caption |
| `figure_label` | string | caption label | e.g. `"Fig. 5"` |
| `caption` | string | PDF caption | Full caption text (for user verification) |
| `paragraphs` | list of strings | PDF body text | Body paragraphs citing this figure (for user verification) |
| `referenced_tables` | list of `{label, caption}` | PDF tables | Tables linked to this figure whose contents were fed to the LLM |
| `source_images` | list of `{filename, is_solar, classifier_score}` | extraction log | The saved rasters grouped into this figure, each with its classifier verdict |
| `phenomenon` | string or `null` | LLM | Figure-level solar structure/event label |
| `active_region` | string or `null` | LLM | NOAA active-region number, e.g. `"11515"` (shared by all panels; often mined from a table) |
| `panels` | list of panel objects | LLM | One entry per sub-panel; see below |

**Per-panel fields:**

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `panel` | string or `null` | LLM | Panel letter (e.g. `"a"`); `null` when the figure has no sub-panels |
| `description` | string or `null` | LLM | One concise phrase describing the panel |
| `image_kind` | string or `null` | LLM | `intensity` / `magnetogram` / `difference` / `ratio` / `lightcurve` / `other`. Only `intensity` and `magnetogram` are fetchable SDO frames; the rest are skipped by `query` |
| `instrument` | string or `null` | LLM | `"AIA"`, `"HMI"`, `"EIT"`, `"LASCO"`, `"XRT"`, etc. |
| `wavelength_angstrom` | integer or `null` | LLM | AIA channel or other wavelength in Ångström |
| `timestamp_start` | ISO UTC string or `null` | LLM | Start of the observation window (caption-level; exact burned-in UT awaits the OCR phase) |
| `timestamp_end` | ISO UTC string or `null` | LLM | End of the observation window |
| `limb_position` | string or `null` | LLM | One of `"NW"`, `"SW"`, `"NE"`, `"SE"`, `"N"`, `"S"`, `"E"`, `"W"`, `"disk"` |
| `fov_arcsec` | `[width, height]` or `null` | LLM | Field of view in arcseconds |
| `center_tx_arcsec` | float or `null` | LLM | Heliprojective Tx of the observation center |
| `center_ty_arcsec` | float or `null` | LLM | Heliprojective Ty of the observation center |
| `heliographic_location` | string or `null` | LLM | Heliographic position, e.g. `"S17W08"` (per panel; may differ) |
| `derived_from` | list of strings or `null` | LLM | For `difference`/`ratio` panels, the source panel letters, e.g. `["d","e"]` |
| `confidence` | `"high"` / `"medium"` / `"low"` | LLM (defaults to `"low"`) | Drives the SDO query strategy in the `query` stage |

The `caption`, `paragraphs`, `referenced_tables`, and `source_images` fields let the user verify the
source that informed each figure's metadata. Keeping a figure's `panels[]` together preserves the
panel *sequence* as a unit (useful for spatiotemporal models and as a VLM training target).
`active_region` / `heliographic_location` give the downstream image-matching step a way to narrow the
disk search to a known region and timestamp.

---

## End-to-End Example

```bash
# 1. Activate the conda environment
conda activate pytorch_jupyter

# 2. Run metadata extraction on the Labrosse paper
./tools/extract_plots.sh metadata --paper-name "2012-01 - Labrosse, N"
# Output:
#   Found 1 paper(s) to process
#   Loading tokenizer: Qwen/Qwen2.5-14B-Instruct
#   Loading model with 8-bit quantization …
#   Model loaded.
#     [ok]    2012-01 - Labrosse, N  (6 observation(s))
#
#   Summary: 1 processed, 0 skipped, 0 failed  (total: 1)

# 3. Inspect the output
cat "output/metadata/2012-01 - Labrosse, N.json"

# 4. Process all papers at once
./tools/extract_plots.sh metadata --all

# 5. Hand off the metadata to the query stage
./tools/extract_plots.sh query --all

# 6. Retry a paper by deleting its JSON and re-running
rm "output/metadata/2012-01 - Labrosse, N.json"
./tools/extract_plots.sh metadata --paper-name "2012-01 - Labrosse, N"
```

---

## Known Limitations

1. **Scanned PDFs produce no text:** PyMuPDF can only extract text from PDFs with an embedded text layer. Papers distributed as scanned images (rare in modern astrophysics journals) will produce empty captions and paragraphs. Those images are skipped by the LLM (no context), so they contribute no observation.

2. **One LLM call per figure:** The pipeline calls the LLM once per figure (fewer calls than the old per-image model). Papers with many figures take proportionally longer. Each call is independent — there is no shared context across figures.

3. **Panels are enumerated from the caption, not the pixels:** Sub-panels (a, b, c…) and their per-panel instrument/wavelength/role come from parsing the caption text into `panels[]`. The exact burned-in timestamp on each panel is **not** read yet — per-panel `timestamp_start` is caption-level (often `null`); precise per-panel UT awaits a later OCR / vision phase. Panels are also not yet cropped to individual images (the `source_images` rasters are referenced whole).

4. **Figure citation matching by number:** `extract_figure_body_refs()` matches body paragraphs to figures by the number in the inline citation (e.g. `"Fig. 2"`). Non-standard citation styles (roman numerals, letters only, unnumbered figures) will produce an empty `paragraphs` list, leaving the LLM with only the caption text.

5. **Caption matching falls back to adjacent pages:** `match_image_to_caption()` searches the same page first, then the next and previous pages. In papers where figure captions are placed far from their images (e.g. end-of-paper caption lists), the matched caption may belong to a different figure.

6. **Model hallucination:** The LLM may invent timestamps or coordinates not present in the paper text. The `caption` and `paragraphs` fields are included in the output specifically to allow manual spot-checking against the source text.

7. **JSON instability:** Even at `temperature=0.1`, the model occasionally produces malformed JSON. When both parsing tiers fail, the figure gets a single `"panel": null` entry with all LLM fields `null`. Inspect the `caption` and `paragraphs` fields in the output to understand what context was available.

8. **First-run model download:** The ~14 GB Qwen2.5-14B-Instruct weights are downloaded from HuggingFace Hub on first run and cached in `models/hub/`. Ensure `scratchsan` has at least 16 GB free before the first run.
```
