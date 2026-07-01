# METADATA_EXTRACTION — Pipeline Documentation

Step-by-step explanation of the automated solar observation metadata extraction pipeline (Stage 1: `metadata`).

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pipeline Structure](#pipeline-structure)
4. [Stage 1: Extract Metadata (`metadata`)](#stage-1-extract-metadata)
5. [Model Loading and Quantization](#model-loading-and-quantization)
6. [HuggingFace Model Cache](#huggingface-model-cache)
7. [Inference Pipeline](#inference-pipeline)
8. [JSON Parsing and Fallback Strategy](#json-parsing-and-fallback-strategy)
9. [Output Format](#output-format)
10. [End-to-End Example](#end-to-end-example)
11. [Known Limitations](#known-limitations)

---

## Overview

The `metadata` stage processes paper directories produced by the `extract` stage and uses a locally-hosted large language model (Qwen2.5-14B-Instruct, 8-bit quantized) to extract structured observation metadata. Unlike a whole-paper approach, the pipeline works **per solar image**: each extracted image is matched to its nearest figure caption and the body paragraphs that cite it, and the LLM is asked to extract observation metadata from that focused context alone. The result is one JSON file per paper listing every solar observation image with its metadata.

```
output/papers/
    └── 2012-01 - Labrosse, N/
        ├── extraction_log.json     ← produced by the extract stage
        ├── paper_2620529.pdf
        ├── solar_001_p3_aia_false_color.png
        └── solar_002_p3_aia_false_color.png
                    │
              (metadata stage)
                    │
                    ▼
output/metadata/
    └── 2012-01 - Labrosse, N.json
```

The stage is fully resumable: re-running skips any paper whose JSON file already exists in the output directory.

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

### 3. HuggingFace Transformers (already required by `label`)

```bash
pip install transformers>=4.30.0
```

### 4. PyMuPDF (already required by `extract` and `label`)

Used for caption extraction and image bounding-box lookup:

```bash
pip install pymupdf
```

### 5. HuggingFace Hub access (first run only)

The Qwen2.5-14B-Instruct model weights (~14 GB) are downloaded from HuggingFace Hub on the first run and cached locally in the project's `models/` directory (see [HuggingFace Model Cache](#huggingface-model-cache)). Subsequent runs use the cache.

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
└── extract_plots.sh                 # Shell entry point (validates env, dispatches)

scripts/
├── stage1_metadata_extraction.py    # Stage 1 orchestrator
└── label_plots.py                   # Shares caption-matching logic (same approach)

utils/
├── caption_extractor.py             # extract_all_captions(), get_all_image_bboxes(),
│                                    # match_image_to_caption(), extract_figure_body_refs()
└── structure_classifier_nlp.py      # classify_structure() — NLP fallback for phenomenon/confidence

models/                              # HuggingFace model cache (HF_HOME)
└── hub/
    ├── models--Qwen--Qwen2.5-14B-Instruct/
    └── models--facebook--bart-large-mnli/
```

The shell script validates that `bitsandbytes` and `transformers` are importable before dispatching to `scripts/stage1_metadata_extraction.py`. All PDF text extraction is handled by `utils/caption_extractor.py` (the same utility used by `label_plots.py`).

---

## Stage 1: Extract Metadata

**Command (single paper):**
```bash
./tools/extract_plots.sh metadata \
    --paper_dir "output/papers/2012-01 - Labrosse, N" \
    --output_dir output/metadata/
```

**Command (all papers):**
```bash
./tools/extract_plots.sh metadata \
    --pdf_dir output/papers/ \
    --output_dir output/metadata/
```

**Implemented in:** `scripts/stage1_metadata_extraction.py`

**What it does, step by step:**

### Step 1: Collect paper directories
**Implemented in:** `main()` — `scripts/stage1_metadata_extraction.py`

- With `--paper_dir`: the single given directory is used directly.
- With `--pdf_dir`: the script calls `os.scandir()` on the given parent directory and collects all subdirectories that contain an `extraction_log.json`. This mirrors how `label_plots.py` discovers paper folders and avoids accidentally processing unrelated directories.

### Step 2: Check for existing output (resumability)
**Implemented in:** `process_paper_dir()` — `scripts/stage1_metadata_extraction.py`

The output JSON path is derived as `<output_dir>/<folder_name>.json` (where `folder_name` is the paper directory's basename, e.g. `2012-01 - Labrosse, N`). If that file already exists, `process_paper_dir()` returns `"skipped"` immediately without loading the model or reading any files. This makes the stage safe to re-run after a crash or partial batch.

### Step 3: Load extraction log and find the PDF
**Functions:** `_load_extraction_log()`, `_find_pdf()` — `scripts/stage1_metadata_extraction.py`

Reads `extraction_log.json` from the paper directory for the solar image list, paper title, and first author. Also finds `paper_*.pdf` inside the same directory for the full-text extraction steps below.

The author string is resolved in order of priority:
1. PDF document metadata (`fitz.open(pdf_path).metadata["author"]`)
2. `first_author` field from `extraction_log.json`

### Step 4: Extract captions, image bboxes, and body references from the PDF
**Functions:** `extract_all_captions()`, `get_all_image_bboxes()`, `extract_figure_body_refs()` — `utils/caption_extractor.py`

Three separate passes over the PDF are run once per paper (not once per image):

1. **`extract_all_captions(pdf_path)`** — scans every text block across all pages and identifies figure captions by the pattern `^(Figure|Fig\.)\s*\d+[a-z]?[.:\s]`. Returns a page-keyed dict of `Caption` objects (each with `figure_label`, full `text`, 1-based page number, and bounding box). Multi-block captions (captions split across consecutive text blocks) are merged automatically.

2. **`get_all_image_bboxes(pdf_path)`** — scans every page for embedded images (using `page.get_images(full=True)`), deduplicating by xref, and returns the bounding rectangle (`fitz.Rect`) of each image in global encounter order. This order matches the `index` field in `extraction_log.json`, so an image's bbox can be looked up directly by its index.

3. **`extract_figure_body_refs(pdf_path)`** — scans all non-caption text blocks and finds paragraphs that contain an inline figure citation such as `"Fig. 2"`, `"Figure 2a"`, or `"Figs. 2 and 3"` (matched by `\b(?:Figs?\.?\s*|Figures?\s+)(\d+[a-zA-Z]?)`). Returns a dict mapping figure number string (e.g. `"2"`, `"2a"`) to a deduplicated list of paragraph texts in document order.

### Step 5: Match each solar image to its caption and body paragraphs
**Functions:** `match_image_to_caption()` — `utils/caption_extractor.py`; `_build_global_bbox_list()` — `scripts/stage1_metadata_extraction.py`

For every solar image in `extraction_log.json` (iterated in global index order, same as `label_plots.py`):

1. Look up the image's bounding rectangle using its global index into the flattened bbox list.
2. Call `match_image_to_caption(page, img_rect, captions_by_page)` to find the nearest caption by vertical gap. The search strategy is:
   - **Same page first** — pick the caption with the smallest vertical distance from the image edges.
   - **Adjacent pages** — if no caption is on the same page, check the next page then the previous.
   - **None** — if no caption is found at all, caption fields are left empty.
3. Extract the trailing figure number from the caption label (e.g. `"Figure 2a"` → `"2a"`) and look it up in the body refs dict to get the paragraphs that reference this figure.

The result for each image is:
- `caption_text` — the full caption string
- `figure_label` — e.g. `"Figure 2"`
- `paragraphs` — list of body paragraph strings that cite this figure

### Step 6: NLP classification (fallback)
**Function:** `classify_structure(caption_text)` — `utils/structure_classifier_nlp.py`

Runs zero-shot classification (facebook/bart-large-mnli) on the caption text to produce a `phenomenon` label and a numeric `confidence` score. These values are only used as fallbacks: if the LLM returns a `phenomenon` or `confidence` field, those take precedence; the NLP result fills in the gap only when the LLM leaves those fields absent or null.

### Step 7: Build prompt and query LLM
**Functions:** `_query_model()`, `SYSTEM_PROMPT`, `USER_TEMPLATE` — `scripts/stage1_metadata_extraction.py`

For each solar image (skipped if both caption and paragraphs are empty), the LLM is called with a focused two-message prompt:

**System message** (`SYSTEM_PROMPT`): instructs the model to return a single JSON object with exactly ten fields, using `null` for absent information. Defines the accepted values for each field and the three `confidence` levels.

**User message** (`USER_TEMPLATE`): embeds the figure label, caption text, and body paragraphs for this specific image:

```
Extract the observation metadata for the solar image described below.
Return ONLY a JSON object.

Figure: Figure 2

Caption:
Fig. 2. Top: Evolution of the 2010-06-13 prominence eruption at 304 Å ...

Body text paragraphs referencing this figure:
- Beginning at 11:10 UT, AIA observed a large kinking loop eruption from AR 11171 ...
- Fig. 2 shows the same information only for the 2011-03-19 event ...
```

This is a per-image call, not a whole-paper call — the LLM sees only the context relevant to the single image being processed.

Generation uses `max_new_tokens=512` (sufficient for one JSON object), `temperature=0.1` (near-deterministic for JSON stability), and `do_sample=True`.

### Step 8: Parse LLM output
**Function:** `_parse_llm_output(raw)` — `scripts/stage1_metadata_extraction.py`

See [JSON Parsing and Fallback Strategy](#json-parsing-and-fallback-strategy).

If parsing fails (returns `{}`), all LLM-derived fields for that image are `null`, but the NLP fallback still fills `phenomenon` and `confidence`. The image is never dropped.

### Step 9: Assemble and write the result JSON
**Function:** `_write_result()` — `scripts/stage1_metadata_extraction.py`

After all solar images are processed, writes one JSON file with the structure described in [Output Format](#output-format). The `status` field is `"success"` if at least one observation was produced, `"failed"` otherwise.

### Step 10: Print progress and summary
**Implemented in:** `main()` — `scripts/stage1_metadata_extraction.py`

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

**Function:** `load_model(model_name)` — `scripts/stage1_metadata_extraction.py`

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

Both `stage1_metadata_extraction.py` (Qwen2.5-14B-Instruct) and `utils/structure_classifier_nlp.py` (facebook/bart-large-mnli) redirect the HuggingFace cache to the project-local `models/` directory:

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
    ├── models--Qwen--Qwen2.5-14B-Instruct/
    │   ├── blobs/           ← actual weight files (~14 GB)
    │   └── snapshots/
    └── models--facebook--bart-large-mnli/
        ├── blobs/
        └── snapshots/
```

**Disk space:** Qwen2.5-14B-Instruct requires ~14 GB; facebook/bart-large-mnli requires ~1.6 GB. Total ~16 GB on `scratchsan`.

**Offline mode:**
```bash
HF_HUB_OFFLINE=1 ./tools/extract_plots.sh metadata --paper_dir ...
```

---

## Inference Pipeline

### Prompt design

The LLM receives one call per solar image, not one call per paper. Each call contains only the context for that specific image: the figure label, its full caption, and the body paragraphs that explicitly cite it. This focused context prevents the model from confusing observations from different figures and makes it easier to extract per-image details such as the specific AIA channel shown.

The system prompt instructs the model to:
1. Return **only** a valid JSON object (not an array) with no surrounding text.
2. Use `null` for absent fields (not `"N/A"` or empty strings).
3. Use exactly the ten defined field names.

The `confidence` field is defined as:
- `"high"` — explicit Heliprojective Tx/Ty coordinates are stated in the paper
- `"medium"` — a limb position and/or field of view are known, but no explicit coordinates
- `"low"` — only a timestamp or instrument is available

### Token budget

A typical per-image prompt (figure label + caption + a few body paragraphs) is 200–600 tokens. The `max_new_tokens=512` cap is sufficient for a single JSON object with ten fields. Both are well within Qwen2.5's 32 768-token context window.

---

## JSON Parsing and Fallback Strategy

**Function:** `_parse_llm_output(raw)` — `scripts/stage1_metadata_extraction.py`

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

If both tiers fail, `_parse_llm_output` returns `{}` (an empty dict). The caller fills all LLM-derived fields with `null` and the NLP fallback (`classify_structure`) covers `phenomenon` and `confidence`. The image record is always written — it is never silently dropped.

---

## Output Format

### Per-paper JSON file

One file per paper, saved as `<output_dir>/<folder_name>.json`:

```json
{
  "paper": "paper_2620529.pdf",
  "paper_authors": "Labrosse, N.",
  "paper_title": "Plasma diagnostic in eruptive prominences from SDO/AIA observations at 304 Å",
  "observations": [
    {
      "observation_filename": "solar_001_p3_aia_false_color",
      "figure": 2,
      "caption": "Fig. 2. Top: Evolution of the 2010-06-13 prominence eruption at 304 Å ...",
      "paragraphs": [
        "Beginning at 11:10 UT, AIA observed a large kinking loop eruption from AR 11171 ...",
        "Fig. 2 shows the same information only for the 2011-03-19 event ..."
      ],
      "timestamp_start": "2010-06-13T11:10:00",
      "timestamp_end": "2010-06-13T11:40:00",
      "instrument": "AIA",
      "wavelength_angstrom": 304,
      "limb_position": "NW",
      "fov_arcsec": [400.0, 400.0],
      "center_tx_arcsec": -750.0,
      "center_ty_arcsec": 300.0,
      "phenomenon": "Prominence",
      "confidence": "high"
    },
    {
      "observation_filename": "solar_002_p3_aia_false_color",
      "figure": 2,
      "caption": "Fig. 2. Top: Evolution of the 2010-06-13 prominence eruption at 304 Å ...",
      "paragraphs": [
        "Beginning at 11:10 UT, AIA observed a large kinking loop eruption from AR 11171 ..."
      ],
      "timestamp_start": "2010-06-13T11:25:00",
      "timestamp_end": null,
      "instrument": "AIA",
      "wavelength_angstrom": 171,
      "limb_position": "NW",
      "fov_arcsec": null,
      "center_tx_arcsec": null,
      "center_ty_arcsec": null,
      "phenomenon": "Prominence",
      "confidence": "medium"
    }
  ],
  "status": "success"
}
```

### Field definitions

**Top-level fields:**

| Field | Description |
|-------|-------------|
| `paper` | PDF filename (e.g. `paper_2620529.pdf`) |
| `paper_authors` | Author string from PDF metadata or `first_author` from the extraction log |
| `paper_title` | Paper title from the extraction log |
| `observations` | Array of per-image observation objects (one entry per solar image) |
| `status` | `"success"` if at least one observation was produced; `"failed"` otherwise |

**Per-observation fields:**

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `observation_filename` | string | extraction log | Stem of the image file, e.g. `solar_001_p3_aia_false_color` |
| `figure` | integer or `null` | caption label | Leading figure number extracted from the matched caption label |
| `caption` | string | PDF caption | Full text of the matched figure caption (for user verification) |
| `paragraphs` | list of strings | PDF body text | Body paragraphs that explicitly cite this figure (for user verification) |
| `timestamp_start` | ISO UTC string or `null` | LLM | Start of the observation window |
| `timestamp_end` | ISO UTC string or `null` | LLM | End of the observation window |
| `instrument` | string or `null` | LLM | `"AIA"`, `"HMI"`, `"EIT"`, `"LASCO"`, `"XRT"`, etc. |
| `wavelength_angstrom` | integer or `null` | LLM | AIA channel or other wavelength in Ångström |
| `limb_position` | string or `null` | LLM | One of `"NW"`, `"SW"`, `"NE"`, `"SE"`, `"N"`, `"S"`, `"E"`, `"W"`, `"disk"` |
| `fov_arcsec` | `[width, height]` or `null` | LLM | Field of view in arcseconds |
| `center_tx_arcsec` | float or `null` | LLM | Heliprojective Tx of the observation center |
| `center_ty_arcsec` | float or `null` | LLM | Heliprojective Ty of the observation center |
| `phenomenon` | string | LLM, then NLP | Solar structure/event label; NLP (`classify_structure`) used as fallback |
| `confidence` | `"high"` / `"medium"` / `"low"` | LLM, then NLP | Drives SDO query strategy in Stage 2 |

The `caption` and `paragraphs` fields are included so the user can inspect the source text that informed each observation's metadata and verify that the LLM extracted from the correct context.

---

## End-to-End Example

```bash
# 1. Activate the conda environment
conda activate pytorch_jupyter

# 2. Run metadata extraction on the Labrosse paper folder
./tools/extract_plots.sh metadata \
    --paper_dir "output/papers/2012-01 - Labrosse, N" \
    --output_dir output/metadata/
# Output:
#   Found 1 paper director(ies) to process
#   Loading tokenizer: Qwen/Qwen2.5-14B-Instruct
#   Loading model with 8-bit quantization …
#   Model loaded.
#     [ok]    2012-01 - Labrosse, N  (6 observation(s))
#
#   Summary: 1 processed, 0 skipped, 0 failed  (total: 1)

# 3. Inspect the output
cat "output/metadata/2012-01 - Labrosse, N.json"

# 4. Process all papers at once
./tools/extract_plots.sh metadata \
    --pdf_dir output/papers/ \
    --output_dir output/metadata/

# 5. Hand off the metadata to Stage 2
./tools/extract_plots.sh query \
    --metadata_dir output/metadata/ \
    --fits_dir output/fits/ \
    --output_dir output/matched/

# 6. Retry a paper by deleting its JSON and re-running
rm "output/metadata/2012-01 - Labrosse, N.json"
./tools/extract_plots.sh metadata \
    --paper_dir "output/papers/2012-01 - Labrosse, N" \
    --output_dir output/metadata/
```

---

## Known Limitations

1. **Scanned PDFs produce no text:** PyMuPDF can only extract text from PDFs with an embedded text layer. Papers distributed as scanned images (rare in modern astrophysics journals) will produce empty captions and paragraphs. Those images will still be written to the output JSON, but all LLM-derived fields will be `null` and `phenomenon`/`confidence` will come from the NLP model alone.

2. **One LLM call per image:** The pipeline calls the LLM once for every solar image. Papers with many solar images (e.g. 20+) take proportionally longer. Unlike the previous whole-paper approach, there is no shared context across images — each call is independent.

3. **Multiple images per figure get the same context:** If several solar images belong to the same figure (e.g. sub-panels a, b, c), they all receive the same caption and body paragraphs. The LLM cannot distinguish which sub-panel it is looking at and may return identical or ambiguous metadata for those images. Sub-panel identification would require passing the actual image content to a vision model.

4. **Figure citation matching by number:** `extract_figure_body_refs()` matches body paragraphs to figures by the number in the inline citation (e.g. `"Fig. 2"`). Non-standard citation styles (roman numerals, letters only, unnumbered figures) will produce an empty `paragraphs` list, leaving the LLM with only the caption text.

5. **Caption matching falls back to adjacent pages:** `match_image_to_caption()` searches the same page first, then the next and previous pages. In papers where figure captions are placed far from their images (e.g. end-of-paper caption lists), the matched caption may belong to a different figure.

6. **Model hallucination:** The LLM may invent timestamps or coordinates not present in the paper text. The `caption` and `paragraphs` fields are included in the output specifically to allow manual spot-checking against the source text.

7. **JSON instability:** Even at `temperature=0.1`, the model occasionally produces malformed JSON. When both parsing tiers fail, all LLM fields are `null` for that image. Inspect the `caption` and `paragraphs` fields in the output to understand what context was available.

8. **First-run model download:** The ~14 GB Qwen2.5-14B-Instruct weights are downloaded from HuggingFace Hub on first run and cached in `models/hub/`. Ensure `scratchsan` has at least 16 GB free before the first run.
