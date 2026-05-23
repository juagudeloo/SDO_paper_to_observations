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

The `metadata` stage processes a directory of solar physics PDFs and uses a locally-hosted large language model (Qwen2.5-14B-Instruct, 8-bit quantized) to extract structured observation metadata from each paper's text. The output is a JSON file per paper listing every solar observation event described in that paper, with enough information to query the SDO archive in Stage 2.

```
papers/raw_pdfs/
    ├── paper_A.pdf
    └── paper_B.pdf
              │
        (metadata stage)
              │
              ▼
papers/metadata/
    ├── paper_A.json      ← {"paper": ..., "observations": [...], "status": "success"}
    ├── paper_B.json
    └── failed/           ← raw model output saved for debugging
        └── paper_C_raw.txt
```

The stage is fully resumable: re-running skips any paper that already has a corresponding JSON file in the output directory.

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

Verify:
```python
import bitsandbytes; print(bitsandbytes.__version__)
```

### 3. HuggingFace Transformers (already required by `label`)

```bash
pip install transformers>=4.30.0
```

### 4. PyMuPDF (already required by `extract` and `label`)

Used for per-figure context extraction from PDFs:

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
└── stage1_metadata_extraction.py    # Stage 1 orchestrator

utils/
├── pdf_extractor.py                 # extract_full_text() — fallback raw-text extraction
└── caption_extractor.py             # extract_all_captions(), extract_figure_body_refs(),
                                     # build_figure_contexts() — per-figure context builder

models/                              # HuggingFace model cache (HF_HOME)
└── hub/
    └── models--Qwen--Qwen2.5-14B-Instruct/
```

The shell script validates that `bitsandbytes` and `transformers` are importable before dispatching to `scripts/stage1_metadata_extraction.py`. PDF context extraction is handled by two utilities:

- **Preferred path** — `build_figure_contexts()` in `utils/caption_extractor.py`: extracts figure captions and the body paragraphs that cite each figure, giving the LLM structured, observation-rich context.
- **Fallback path** — `extract_full_text()` in `utils/pdf_extractor.py`: raw page-concatenated text, used only when no figure captions are found (e.g., unusual PDF structure).

---

## Stage 1: Extract Metadata

**Command:**
```bash
./tools/extract_plots.sh metadata \
    --paper-dir "output/papers/2012-01 - Labrosse, N" \
    --output_dir output/metadata/
```

**Implemented in:** `scripts/stage1_metadata_extraction.py` (orchestrator) + `utils/caption_extractor.py` (per-figure context) + `utils/pdf_extractor.py` (fallback text).

**What it does, step by step:**

### Step 1: Check for existing output (resumability)
**Implemented in:** `process_pdf()` — `scripts/stage1_metadata_extraction.py`

Before doing any work for a paper, `process_pdf()` derives the output JSON path as `<output_dir>/<stem>.json` (where `stem` is the PDF filename without extension). If that file already exists on disk, the function returns `"skipped"` immediately without loading the model or reading the PDF. This makes the stage safe to re-run after a crash or timeout.

### Step 2: Extract per-figure context from PDF
**Function:** `extract_pdf_context(pdf_path, max_chars=12000)` — `scripts/stage1_metadata_extraction.py`

This is the main text-extraction step. It follows a two-path strategy:

**Preferred path — per-figure context blocks:**

Calls `build_figure_contexts(pdf_path)` from `utils/caption_extractor.py`, which internally runs two passes:

1. `extract_all_captions(pdf_path)` — scans every text block across all PDF pages and identifies figure captions by the pattern `^(Figure|Fig\.)\s*\d+[a-z]?[.:\s]`. Returns a page-keyed dict of `Caption` objects (each with `figure_label`, full `text`, page number, and bounding box). Multi-block captions (captions split across consecutive text blocks) are merged automatically.

2. `extract_figure_body_refs(pdf_path)` — scans all *non-caption* text blocks and finds paragraphs that contain an inline figure citation such as `"Fig. 2"`, `"Figure 2a"`, or `"Figs. 2 and 3"` (matched by `\b(?:Figs?\.?\s*|Figures?\s+)(\d+[a-zA-Z]?)`). Returns a dict mapping figure number (e.g. `"2"`, `"2a"`) to a deduplicated list of paragraph texts in document order.

`build_figure_contexts()` then joins the two: for each caption it extracts the trailing number from the figure label (e.g. `"Figure 2"` → `"2"`), looks up the matching body paragraphs, and returns a list of dicts:

```python
{
    "figure_label": "Figure 2",
    "caption": "Fig. 2. Top: Evolution of the 2010-06-13 prominence eruption ...",
    "body_refs": [
        "Beginning at 11:10 UT, AIA observed a large kinking loop eruption from AR 11171 ...",
        "Fig. 2 shows the same information only for the 2011-03-19 event ...",
    ]
}
```

`_format_figure_contexts(contexts, max_chars=12000)` serializes these dicts into the prompt string:

```
[Figure 2]
Caption: Fig. 2. Top: Evolution of the 2010-06-13 prominence eruption ...
Body text references:
  - Beginning at 11:10 UT, AIA observed a large kinking loop eruption from AR 11171 ...
  - Fig. 2 shows the same information only for the 2011-03-19 event ...

[Figure 4]
Caption: Fig. 4. Same as Fig. 2 for the 2011-03-19 event ...
Body text references:
  - ...
```

This design is motivated by a key observation about solar physics papers: figure captions are brief labels ("Evolution of the 2010-06-13 prominence eruption"), while the observation details (instrument, time range, field of view, heliographic position) are described in the surrounding body text. The `label` stage already extracts captions; this stage extends that by also pulling the body paragraphs that give those captions their physical meaning.

**Fallback path — raw full text:**

If `build_figure_contexts()` returns an empty list (no figure captions parseable) or raises an exception, `extract_pdf_context()` falls back to `extract_full_text(pdf_path, max_chars=12000)` from `utils/pdf_extractor.py`. This concatenates page text in order up to `max_chars` characters.

In either case, if the resulting string is empty (scanned PDF, no text layer), `process_pdf()` records a `"failed"` status without calling the model.

### Step 3: Build the chat prompt
**Function:** `build_messages(text)` — `scripts/stage1_metadata_extraction.py`

Returns a two-element list of `{"role": ..., "content": ...}` dicts for Qwen's chat template:

- **System message** (`SYSTEM_PROMPT` constant): instructs the model to return a JSON array only, with no markdown and no prose. Defines the ten required fields and their types, and specifies the three `confidence` levels.
- **User message** (`USER_TEMPLATE` constant): embeds the per-figure context text between `--- PAPER FIGURES ---` delimiters. The template preamble tells the model that each block contains a figure caption followed by body-text passages with the observational detail.

The `confidence` field is defined in the prompt as:
- `"high"` — explicit Heliprojective Tx/Ty coordinates are stated in the paper
- `"medium"` — a limb position and/or field of view are known, but no explicit coordinates
- `"low"` — only a timestamp is reliable

This field drives the strategy selection in Stage 2.

### Step 4: Run inference
**Function:** `query_model(tokenizer, model, messages)` — `scripts/stage1_metadata_extraction.py`

1. `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` — converts the chat messages list into the model's expected text format, including the special tokens that mark the assistant's turn.
2. `tokenizer(prompt, return_tensors="pt").to(model.device)` — tokenizes and moves to the GPU.
3. `model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)` — generates up to 2 048 new tokens. Temperature 0.1 keeps the output near-deterministic (important for JSON stability) while still allowing `do_sample=True` to avoid degenerate repetition.
4. The input token IDs are sliced off the output (`output_ids[0][input_len:]`) before decoding, so the returned string contains only the model's new text — not the prompt echoed back.

### Step 5: Parse the model output
**Function:** `parse_model_output(raw)` — `scripts/stage1_metadata_extraction.py`

See [JSON Parsing and Fallback Strategy](#json-parsing-and-fallback-strategy) for the full algorithm.

### Step 6: Write the result JSON
**Function:** `_write_result(path, paper, observations, status)` — `scripts/stage1_metadata_extraction.py`

On success, writes:
```json
{
  "paper": "filename.pdf",
  "observations": [ ... ],
  "status": "success"
}
```

On failure (JSON parse error, empty text, or any other exception), saves the raw model output to `<output_dir>/failed/<stem>_raw.txt` (so it can be inspected manually) and writes:
```json
{
  "paper": "filename.pdf",
  "observations": [],
  "status": "failed"
}
```

Writing even a failed record to the output directory means the paper is treated as "already processed" on the next run. To retry a failed paper, delete its JSON file before re-running.

### Step 7: Print progress and summary
**Implemented in:** `main()` — `scripts/stage1_metadata_extraction.py`

After each paper, prints one line:
```
  [ok]    paper_A.pdf  (4 event(s))
  [skip]  paper_B.pdf
  [fail]  paper_C.pdf
```

At the end, prints a summary:
```
Summary: 8 processed, 3 skipped, 1 failed  (total PDFs: 12)
```

---

## Model Loading and Quantization

**Function:** `load_model(model_name)` — `scripts/stage1_metadata_extraction.py`

The model is loaded once per run (before the PDF loop) and kept in GPU memory for the entire batch. Loading takes approximately 3–5 minutes on first run (downloading weights) and 1–2 minutes on subsequent runs.

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

`load_in_8bit=True` uses LLM.int8() quantization (from the `bitsandbytes` library): weights are quantized to 8-bit integers while activations remain in float16. This halves VRAM usage compared to full float16 loading (14B × 2 bytes → ~7 GB instead of ~28 GB), with a very small accuracy penalty that is negligible for structured extraction tasks.

`device_map="auto"` lets `accelerate` decide how to distribute the model layers across available GPU(s) and CPU RAM. On a single 20 GB GPU this places all layers on the GPU.

`trust_remote_code=True` is required because Qwen uses custom model code hosted on HuggingFace.

`model.eval()` disables dropout and batch normalization training modes, making inference deterministic given the same input.

---

## HuggingFace Model Cache

Both `stage1_metadata_extraction.py` (Qwen2.5-14B-Instruct) and `utils/structure_classifier_nlp.py` (facebook/bart-large-mnli, used by the `label` stage) redirect the HuggingFace cache away from the default `~/.cache/huggingface/` to the project-local `models/` directory:

```python
os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parent.parent / "models"),
)
```

This line is placed at module level **before** any `from transformers import ...` statement, so the `transformers` library picks up the path at initialisation time.

`setdefault` is used instead of a hard assignment so that a value already set in the environment (e.g. via a SLURM job script) takes precedence. The path is derived from `__file__` rather than the current working directory, so the redirection works regardless of where the script is invoked from.

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

**Offline mode:** Once downloaded, set `HF_HUB_OFFLINE=1` to prevent any network access:
```bash
HF_HUB_OFFLINE=1 ./tools/extract_plots.sh metadata --paper-dir ...
```

---

## Inference Pipeline

### Prompt design

The system prompt instructs the model to:
1. Return **only** a valid JSON array with no surrounding text.
2. Use `null` for missing fields (not `"N/A"` or empty strings).
3. Return an empty array `[]` if no observation events are found.

The user message tells the model the input is structured as one block per figure, with the caption label followed by the body-text passages that reference it. This separates the brief figure label from the observational detail in the main text, making it easier for the model to assign timestamps, instruments, and positions to the correct events.

### Token budget

With 12 000 characters of per-figure context (~3 000 tokens at ~4 chars/token) plus the system prompt (~400 tokens), the total input is comfortably within Qwen2.5's 32 768-token context window. The `max_new_tokens=2048` cap on the output allows for up to ~20–30 observation events before being truncated.

---

## JSON Parsing and Fallback Strategy

**Function:** `parse_model_output(raw)` — `scripts/stage1_metadata_extraction.py`

Despite the strict system prompt, language models occasionally wrap their output in markdown code fences or add a short preamble. The parser handles this in three tiers:

**Tier 1: Strip markdown fences and try direct parse**

```python
cleaned = re.sub(r"```(?:json)?\s*", "", raw)
cleaned = cleaned.strip("`\n ")
result = json.loads(cleaned)
```

This handles the most common case: the model returns plain JSON or JSON wrapped in ` ```json ... ``` `. After stripping, `json.loads` parses the whole string.

If the result is a dict (the model occasionally returns `{"observations": [...]}`), the parser checks keys `"observations"`, `"events"`, and `"data"` in that order and returns the first list-valued field found.

**Tier 2: Regex extraction of the outermost JSON array**

If `json.loads` raises `JSONDecodeError` (e.g., the model added a sentence before the array), the parser searches for the outermost `[...]` block:

```python
match = re.search(r"\[.*\]", cleaned, re.DOTALL)
if match:
    result = json.loads(match.group())
```

`re.DOTALL` makes `.` match newlines, so multi-line JSON arrays are captured correctly.

**Tier 3: Failure**

If both tiers fail, `parse_model_output` raises `ValueError`. `process_pdf()` catches this, saves the raw output to `failed/<stem>_raw.txt`, and records `status: "failed"` in the output JSON.

---

## Output Format

### Per-paper JSON file

One file per paper, saved as `<output_dir>/<pdf_stem>.json`:

```json
{
  "paper": "2012-01 - Labrosse, N.pdf",
  "observations": [
    {
      "timestamp_start": "2010-09-19T08:00:00",
      "timestamp_end": "2010-09-19T10:30:00",
      "instrument": "AIA",
      "wavelength_angstrom": 304,
      "limb_position": "SW",
      "fov_arcsec": [400.0, 400.0],
      "center_tx_arcsec": -550.0,
      "center_ty_arcsec": -250.0,
      "phenomenon": "prominence eruption",
      "confidence": "high"
    },
    {
      "timestamp_start": "2010-09-19T08:00:00",
      "timestamp_end": null,
      "instrument": "HMI",
      "wavelength_angstrom": null,
      "limb_position": null,
      "fov_arcsec": null,
      "center_tx_arcsec": null,
      "center_ty_arcsec": null,
      "phenomenon": "photospheric magnetic field context",
      "confidence": "low"
    }
  ],
  "status": "success"
}
```

### Field definitions

| Field | Type | Description |
|-------|------|-------------|
| `timestamp_start` | ISO UTC string | Start of the observation window (always present) |
| `timestamp_end` | ISO UTC string or `null` | End of the observation window if stated |
| `instrument` | string or `null` | `"AIA"`, `"HMI"`, `"EIT"`, `"LASCO"`, etc. |
| `wavelength_angstrom` | integer or `null` | AIA channel or other wavelength in Ångström |
| `limb_position` | string or `null` | One of `"NW"`, `"SW"`, `"NE"`, `"SE"`, `"N"`, `"S"`, `"E"`, `"W"`, `"disk"` |
| `fov_arcsec` | `[width, height]` or `null` | Field of view in arcseconds |
| `center_tx_arcsec` | float or `null` | Heliprojective Tx of the observation center |
| `center_ty_arcsec` | float or `null` | Heliprojective Ty of the observation center |
| `phenomenon` | string | Short description of the solar event (always present) |
| `confidence` | `"high"` / `"medium"` / `"low"` | Drives strategy selection in Stage 2 |

### Failed papers

Papers that fail JSON parsing have their raw model output saved in:
```
<output_dir>/failed/<pdf_stem>_raw.txt
```

This file contains the unmodified string returned by the model, including any markdown or preamble that caused parsing to fail. It is useful for debugging prompt failures or tuning the post-processing logic.

---

## End-to-End Example

```bash
# 1. Activate the conda environment
conda activate pytorch_jupyter

# 2. Run metadata extraction on the Labrosse paper folder
./tools/extract_plots.sh metadata \
    --paper-dir "output/papers/2012-01 - Labrosse, N" \
    --output_dir output/metadata/
# Output:
#   Loading tokenizer: Qwen/Qwen2.5-14B-Instruct
#   Loading model with 8-bit quantization …
#   Model loaded.
#   Found 1 PDF(s) in output/papers/2012-01 - Labrosse, N
#     [ok]    paper_2620529.pdf  (4 event(s))
#
#   Summary: 1 processed, 0 skipped, 0 failed  (total PDFs: 1)

# 3. Inspect the output
cat output/metadata/paper_2620529.json

# 4. Hand off the Labrosse metadata to Stage 2
./tools/extract_plots.sh query \
    --metadata_dir output/metadata/ \
    --fits_dir output/fits/ \
    --output_dir output/matched/

# 5. If needed, retry a failed paper by deleting its JSON and re-running
rm output/metadata/paper_2620529.json
./tools/extract_plots.sh metadata \
    --paper-dir "output/papers/2012-01 - Labrosse, N" \
    --output_dir output/metadata/
```

---

## Known Limitations

1. **Scanned PDFs produce no text:** PyMuPDF can only extract text from PDFs with embedded text layers. Papers distributed as scanned images (rare in modern astrophysics journals but common for pre-2000 papers) will produce empty text and be recorded as failed. OCR is not performed.

2. **Figure citations without caption matches:** `extract_figure_body_refs()` matches body paragraphs to figures by the number in the inline citation (e.g., `"Fig. 2"`). If the paper uses non-standard citation styles (e.g., roman numerals, letters only, or unnumbered figures) the body-refs list for those figures will be empty and the model will have only the caption text to work from.

3. **Context window truncation:** The per-figure context string is truncated to 12 000 characters. Papers with many figures may have their later figures cut off. If this is suspected, inspect `failed/<stem>_raw.txt` or increase the `max_chars` value in `extract_pdf_context()`.

4. **Model hallucination:** The LLM may invent observation events not present in the paper, or assign plausible but incorrect timestamps and coordinates. Each extracted event should be treated as a candidate, not a ground truth. The `confidence` field provides a rough reliability signal, but manual spot-checking is recommended for critical use cases.

5. **JSON instability at low temperature:** Even at `temperature=0.1`, the model occasionally produces malformed JSON (unclosed brackets, trailing commas). The two-tier parser handles most cases, but papers saved in `failed/` require manual inspection.

6. **First-run model download:** The ~14 GB Qwen2.5-14B-Instruct weights are downloaded from HuggingFace Hub on the first call and cached in `models/hub/` inside the project directory (not `~/.cache/huggingface/`). The download takes ~15–30 minutes depending on network speed. Ensure `scratchsan` has at least 16 GB free before first run.

7. **Single paper per process:** The model is loaded once and kept resident for the full batch. Processing a single paper in isolation still pays the full ~2 minute model load time. For large batches (100+ papers), this amortizes to negligible cost.

8. **`--paper-dir` looks for PDFs inside the directory:** When using the `--paper-dir` alias (matching the `label` command interface), the script searches for `*.pdf` files inside the given directory. If the extract stage was run with `--keep-pdf`, the PDF is stored as `paper_<id>.pdf` inside the paper folder and will be found automatically. If only the paper folder exists but no PDF is inside it, no PDFs will be found.
