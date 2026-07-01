# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Doctoral research pipeline that matches solar observations from NASA's SDO (Solar Dynamics
Observatory) satellite with images published in scientific papers. Given a corpus of SDO papers,
it extracts the solar-observation figures, reads the observational metadata (time, wavelength,
coordinates) out of the surrounding text with an LLM, then re-queries the SDO/VSO archive to
reproduce the matching observation as a cropped submap.

## Architecture

The system is a **four-stage CLI pipeline**, not a notebook. Every stage is a `scripts/*.py`
program that imports shared logic from `utils/`, and all stages are dispatched through a single
wrapper, `tools/extract_plots.sh`, which activates the conda env and validates dependencies per
stage. Data flows through a **canonical `output/` layout** keyed by the paper-name string; each
stage after `extract` addresses a paper by that name (`--paper-name`) or processes them all
(`--all`).

| Stage | Command | Script | Consumes → Produces |
|-------|---------|--------|---------------------|
| 1 | `list`     | `list_papers.py`         | date range → `output/searched_papers/*.csv,*.md` |
| 2 | `extract`  | `extract_plots.py`       | paper ID → `output/images/<name>/*.png` + `extraction_log.json`, `output/papers/<name>.pdf` |
| 3 | `metadata` | `metadata_extraction.py` | paper name → `output/metadata/<name>.json` |
| 4 | `query`    | `sdo_query.py`           | paper name → `output/matched/` (cropped submap PNG + JSON), FITS cache in `output/fits/` |

The canonical layout under the output root (default `output/`, set with `--output-dir`):

```
papers/    <name>.pdf                         (kept by extract unless --no-keep-pdf)
images/    <name>/*.png + extraction_log.json (extract; log carries per-image bbox)
metadata/  <name>.json                         (metadata)
matched/   ...                                 (query)
fits/      ...                                 (query FITS cache)
failed/    ...                                 (failures)
```

Key cross-cutting design points a change is likely to touch:

- **`utils/api_client.py` is the sole gateway to the external NASA ADS SDO API.** The API base
  URL comes from `SDO_API_URL` (default `http://localhost:8000`). Stages `list`/`extract` require
  it running; it lives in the sibling repo `../NASA_ADS_SDO` and is started with `./run_api.sh`.
- **Folder naming and the layout are canonical and shared.** `utils/folder_naming.py` builds the
  `YYYY-MM - LastName, F` name from paper metadata (the DB stores dates as `YYYY-MM-00`, and its
  authors field is empty, so the first author is parsed out of the PDF's first-page text), *and*
  resolves every canonical path (`pdf_path`, `images_dir`, `log_path`, `metadata_json`,
  `matched_dir`, `fits_dir`, plus `iter_paper_names` for `--all`). No stage hardcodes `output/...`
  paths — changing the layout is a one-file change here.
- **Image geometry is captured once, at extract.** `utils/pdf_extractor.py` records each image's
  page-placement `bbox`; `extract_plots.py` writes it (with the saved PNG `filename`) into
  `extraction_log.json`. Downstream `metadata` reads the bbox from the log and matches to the
  nearest figure caption by vertical proximity via `utils/caption_extractor.py`
  (`match_image_to_caption`) — it does **not** re-derive image bboxes from the PDF. It still opens
  the PDF for caption text and for the body-text paragraphs that cite each figure
  (`extract_figure_body_refs`), which feed the LLM.
- **One classifier.** `utils/solar_classifier.py` decides *is this image a solar observation*
  using classical CV (Hough circles, HSV palette analysis, HMI grayscale/texture heuristics; raw
  score ≥ 5 → solar). The old zero-shot caption classifier (`structure_classifier_nlp.py`, BART)
  was removed — the metadata LLM's `phenomenon` field supersedes it.
- **Model cache is redirected into the repo.** `metadata_extraction.py` sets `HF_HOME` to
  `models/` before importing `transformers`; the Qwen weights download there, not to `~/.cache`.
- **Metadata LLM:** `Qwen/Qwen2.5-14B-Instruct`, 8-bit quantised (needs `bitsandbytes` + GPU).
  It reads caption + citing paragraphs per image and emits structured observation metadata.
- **`query` has three fallback strategies** (see `LIMB_BOXES` / strategy A/B/C in
  `sdo_query.py`), degrading from explicit Heliprojective Tx/Ty + FOV (`high` confidence),
  to an approximate limb bounding box (`medium`), to a full-disk map for downstream CV (`low`).

## Commands

All work goes through the wrapper (it activates conda env `pytorch_jupyter` and prepends its
`lib/` to `LD_LIBRARY_PATH` to avoid stale system `libstdc++`):

```bash
# Stage 1 — list papers in a date range
./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01 [--format csv|md|both]

# Stage 2 — extract solar images (PDF kept in output/papers/ by default; --no-keep-pdf to omit)
./tools/extract_plots.sh extract --id 2620529 [--source arxiv|publisher] [--min-score 0.25]

# Stage 3 — extract observation metadata via LLM (one paper, or --all)
./tools/extract_plots.sh metadata --paper-name "2012-01 - Labrosse, N"

# Stage 4 — query SDO/VSO and produce cropped submaps (one paper, or --all)
./tools/extract_plots.sh query --all

# Environment overrides
SDO_API_URL=http://host:8000 CONDA_ENV=pytorch_jupyter ./tools/extract_plots.sh list ...
```

Scripts can also be run directly (`python3 scripts/<name>.py ...`) if the env is already active;
each script inserts the project root on `sys.path` so `from utils... import` works from anywhere.

> Note: `./tools/extract_plots.sh test` runs `unittest discover` under `scripts/tests/`, but that
> directory does not currently exist — there is no test suite yet.

## Environment & Dependencies

- Conda env: **`pytorch_jupyter`** (override with `CONDA_ENV`).
- Core: `pip install -r requirements_extract.txt` (numpy, Pillow, opencv-python, requests,
  matplotlib, pymupdf, transformers). `list`/`extract` need only these.
- `metadata` additionally needs `bitsandbytes` + `accelerate` (8-bit quantisation, GPU) for the
  Qwen LLM.
- `query` additionally needs `sunpy` + `astropy` (VSO query, FITS, coordinate transforms).
- The wrapper checks these per-stage and prints an install hint if missing.

## Documentation

Each pipeline area has a long-form design doc in `docs/`: `EXTRACT_PLOTS.md`,
`IMAGE_CAPTION_PIPELINE.md`, `METADATA_EXTRACTION.md`, `SDO_QUERY.md`. Consult these before
changing a stage's algorithm — they explain the heuristics in detail.

## Conventions

- **Never import from `typing`.** Use native Python 3.10+ syntax (`list[...]`, `dict[...]`,
  `X | None`). Some existing files still use `typing`; new/edited code should not.

## Notes

- `notebooks/example_notebook.ipynb` is exploratory (the original sunpy/coordinate-conversion
  prototype) and is not part of the CLI pipeline.
- The VS Code workspace (`SDO_paper_to_observations.code-workspace`) also includes the sibling
  `../NASA_ADS_SDO` project that serves the paper API.
