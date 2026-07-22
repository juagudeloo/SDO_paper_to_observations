# EXTRACT_PLOTS — Pipeline Documentation

Step-by-step explanation of the SDO plot extraction pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pipeline Structure](#pipeline-structure)
4. [Stage 1: List Papers (`list`)](#stage-1-list-papers)
5. [Stage 2: Extract Solar Images (`extract`)](#stage-2-extract-solar-images)
6. [Solar Observation Classification Algorithm](#solar-observation-classification-algorithm)
7. [Output Folder Naming](#output-folder-naming)
8. [End-to-End Example](#end-to-end-example)
9. [Known Limitations](#known-limitations)

---

## Overview

The pipeline has two stages:

1. **List** — Query the NASA ADS SDO API for papers published in a date range and write a CSV file so you can browse and select papers of interest.
2. **Extract** — Given a paper ID, download the PDF, extract every embedded image, classify each as a solar observation or not, and save the solar images to a named folder.

```
[NASA ADS SDO API] ──list──> papers_YYYYMMDD_YYYYMMDD.csv
                                         |
                              (pick paper ID)
                                         |
[NASA ADS SDO API] ──extract──> output/images/YYYY-MM - Author, I/
                                    ├── solar_001_p2_aia_false_color.png
                                    ├── solar_002_p3_hmi_grayscale.png
                                    └── extraction_log.json
                               output/papers/YYYY-MM - Author, I.pdf
```

`extract` writes directly into the canonical `output/` layout (`images/`,
`papers/`) that the later stages — `metadata` and `query` — consume, each
addressing a paper by its canonical name. See `docs/METADATA_EXTRACTION.md` and
`docs/SDO_QUERY.md` for those stages.

---

## Prerequisites

### 1. Start the NASA ADS SDO API

The API reads from a local SQLite database of ~30,000 SDO papers (2010–2024).

```bash
cd ../NASA_ADS_SDO
./run_api.sh          # starts at http://localhost:8000
```

Verify it is running:
```bash
curl http://localhost:8000/
# Expected: {"message": "SDO Documents API", "version": "...", "docs": "/docs"}
```

### 2. Python library: PyMuPDF

The pipeline uses **PyMuPDF** (`fitz`) for all PDF operations — image extraction and text extraction from the first page. poppler-utils (`pdfimages`, `pdftotext`) is **not** required.

```bash
pip install pymupdf
```

Verify the import works:
```python
import fitz; print(fitz.version)
```

### 3. Python environment (pytorch_jupyter conda env)

The shell script uses the `pytorch_jupyter` conda environment, which already has all required Python packages (`cv2`, `requests`, `PIL`, `numpy`).

To install any missing packages:
```bash
conda activate pytorch_jupyter
pip install -r requirements_extract.txt
```

---

## Pipeline Structure

```
tools/
└── extract_plots.sh      # Shell entry point (validates env, dispatches)

scripts/                  # Main entry points (run directly)
├── list_papers.py        # Stage 1: date range → CSV
└── extract_plots.py      # Stage 2: paper ID → images

utils/                    # Utility modules (imported by scripts/)
├── api_client.py         # API communication (health check, pagination, download)
├── folder_naming.py      # Date formatting and author name parsing
├── pdf_extractor.py      # PyMuPDF-based image and text extraction
└── solar_classifier.py   # Image classifier (Hough circles + HSV analysis)
```

The shell script (`tools/extract_plots.sh`) is a thin wrapper that validates the environment and dispatches to the appropriate Python script in `scripts/`. All reusable algorithmic logic lives in `utils/`.

---

## Stage 1: List Papers

**Command:**
```bash
./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01
```

**Implemented in:** `scripts/list_papers.py` (orchestrator) + `utils/api_client.py` (API calls)

**What it does:**

1. **Validate inputs** — `validate_date()` in `list_papers.py` checks that `--start` and `--end` are in `YYYY-MM-DD` format and that start ≤ end. Fails fast with a clear message before any network call.

2. **Check the API is alive** — `check_api_health(base_url)` in `utils/api_client.py` sends `GET /` and confirms the JSON response contains a `"message"` key. If the server is unreachable (connection error or timeout), it prints the start-the-server instructions and exits.

3. **Fetch papers year by year** — `get_documents_for_year(base_url, year)` in `utils/api_client.py` calls `GET /documents/?year=YYYY&skip=N&limit=1000` in a loop, advancing the `skip` offset until the page returned is shorter than the page size. This handles databases with more than 1 000 papers per year without missing any records.

4. **Filter to the exact month range** — `pub_date_in_range(pub_date, start_date, end_date)` in `utils/api_client.py` compares dates at month granularity (ignoring the day, which the database always stores as `00`). The year-level API query is coarse; this function performs the precise client-side cut.

5. **Sort results** — `main()` in `list_papers.py` sorts all collected documents by `(publication_date, id)` so the output file is in chronological order.

6. **Write output files** — `main()` writes a CSV via Python's `csv.DictWriter` (truncating long titles with `truncate_title()`) and/or a Markdown file with abstracts via `write_markdown()`. Both are saved under `output/searched_papers/` by default.

**Output columns in the CSV:**

| Column | Description |
|--------|-------------|
| `id` | Paper ID in the database (use this for the `extract` command) |
| `title` | Paper title (truncated to 80 chars) |
| `authors` | Author list (note: often empty in the DB — see paper directly) |
| `publication_date` | Publication date as stored (format: YYYY-MM-00) |
| `doi` | Digital Object Identifier |
| `bibcode` | NASA ADS bibliographic code |
| `citation_count` | Number of citations |
| `ads_url` | Link to the paper on NASA ADS |

**Output file:** `papers_20120102_20130301.csv` (in current directory by default)

**Date filtering note:** The database stores publication dates as `YYYY-MM-00` (day is always `00`). Filtering is done at month granularity: a paper from `2012-07-00` is included in the range `2012-01-02` to `2013-03-01` because July 2012 falls within that range.

---

## Stage 2: Extract Solar Images

**Command:**
```bash
./tools/extract_plots.sh extract --id 2620529
./tools/extract_plots.sh extract --id 2620529 --if-exists overwrite --purge-downstream
```

**Implemented in:** `scripts/extract_plots.py` (orchestrator), with helpers from `utils/api_client.py`, `utils/folder_naming.py`, `utils/pdf_extractor.py`, and `utils/solar_classifier.py`.

**What it does, step by step:**

### Step 0: Detect a prior extraction of this paper ID
**Functions:** `_find_existing_paper_by_id()`, `_resolve_overwrite()` — `scripts/extract_plots.py`

Before any network call, scans `<root>/images/*/extraction_log.json` for an entry whose `paper_id` matches `--id` — the ID is already recorded there from a previous `extract` run, so this is a local-only check with no API/download cost.

If a match is found, `--if-exists` decides what happens next:
- **`skip`** (or answering "no" under `ask`) — prints a message and exits without touching the network.
- **`overwrite`** (or answering "yes" under `ask`) — deletes the old `images/<name>/` directory and its kept PDF, then the pipeline proceeds as normal from Step 1, rebuilding both from scratch. This prevents orphaned files: since saved-image filenames are numbered per-run (`solar_001_...`, `solar_002_...`), a prior run with a different `--min-score` or `--save-all` could otherwise leave stale files alongside the new ones.
- **`ask`** (the default) — prompts interactively (`Paper 'X' already exists (N image(s) saved). Overwrite? [y/N]:`). If stdin is not a TTY (e.g. running under a script or cron without `--if-exists` set explicitly), it **aborts with an error instead of hanging** on `input()`.

When overwriting, if `<root>/metadata/<name>.json` and/or `<root>/matched/<name>__*` already exist for this paper (produced by the downstream `metadata`/`query` stages), they are left in place with a warning by default — since regenerating them is expensive (an ~8 min model load for `metadata`, fresh VSO downloads for `query`). Pass `--purge-downstream` to also delete them (under interactive `ask`, this is a second, separate y/N prompt instead).

### Step 1: Fetch metadata
**Function:** `get_document_by_id(base_url, doc_id)` — `utils/api_client.py`

Sends `GET /documents/{id}` to the API and returns the full JSON record for the paper (title, publication date, bibcode, DOI, etc.). Raises a clear `ValueError` if the ID is not in the database (HTTP 404), so the user knows immediately if they typed a wrong ID.

### Step 2: Download PDF
**Function:** `download_pdf(base_url, doc_id, dest_path, source=None)` — `utils/api_client.py`

Sends `GET /documents/{id}/download-pdf` with optional `?source=arxiv` or `?source=publisher`. The response is a binary PDF streamed in 8 KB chunks to avoid loading the whole file into memory. The file is saved to a system temporary directory that is cleaned up automatically at the end. If no PDF is publicly available the function raises `RuntimeError` and prints the ADS abstract page URL so the user can access the paper manually.

### Step 3: Parse author name
**Functions:** `extract_first_page_text(pdf_path)` — `utils/pdf_extractor.py`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`parse_first_author(authors_field, bibcode, pdf_text)` — `utils/folder_naming.py`

`extract_first_page_text` uses **PyMuPDF** (`fitz`) to pull the raw text from the first PDF page — fast and reliable without needing poppler.

`parse_first_author` then applies three strategies in order:
1. **`_parse_authors_field()`** — tries the database `authors` field (always empty in this DB, but future-proof).
2. **`_parse_pdf_text()`** — applies three regular expressions against the first 3 000 characters of extracted text to match the common astronomy author formats (`F. I. LastName`, `LastName, F. I.`, `FirstName LastName`).
3. **`_parse_bibcode()`** — fallback: the last character of the NASA ADS bibcode is always the first author's surname initial (e.g. `2012A&A...543A..53S` → `S`).

### Step 4: Create output folder
**Functions:** `format_publication_date(pub_date)` and `build_folder_name(pub_date, last_name, first_initial)` — `utils/folder_naming.py`

`format_publication_date` converts the DB format `YYYY-MM-00` (day always `00`) to a human-readable string: `2012-07-00` → `2012-07`, `2010-00-00` → `2010`.

`build_folder_name` combines the formatted date and author into `YYYY-MM - LastName, F` (the canonical **paper name**), then strips any characters forbidden by common filesystems (`/ \ : * ? " < > |`). The per-paper image directory is resolved via `folder_naming.images_dir(root, name)` → `<root>/images/<name>/`, where `root` is `--output-dir` (default `./output/`). All canonical paths (`images_dir`, `pdf_path`, `log_path`, …) live in `utils/folder_naming.py` so no stage hardcodes `output/...`.

### Step 5: Extract images
**Function:** `extract_pdf_images(pdf_path, output_dir)` — `utils/pdf_extractor.py`

Uses **PyMuPDF** (`fitz`) to iterate every page and collect embedded images via `page.get_images(full=True)`. PyMuPDF is preferred over the older `pdfimages` (poppler) because it also finds images inside Form XObjects and indirect image streams that poppler silently skips.

Each unique image (tracked by its internal `xref` reference number to avoid duplicates from shared objects) is decoded and saved as a PNG via the helper `_save_as_png()` — which converts JPEG, JPEG2000, and CMYK images to standard RGB PNG on the fly using **Pillow**.

The function returns a list of `ImageMetadata` dataclass objects, one per image, carrying: sequential index, page number, width × height, color space (`rgb`/`gray`/`cmyk`), original encoding, the PNG file path, and the image's **page-placement `bbox`** `(x0, y0, x1, y1)`. The bbox comes from `page.get_image_rects(xref)` (largest-area rect when an image is placed more than once) and is what lets the downstream `metadata` stage match each image to its figure caption without re-parsing the PDF.

### Step 6: Classify images
**Function:** `classify_image(img_meta)` — `utils/solar_classifier.py`

The main entry point runs two sub-functions:

- **`_metadata_prefilter(img_meta)`** — immediately rejects images smaller than 200 × 200 px (logos, icons) or with a palette/indexed color space (diagrams). Returns a rejected `ClassificationResult` without ever opening the file.
- **`_classify_pixels(img_bgr, path)`** — loads the PNG with OpenCV, converts to both grayscale and HSV, then accumulates an integer raw score across Steps 2–8 of the [classification algorithm](#solar-observation-classification-algorithm).

The classification result (a `ClassificationResult` dataclass) contains:
- `is_solar`: `True` if raw score ≥ 5
- `score`: raw score normalized to 0–1 (dividing by 20)
- `signals`: ordered list of string tags explaining each score contribution (e.g. `dark_background`, `full_disk_circle_r320`)
- `image_type`: `aia_false_color`, `hmi_grayscale`, `unknown`, or `rejected`

### Step 7: Save solar images
**Implemented in:** `main()` — `scripts/extract_plots.py`

Iterates the list of `(ImageMetadata, ClassificationResult)` triples for images that passed the filter (`is_solar=True` and `score >= --min-score`, default 0.25). Each is copied from the temp directory into `output/images/<name>/` using `shutil.copy()` with a descriptive filename that encodes its sequential number, PDF page, and detected type:
```
solar_001_p2_aia_false_color.png
solar_002_p3_hmi_grayscale.png
```
The saved filename is written back into that image's log entry (`filename`) so the `metadata` stage can link each observation to a real file on disk.

### Step 8: Write extraction log
**Implemented in:** `main()` — `scripts/extract_plots.py`

Writes `extraction_log.json` into `output/images/<name>/` using Python's built-in `json.dump`. The log contains the paper metadata at the top, plus one entry per image (both kept and rejected) with its page, **bbox**, size, color space, classification score, signals list, image type, and — for saved images — the `filename`. This log is both the audit trail (why an image was kept or discarded) *and* the input contract for the `metadata` stage (which images are solar, where they sit on the page, and what file they map to) — so `metadata` never re-derives any of it from the PDF.

### Step 9: Keep the PDF
**Implemented in:** `main()` — `scripts/extract_plots.py`

The downloaded PDF is copied to its canonical location `output/papers/<name>.pdf` (`folder_naming.pdf_path`). This happens **by default** because the `metadata` stage needs the PDF for caption text and citing paragraphs; pass `--no-keep-pdf` to skip it.

---

## Solar Observation Classification Algorithm

**Implemented in:** `utils/solar_classifier.py` — primarily `_classify_pixels()`, called by `classify_image()`.

The classifier uses a sequence of additive integer scoring steps. An image is classified as solar if the raw score is **≥ 5**. All thresholds and score weights are defined as module-level constants at the top of `utils/solar_classifier.py` so they can be tuned in one place without touching the logic.

### Step 1: Metadata pre-filter (fast, no pixel reading)
**Function:** `_metadata_prefilter(img_meta)` — `utils/solar_classifier.py`

This function checks the `ImageMetadata` fields (width, height, color_space) that were already available from the PDF extraction — no file I/O needed. If rejected, `classify_image()` returns immediately without ever calling `cv2.imread()`.

| Condition | Action | Signal |
|-----------|--------|--------|
| width < 200 or height < 200 | Reject immediately | `too_small` |

**Why:** Small images are logos, icons, or decorative elements. Palette-indexed images are no longer rejected here — paper figures are often saved as indexed PNGs to reduce file size even when they show real solar observations. They are instead converted to BGR by `cv2.imread()` and proceed to pixel-level classification.

### Step 2: Background color analysis (±4 points)
**Function:** `_classify_pixels()` — `utils/solar_classifier.py` (background analysis block)

After loading the image with `cv2.imread()` and converting to HSV with `cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)`, the code builds a boolean `border_mask` covering the outer 5% ring of pixels on all four sides. It then samples the **V** (Value/brightness) channel of those border pixels.

| Condition | Score | Signal |
|-----------|-------|--------|
| >50% of border pixels have V > 240 (white) | −4 | `white_background` |
| >20% of border pixels have V < 25 (dark) | +2 | `dark_background` |

**Why:** Scientific plots (light curves, spectra) almost always have white backgrounds. Solar observations from space have a very dark background — the background is the vacuum of space or the solar limb region.

### Step 3: Global dark pixel ratio (+1 or +2)
**Function:** `_classify_pixels()` — dark pixel ratio block

Computes the fraction of *all* pixels whose V channel is < 20 (near-black) using NumPy's vectorised comparison `(v_ch < 20).mean()`.

| Condition | Score | Signal |
|-----------|-------|--------|
| > 15% dark pixels | +2 | `dark_region_XX%` |
| 5–15% dark pixels | +1 | `moderate_dark_region` |

**Why:** Solar full-disk images have a large black region around the solar disk. Even cropped active-region images often show dark corona at the edges.

### Step 4: Hough Circle Transform (+3 or +8)
**Function:** `_classify_pixels()` — Hough circle block

This is the most important signal. It detects the circular solar disk. The grayscale image is first blurred with a 9×9 Gaussian kernel (`cv2.GaussianBlur`) to suppress noise, then passed to `cv2.HoughCircles` with the HOUGH_GRADIENT method:

```python
min_radius = int(min(H, W) * 0.30)   # circle must be 30–55% of shorter dimension
max_radius = int(min(H, W) * 0.55)
circles = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT,
                           dp=1.5, minDist=min(H,W)*0.4,
                           param1=100, param2=40,
                           minRadius=min_radius, maxRadius=max_radius)
```

If a circle is found, the code checks whether the detected center `(cx, cy)` falls within the central 50% of the image (i.e., `0.25·W < cx < 0.75·W` and `0.25·H < cy < 0.75·H`). This distinguishes a full-disk image (circle centered) from a partially-visible disk (circle off-center). The flag `is_full_disk` is set here and reused in Steps 6 and 7.

| Condition | Score | Signal |
|-----------|-------|--------|
| Circle found, center in middle 50% of image | +8 | `full_disk_circle_rNNN` |
| Circle found, center off-center | +3 | `off_center_circle` |

**Why:** Full-disk SDO images always show the solar disk as a large circle centered in the image. The radius range (30–55% of the shorter dimension) is tuned to match full-disk solar images at typical figure sizes in papers.

### Step 5: AIA False-Color Detection (+4 or +8)
**Function:** `_classify_pixels()` — AIA color detection block

SDO/AIA produces false-colored images at different EUV wavelengths, each with a characteristic color palette. The code converts the image to HSV and builds three Boolean pixel masks (one per color family) combined with a saturation threshold (`s_ch > 80`) so that faint, nearly-grey pixels are ignored:

```python
orange_mask = (h_ch >= 5)  & (h_ch <= 30)  & (s_ch > 80)   # 193, 94, 1600 Å
blue_mask   = (h_ch >= 42) & (h_ch <= 65)  & (s_ch > 80)   # 171 Å
red_mask    = ((h_ch <= 5) | (h_ch >= 170)) & (s_ch > 80)  # 304 Å
aia_ratio   = orange_mask.mean() + blue_mask.mean() + red_mask.mean()
```

| Wavelength | Color | HSV hue range (OpenCV 0–179) |
|------------|-------|------------------------------|
| 193 Å, 94 Å, 1600 Å | Orange/gold | 5–30 |
| 171 Å | Blue-green | 42–65 |
| 304 Å | Red | 0–5 or 170–179 |

| Condition | Score | Signal |
|-----------|-------|--------|
| > 15% of pixels match AIA colors | +8 | `aia_false_color_XX%` |
| 5–15% of pixels match AIA colors | +4 | `aia_weak_color_XX%` |
| Sets `image_type = "aia_false_color"` | | |

### Step 6: HMI Grayscale Detection (+3 or +4)
**Function:** `_classify_pixels()` — HMI detection block

SDO/HMI images (continuum intensity, line-of-sight magnetic field, Dopplergram) are grayscale or near-grayscale. The test is simple: compute `s_ch.mean()` (mean saturation over the whole image). A very low mean saturation means all pixels are nearly grey.

| Condition | Score | Signal |
|-----------|-------|--------|
| Mean saturation < 15 AND full disk detected | +4 | `hmi_grayscale_satN.N` |
| Mean saturation < 15 AND no full disk (cropped region) | +3 | `hmi_region_satN.N` |

**Why:** Cropped HMI active-region images are common in papers and will not have passed the Hough circle test, so they need their own unconditional positive score to overcome the threshold.

### Step 7: Texture Analysis for Cropped Regions (+2)
**Function:** `_classify_pixels()` — texture block

Cropped active regions (not full-disk) won't trigger the Hough circle. However, the solar surface has characteristic granulation and sunspot structure that produces high pixel variance. The test is `gray.std()` — the standard deviation of grayscale pixel values across the entire image.

| Condition | Score | Signal |
|-----------|-------|--------|
| `std(gray) > 40` AND `is_full_disk = False` | +2 | `high_texture_stdNN` |

### Step 8: Scientific Plot Penalty (−2 or −5)
**Function:** `_classify_pixels()` — plot penalty block

Scientific plots (light curves, power spectra, scatter plots) tend to have dense edges (many axis ticks and text) and large white interior regions. Two metrics are combined:

```python
edge_density        = cv2.Canny(gray, 50, 150).mean() / 255.0
interior_white_ratio = (gray[H//8:7*H//8, W//8:7*W//8] > 240).mean()
```

`cv2.Canny` runs the Canny edge detector. Dividing by 255 gives the fraction of edge pixels. The interior crop (central 75% in each dimension) avoids counting border artefacts.

| Condition | Score | Signal |
|-----------|-------|--------|
| edge_density > 0.15 AND interior white > 30% | −5 | `scientific_plot_edgesN.NN_whiteXX%` |
| edge_density > 0.10 AND interior white > 20% | −2 | `possible_diagram` |

### Final Decision
**Function:** end of `_classify_pixels()` — `utils/solar_classifier.py`

```python
is_solar   = (raw_score >= SOLAR_SCORE_THRESHOLD)   # constant = 5
normalized = max(0.0, min(1.0, raw_score / SCORE_NORMALIZATION))  # constant = 20
```

The raw integer score is compared against the threshold (5). The normalized confidence (0–1) is returned alongside the `is_solar` boolean, the `signals` list, and the `image_type` string. All four fields are packed into a `ClassificationResult` dataclass and returned up the call stack to `main()` in `extract_plots.py`.

---

## Output Folder Naming

Each paper is identified by a canonical **name**:

```
YYYY-MM - LastName, F
```

For example: `2012-01 - Labrosse, N`. That name keys the whole canonical layout under the output root (default `output/`):

```
output/
  papers/    <name>.pdf                         # kept PDF
  images/    <name>/*.png + extraction_log.json # this stage
  metadata/  <name>.json                        # metadata stage
  matched/   ...                                # query stage
  fits/      ...                                # query FITS cache
```

**Date handling:** The database stores publication dates as `YYYY-MM-00` (day is always `00`). The formatter strips the `00` day to produce `YYYY-MM`. For year-only entries (`YYYY-00-00`), it produces just `YYYY`.

**Author detection:** The database's `authors` field is always empty. The pipeline extracts the first author from the PDF's first page text using regex patterns for common astronomy journal author formats. If parsing fails, the NASA ADS bibcode's last character (which encodes the first author's surname initial) is used as a fallback.

---

## End-to-End Example

```bash
# 1. Start the API (in a separate terminal)
cd ../NASA_ADS_SDO && ./run_api.sh

# 2. List papers from 2012 to 2013
./tools/extract_plots.sh list --start 2012-01-01 --end 2013-12-31
# Output: papers_20120101_20131231.csv
# Open the CSV, browse titles, pick a paper ID (e.g. 2620529)

# 3. Extract solar images from that paper (PDF kept in output/papers/ by default)
./tools/extract_plots.sh extract --id 2620529
# Output:
#   Processing paper ID 2620529
#     Title     : Plasma diagnostic in eruptive prominences from SDO/AIA observations...
#     Published : 2012-01-00
#     Bibcode   : 2012A&A...537A.100L
#     PDF size  : 0.90 MB
#     Author    : Labrosse, N
#     Output    : output/images/2012-01 - Labrosse, N
#   Found 22 embedded images in PDF
#     [SOLAR] img-000 p3 752x752 rgb  score=0.60 [dark_background, dark_region_17%, aia_false_color_102%]
#     ...
#   Extraction complete: 22/22 images classified as solar observations
#   Output folder: output/images/2012-01 - Labrosse, N
#   Log: output/images/2012-01 - Labrosse, N/extraction_log.json
#   PDF kept at: output/papers/2012-01 - Labrosse, N.pdf

# 4. View the extracted images
ls output/images/2012-01\ -\ Labrosse,\ N/
```

---

## Known Limitations

1. **Multi-panel figures**: Many SDO papers show composite figures (e.g. a 3×2 grid of AIA images at different wavelengths). PyMuPDF extracts the entire figure as one embedded image object, which is the correct behavior for downstream image matching.

2. **Author detection from PDF**: Parsing author names from raw PDF text is heuristic and may fail for unusual journal formats. In that case, the bibcode fallback provides an initial letter instead of a full last name.

3. **Empty authors field in DB**: The NASA ADS SDO database has an empty `authors` column for all records. Author names are always extracted from the PDF text.

4. **Papers without public PDFs**: Some papers (especially publisher-only papers) may not have freely accessible PDFs. The pipeline will print a warning with the ADS abstract page URL where you can access the paper directly.

5. **Cropped active region images**: Images that show only a small portion of the solar disk (no visible limb) may score lower than full-disk images. The texture analysis step helps, but may not always be sufficient. Adjusting `--min-score` downward can recover these.

6. **CMYK images**: Publisher PDFs often embed images in CMYK color space. PyMuPDF extracts them as raw bytes and `_save_as_png()` converts them to RGB via Pillow before saving, so they are handled correctly by the classifier.
