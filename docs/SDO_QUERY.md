# SDO_QUERY — Pipeline Documentation

Step-by-step explanation of the automated SDO archive querying and submap extraction pipeline (Stage 2: `query`).

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pipeline Structure](#pipeline-structure)
4. [Stage 2: Query SDO (`query`)](#stage-2-query-sdo)
5. [FITS Download and Caching](#fits-download-and-caching)
6. [Extraction Strategies](#extraction-strategies)
7. [Output Format](#output-format)
8. [End-to-End Example](#end-to-end-example)
9. [Known Limitations](#known-limitations)

---

## Overview

The `query` stage reads the JSON metadata files produced by Stage 1 (`metadata`) and, for each observation event, downloads the closest matching FITS file from the Virtual Solar Observatory (VSO), crops the solar map to the relevant region using one of three strategies, and saves a normalised PNG alongside a companion JSON file.

```
papers/metadata/
    ├── paper_A.json          (produced by metadata stage)
    └── paper_B.json
              │
         (query stage)
              │
    ┌─────────┴─────────┐
    ▼                   ▼
papers/sdo_fits/        papers/matched/
    └── paper_A__000    └── paper_A__000.png   ← normalised uint8 PNG
        __AIA__171        paper_A__000.json   ← companion metadata JSON
        __2010-...fits
```

Three cropping strategies are applied in priority order based on the `confidence` field extracted in Stage 1:

| Strategy | Condition | Source data used |
|----------|-----------|-----------------|
| **A** | `confidence="high"` + explicit Tx/Ty | `center_tx_arcsec`, `center_ty_arcsec`, `fov_arcsec` |
| **B** | `confidence="medium"` + limb position | `limb_position` → approximate bounding box |
| **C** | `confidence="low"` or no position | Full-disk normalised map |

The stage is fully resumable: re-running skips events whose output PNG already exists.

---

## Prerequisites

### 1. sunpy and astropy

```bash
pip install sunpy astropy
```

Verify:
```python
import sunpy; import astropy; print(sunpy.__version__, astropy.__version__)
```

### 2. OpenCV and NumPy (already required by the extract stage)

Used for image normalisation and PNG writing:
```bash
pip install opencv-python numpy
```

### 3. Internet access to the VSO

The Virtual Solar Observatory (`vso.nascom.nasa.gov`) must be reachable for `Fido.search` and `Fido.fetch` calls. The stage gracefully handles connection errors and marks events as failed rather than crashing.

### 4. Stage 1 output

The `--metadata_dir` must contain at least one JSON file with `"status": "success"` produced by `stage1_metadata_extraction.py`. Files with `"status": "failed"` are silently skipped.

### 5. Python environment (pytorch_jupyter conda env)

The shell script activates this environment automatically. `sunpy` and `astropy` must be installed inside it.

---

## Pipeline Structure

```
tools/
└── extract_plots.sh      # Shell entry point (validates env, dispatches)

scripts/                  # Main entry points (run directly)
└── stage2_sdo_query.py   # Stage 2 orchestrator (all logic self-contained)

utils/                    # No new utilities — stage2 imports only stdlib + sunpy/cv2
```

Unlike the earlier stages, `stage2_sdo_query.py` does not import from `utils/`. All coordinate manipulation, FITS handling, and image normalisation logic is contained in the script itself. This keeps the sunpy/astropy dependency isolated from the classical OpenCV pipeline in `utils/`.

---

## Stage 2: Query SDO

**Command:**
```bash
./tools/extract_plots.sh query \
    --metadata_dir papers/metadata/ \
    --fits_dir papers/sdo_fits/ \
    --output_dir papers/matched/
```

**Implemented in:** `scripts/stage2_sdo_query.py`.

**What it does, step by step:**

### Step 1: Load all observation events
**Function:** `load_all_events(metadata_dir)` — `scripts/stage2_sdo_query.py`

Iterates every `*.json` file in `metadata_dir` in sorted filename order. Reads each file and skips it if `"status"` is not `"success"`. For files that pass, unpacks the `"observations"` list and appends one `(paper_stem, event_index, obs_dict)` tuple per event to the global event list.

`paper_stem` is the filename without extension (e.g., `"2012-01 - Labrosse, N"`). `event_index` is the zero-based position of the observation within that paper's list. These two values together uniquely identify an event and are used to build deterministic output filenames.

### Step 2: Check for existing output (resumability)
**Implemented in:** `process_event()` — `scripts/stage2_sdo_query.py`

Derives the output PNG path as `<output_dir>/<safe_stem>__<event_idx:03d>.png` (with characters unsafe for filenames replaced by underscores). If the PNG already exists, returns `"skipped"` immediately without any network call or disk I/O. The companion JSON is assumed to be present alongside the PNG.

### Step 3: Download the FITS file
**Function:** `fetch_fits(obs, fits_dir, cache_path)` — `scripts/stage2_sdo_query.py`

See [FITS Download and Caching](#fits-download-and-caching) for the full algorithm.

### Step 4: Load the FITS as a sunpy Map
**Function:** `load_map(fits_path)` — `scripts/stage2_sdo_query.py`

Wraps `sunpy.map.Map(fits_path)` inside `warnings.catch_warnings` with `warnings.simplefilter("ignore")`. This suppresses `astropy.io.fits.verify.VerifyWarning` (triggered by non-standard FITS headers, common in SDO/HMI files) without cluttering the console output. Returns `None` on any exception, which `process_event()` treats as a failure.

### Step 5: Select and apply an extraction strategy
**Functions:** `apply_strategy_a()`, `apply_strategy_b()` — `scripts/stage2_sdo_query.py`

See [Extraction Strategies](#extraction-strategies) for the full algorithm.

### Step 6: Normalise and save the PNG
**Functions:** `normalize_to_uint8(data)`, `save_outputs(smap, output_png, companion)` — `scripts/stage2_sdo_query.py`

`normalize_to_uint8` applies:
```python
cv2.normalize(np.abs(data), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
```

`np.abs()` is applied first to handle signed data from HMI line-of-sight magnetic field maps (which have negative values for south-pointing fields). `cv2.NORM_MINMAX` scales the full dynamic range to [0, 255], preserving relative intensity contrast without any manual thresholding.

`save_outputs` calls `cv2.imwrite` to write the PNG, then writes the companion JSON to the same directory with the same stem and a `.json` extension.

### Step 7: Print progress and strategy breakdown
**Implemented in:** `main()` — `scripts/stage2_sdo_query.py`

After each event, prints one line:
```
  [strategy_a   ]  2012-01 - Labrosse, N [000]  2010-09-19T08:00:00  AIA 304Å
  [strategy_b   ]  2012-01 - Labrosse, N [001]  2010-09-19T08:00:00  HMI noneÅ
  [skipped      ]  2012-01 - Song, Y [000]       2012-07-04T09:54:53  AIA 171Å
  [failed       ]  2012-01 - White, R [000]      2012-01-15T06:30:00  AIA 193Å
```

At the end, prints the strategy breakdown:
```
Summary (12 events):
  Strategy A (high confidence) : 4
  Strategy B (medium confidence): 5
  Strategy C (full disk)        : 2
  Skipped (already done)        : 1
  Failed                        : 0
```

---

## FITS Download and Caching

**Function:** `fetch_fits(obs, fits_dir, cache_path)` — `scripts/stage2_sdo_query.py`

All `sunpy` and `astropy` calls in this function are wrapped in a single `try/except Exception` block. If any step raises an error (network timeout, empty VSO response, download failure), the function logs a warning and returns `None`. The calling function records the event as `"failed"`.

### Cache key

Before making any network calls, `fetch_fits` checks whether `cache_path` already exists. The cache path is built by `_fits_cache_key(paper_stem, event_idx, obs)` using:

```
<safe_paper_stem>__<event_idx:03d>__<instrument>__<wavelength>__<timestamp>.fits
```

For example:
```
2012-01___Labrosse__N__000__AIA__304__2010-09-19T08_00_00.fits
```

All characters outside `[A-Za-z0-9\-_]` are replaced with underscores by the `_safe()` helper, making the filename safe on all common filesystems. If this file exists, `fetch_fits` returns its path immediately — no VSO call is made.

### VSO query

The time window is ±60 seconds around `timestamp_start`:

```python
start = (ts - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")
end   = (ts + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")
search_attrs = [a.Time(start, end)]
```

Additional attributes are added only when the metadata provides them:

```python
if obs.get("instrument"):
    search_attrs.append(a.Instrument(obs["instrument"]))
if obs.get("wavelength_angstrom"):
    search_attrs.append(a.Wavelength(int(wavelength) * u.angstrom))
```

Using `a.Instrument` and `a.Wavelength` as optional filters lets the query succeed even for events where the LLM extracted only a timestamp. Without them, the VSO returns all available instruments for that time window — often dozens of records — and the first result is used.

### Download

```python
downloaded = Fido.fetch(results[0, 0], path=fits_dir)
```

`results[0, 0]` selects the single closest record from the first available VSO client. `Fido.fetch` downloads the file to `fits_dir` with its original VSO filename. If the returned list is non-empty, the downloaded file is renamed to `cache_path` with `os.rename()`.

**Why rename?** The VSO generates filenames from the FITS header (e.g., `hmi.ic_45s.2012.07.04_09.52.38_TAI.continuum.fits`). Renaming to the deterministic `cache_path` makes the cache lookup reliable across runs, even if the VSO changes its naming convention.

---

## Extraction Strategies

### Strategy A — Explicit coordinates
**Function:** `apply_strategy_a(smap, obs)` — `scripts/stage2_sdo_query.py`

Applied when `confidence="high"` and `center_tx_arcsec` is not null.

Uses `center_tx_arcsec`, `center_ty_arcsec`, and `fov_arcsec` to construct two corner coordinates in the Heliprojective frame and extract a rectangular submap:

```python
tx, ty = obs["center_tx_arcsec"], obs["center_ty_arcsec"]
fov_w, fov_h = obs.get("fov_arcsec") or [300.0, 300.0]

bl = SkyCoord((tx - fov_w/2)*u.arcsec, (ty - fov_h/2)*u.arcsec,
              frame=smap.coordinate_frame)
tr = SkyCoord((tx + fov_w/2)*u.arcsec, (ty + fov_h/2)*u.arcsec,
              frame=smap.coordinate_frame)
return smap.submap(bl, top_right=tr)
```

`smap.coordinate_frame` provides the Helioprojective frame anchored to this specific FITS file's observer position and time — the correct frame for interpreting the arcsecond offsets from solar disk center. If `fov_arcsec` is null, a default 300×300 arcsecond field of view is used (a conservative estimate that captures most active region events).

If `smap.submap()` raises an exception (e.g., the requested coordinates fall outside the map extent), the function returns the full map and logs a warning. The companion JSON records `strategy: "strategy_a"` regardless, so downstream tools can identify which events had fallback behaviour.

### Strategy B — Limb position
**Function:** `apply_strategy_b(smap, limb)` — `scripts/stage2_sdo_query.py`

Applied when `confidence="medium"` and `limb_position` is not null.

Looks up the limb position string in the `LIMB_BOXES` constant dict, which maps each cardinal/ordinal position to an approximate Heliprojective bounding box:

| Position | Tx range (arcsec) | Ty range (arcsec) | Typical content |
|----------|-------------------|-------------------|-----------------|
| `NW` | [−800, −100] | [200, 800] | Northwest limb — prominences, loop arcades |
| `SW` | [−800, −100] | [−800, −200] | Southwest limb |
| `NE` | [100, 800] | [200, 800] | Northeast limb |
| `SE` | [100, 800] | [−800, −200] | Southeast limb |
| `N` | [−400, 400] | [400, 900] | North polar region — polar crown filaments |
| `S` | [−400, 400] | [−900, −400] | South polar region |
| `E` | [400, 900] | [−400, 400] | East limb — emerging active regions |
| `W` | [−900, −400] | [−400, 400] | West limb — decaying active regions |
| `disk` | — | — | Falls through to Strategy C |

**Sign convention:** In Helioprojective coordinates, positive Tx is west (right on a standard solar image) and positive Ty is north (up). The boxes are deliberately wide (~700 arcsec) to ensure the region of interest is captured even when the paper's stated position is approximate.

For `limb_position="disk"` or any unrecognized string, `LIMB_BOXES` returns `None` and `apply_strategy_b` returns the full map (equivalent to Strategy C).

The submap call mirrors Strategy A exactly:
```python
bl = SkyCoord(tx_min*u.arcsec, ty_min*u.arcsec, frame=smap.coordinate_frame)
tr = SkyCoord(tx_max*u.arcsec, ty_max*u.arcsec, frame=smap.coordinate_frame)
return smap.submap(bl, top_right=tr)
```

### Strategy C — Full disk
No submap operation is performed. The full-disk normalised map is saved directly. This strategy is used for:
- Events with `confidence="low"` (only a timestamp was reliably extracted)
- Events with `limb_position="disk"` (the paper explicitly describes a full-disk context image)
- Fallback from Strategy B when `limb_position` is unknown

Full-disk images are larger on disk (~4096×4096 pixels for AIA, ~4096×4096 for HMI) but are the most valuable inputs for the downstream Stage 3 feature-matching pipeline, since the full disk provides the most context for RANSAC alignment.

---

## Output Format

### Output PNG

A normalised uint8 grayscale PNG saved as:
```
<output_dir>/<safe_paper_stem>__<event_idx:03d>.png
```

For example:
```
papers/matched/2012-01___Labrosse__N__000.png
```

The image contains the cropped or full-disk solar region, normalised so the minimum pixel value maps to 0 and the maximum to 255.

### Companion JSON

A JSON file with the same stem as the PNG, saved alongside it:

```json
{
  "paper": "2012-01 - Labrosse, N",
  "event_index": 0,
  "strategy": "strategy_a",
  "observation": {
    "timestamp_start": "2010-09-19T08:00:00",
    "instrument": "AIA",
    "wavelength_angstrom": 304,
    "limb_position": "SW",
    "fov_arcsec": [400.0, 400.0],
    "center_tx_arcsec": -550.0,
    "center_ty_arcsec": -250.0,
    "phenomenon": "prominence eruption",
    "confidence": "high"
  },
  "bounds_arcsec": {
    "tx_min": -750.0,
    "ty_min": -450.0,
    "tx_max": -350.0,
    "ty_max": -50.0
  },
  "fits_file": "2012-01___Labrosse__N__000__AIA__304__2010-09-19T08_00_00.fits"
}
```

`bounds_arcsec` is read back from the actual submap's `bottom_left_coord` and `top_right_coord` properties after the `smap.submap()` call, so it reflects the true bounds of the saved PNG — not the requested bounds. These can differ slightly due to sunpy's pixel-snapping when submap boundaries fall between pixels.

### FITS cache

Downloaded FITS files are retained in `--fits_dir`. They are never deleted by the pipeline. Re-running the `query` command reuses cached FITS files without re-downloading, even if the output PNG is deleted and needs to be regenerated.

---

## End-to-End Example

```bash
# 1. Activate the conda environment
conda activate pytorch_jupyter

# 2. Run Stage 1 first to produce metadata (or use existing metadata files)
./tools/extract_plots.sh metadata \
    --pdf_dir papers/raw_pdfs/ \
    --output_dir papers/metadata/

# 3. Run Stage 2
./tools/extract_plots.sh query \
    --metadata_dir papers/metadata/ \
    --fits_dir papers/sdo_fits/ \
    --output_dir papers/matched/
# Output:
#   Loaded 6 observation event(s) from papers/metadata/
#   [strategy_a   ]  2012-01 - Labrosse, N [000]  2010-09-19T08:00:00  AIA 304Å
#   [strategy_a   ]  2012-01 - Labrosse, N [001]  2010-09-19T08:00:00  AIA 171Å
#   [strategy_b   ]  2012-01 - Song, Y [000]       2012-07-04T09:54:53  HMI noneÅ
#   [strategy_c   ]  2012-01 - Song, Y [001]       2012-07-04T09:54:53  AIA 193Å
#   [strategy_c   ]  2012-01 - Song, Y [002]       2012-07-04T10:02:00  AIA 193Å
#   [failed       ]  2012-01 - White, R [000]      2012-01-15T06:30:00  AIA 193Å
#
#   Summary (6 events):
#     Strategy A (high confidence) : 2
#     Strategy B (medium confidence): 1
#     Strategy C (full disk)        : 2
#     Skipped (already done)        : 0
#     Failed                        : 1

# 4. Inspect the outputs
ls papers/matched/
# 2012-01___Labrosse__N__000.png
# 2012-01___Labrosse__N__000.json
# 2012-01___Labrosse__N__001.png
# ...

cat papers/matched/2012-01___Labrosse__N__000.json

# 5. Re-run is safe — skips all events whose PNGs already exist
./tools/extract_plots.sh query \
    --metadata_dir papers/metadata/ \
    --fits_dir papers/sdo_fits/ \
    --output_dir papers/matched/
# Output:
#   Loaded 6 observation event(s) from papers/metadata/
#   [skipped      ]  2012-01 - Labrosse, N [000]  ...
#   [skipped      ]  2012-01 - Labrosse, N [001]  ...
#   ...
#   Summary (6 events):
#     Skipped (already done) : 5
#     Failed                 : 1
```

---

## Known Limitations

1. **VSO availability and timeouts:** The Virtual Solar Observatory is an external service. Queries can fail due to network timeouts, VSO maintenance windows, or temporary unavailability of a specific data provider. Failed events are logged and can be retried by re-running the stage (their output PNGs do not exist, so they are not skipped).

2. **Empty VSO results for older data:** Not all SDO observations are indexed in the VSO with complete wavelength and instrument metadata. If `Fido.search` returns zero results with a wavelength filter, consider re-running with a broader query by temporarily removing `wavelength_angstrom` from the metadata (or adjusting the ±60 second time window in the source code).

3. **FITS file size:** AIA full-resolution FITS files are ~67 MB each; HMI files can be larger. A batch of 2 000 events would require ~130 GB of cache storage. The `sdo_fits/` cache directory should be on a high-capacity filesystem. The pipeline never re-downloads existing cache files, so partial batches accumulate correctly.

4. **Strategy A defaults to 300×300 arcsec FOV:** When `center_tx_arcsec` and `center_ty_arcsec` are available but `fov_arcsec` is null, the submap defaults to a 300×300 arcsecond field. This is an approximately 150×150 Mm patch at disk center — enough to contain most active regions but potentially too small for large-scale structures (filament channels, coronal holes). If the downstream matching fails, check the companion JSON to see whether this default was applied.

5. **Strategy B boxes are approximate:** The limb bounding boxes in `LIMB_BOXES` are fixed and do not adjust for the solar P-angle or B0 angle (Earth's heliographic latitude and tilt relative to the Sun). Near solstice, the solar north pole can be displaced by up to ~7° from the image vertical. For precise work, Strategy A coordinates from the paper should always be preferred over Strategy B limb boxes.

6. **`disk` limb position falls through to Strategy C:** Papers that state their observation covers the full disk are assigned `limb_position="disk"` by the LLM. These events skip Strategy B and save the full-disk map as Strategy C. This is the correct behavior, but the full-disk PNG is larger and requires more processing in the downstream feature-matching stage.

7. **Companion JSON `bounds_arcsec` may differ from requested bounds:** sunpy snaps submap boundaries to the nearest pixel, so the actual bounding box in the companion JSON may differ from the requested coordinates by a fraction of an arcsecond (typically < 0.5 arcsec for AIA at 0.6 arcsec/pixel). This is expected and does not affect the downstream matching pipeline.

8. **HMI signed magnetic data:** HMI line-of-sight magnetograms contain negative pixel values (south-pointing fields). `normalize_to_uint8` applies `np.abs()` before normalisation, which makes the PNG symmetric around zero (both polarities appear bright). If polarity information is needed for downstream tasks, the original FITS file in `sdo_fits/` preserves the signed data.
