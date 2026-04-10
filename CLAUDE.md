# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a doctoral research project for matching solar observations from NASA's SDO (Solar Dynamics Observatory) satellite with images from published scientific papers. The goal is to enable coordinate and feature-level mapping between paper figures and actual observational data.

## Environment & Dependencies

There is no top-level `requirements.txt` or `environment.yml`. The project uses:
- **sunpy** — solar data access (Fido/VSO interface) and FITS file handling
- **astropy** — astronomical coordinate transformations (Heliprojective, SkyCoord)
- **OpenCV (cv2)** — classical feature detection (SIFT/ORB) and RANSAC alignment
- **LightGlue** (local submodule in `LightGlue/`) — deep learning-based feature matcher (ICCV 2023)
- **PyTorch + CUDA** — GPU acceleration for LightGlue
- **scikit-image**, **matplotlib**, **numpy**

To install LightGlue (local submodule):
```bash
cd LightGlue && pip install -e .
```

## Running Notebooks

Notebooks live in `notebooks/`. Run with Jupyter:
```bash
jupyter notebook notebooks/1-sunpy_images.ipynb
```

Downloaded SDO data is cached in `notebooks/data/sunpy_images/` as FITS files.

## Architecture & Workflow

The main workflow is in `notebooks/1-sunpy_images.ipynb`:

1. **Data download** — Queries SDO/HMI data via `sunpy.net.Fido` (Virtual Solar Observatory). Target timestamp: `2012-07-04T09:54:53Z ± 20s`, instrument: HMI continuum intensity.

2. **Coordinate conversion** — Loads FITS files via `sunpy.map.Map`, converts pixel ↔ Heliprojective coordinates using `astropy.coordinates.SkyCoord`.

3. **Region extraction** — Selects pixel subregions (e.g., rows 2500–2800, cols 1200–1800), extracts submaps using pixel-to-world conversion.

4. **Image preprocessing** — Normalizes raw HMI data (handles signed magnetic data via absolute value) to uint8 using `cv2.normalize`.

5. **Feature matching — two pipelines:**
   - **Classical (OpenCV):** Bilateral filter → SIFT (fallback: ORB) → kNN matching with Lowe's ratio test (threshold 0.65) → RANSAC affine estimation. Outputs scale, rotation, and translation.
   - **Deep learning (LightGlue):** Loads images as GPU tensors → DISK keypoint extraction → LightGlue neural matcher → matched keypoint coordinates.

6. **Outputs** — Normalized full-disk images, warped image pairs, match visualizations, and affine transformation parameters.

## Key Files

- `notebooks/1-sunpy_images.ipynb` — Main analysis notebook (entire pipeline)
- `LightGlue/lightglue/` — LightGlue Python package (DISK extractor + matcher)
- `images/` — Input images: `paper_image.png` (from publication), `image_0.png` (full SDO), `image_1.png` (cropped region)
- `notebooks/data/sunpy_images/` — Cached downloaded FITS files
- `pdfs/` — Reference scientific paper

## Notes

- LightGlue requires a GPU; the notebook loads images as CUDA tensors.
- SIFT may be unavailable in some OpenCV builds (patent restrictions); ORB is the fallback.
- The VS Code workspace (`SDO_paper_to_observations.code-workspace`) also includes `../NASA_ADS_SDO` as a related project folder.