# SDO Paper to Observation Pipeline Documentation

This pipeline is designed to automate the process of matching solar images found in scientific papers to the original high-resolution observations from the **Solar Dynamics Observatory (SDO)**, specifically using **HMI Continuum** data.

---

## 🚀 Execution Parameters

You can run the pipeline from the terminal using `python src/pipeline.py`. Below are the available command-line arguments:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--paper` | `str` | **Required** | Path to the image file extracted from the paper (e.g., `images/paper_image.png`). |
| `--date` | `str` | **Required** | The target UTC date and time for the observation (e.g., `'2012-07-04T09:54:53Z'`). |
| `--downsample` | `float` | `1.0` | **(deprecated)** uniform scaling factor applied to **both** images. Use the newer flags if you need fine-grained control. |
+| `--downsample-paper` | `float` | `1.0` | Downsample factor applied only to the paper image. |
+| `--downsample-original` | `float` | `1.0` | Downsample factor applied only to the HMI observation. |
| `--conf` | `float` | `0.1` | Confidence threshold for the LightGlue matcher (0.0 to 1.0). |
| `--outdir` | `str` | `./outputs` | Directory where results and visualizations will be saved. |
| `--show` | `flag` | `False` | If included, displays the result summary window at the end of the run. |

---

## ⚡ The `--downsample` Parameter (Speed Optimization)

SDO images are natively **4096 x 4096 pixels**. Extracting AI features (DISK) and matching them (LightGlue) on these full-resolution images is computationally expensive and slow (can take several minutes on CPU).

The `--downsample` parameter historically allowed you to perform **fast testing** by shrinking both images. That behavior is still available for backwards compatibility, but you now have more control:

- `--downsample-original 0.25`: downscales only the 4096×4096 HMI map to 1024×1024, greatly speeding up computation while leaving the paper scan untouched.
- `--downsample-paper 0.5`: if your paper image is large (e.g. a full‑page scan), you can speed things up by shrinking it independently.
- `--downsample 0.5`: (deprecated) applies the same factor to both images, exactly like before.

Typical recommendations:
- Use `--downsample-original` for most testing (0.25 or 0.5) and leave the paper at 1.0.
- Reserve `--downsample-paper` only if the scan itself is massive and you know it still contains sufficient detail.
- `--downsample 1.0` (or omitting the flag) uses full resolution on both images.

> **Warning**: Using a value that reduces the paper image below ~128 pixels on a side will normally be skipped automatically, but it may still produce too few keypoints if the image has very little structure.
---

## 🛠️ Pipeline Architecture

The pipeline consists of four main stages:

### 1. Data Fetching (`data_fetcher.py`)
Uses `sunpy` to query the JSOC/VSO database. It automatically searches for the closest HMI Continuum FITS file matching your target date, downloads it, and caches it locally to avoid repeated downloads.

### 2. Preprocessing (`preprocessor.py`)
- **SDO Data**: Converts raw FITS data (which can be 32-bit float) into a normalized 8-bit grayscale format compatible with computer vision models.
- **Paper Image**: Ensures the input image is grayscale and normalized.
- **Padding**: Both images are padded to be multiples of 16, a strict requirement for the **DISK** feature extraction model.

### 3. Feature Matching (`matcher.py`)
This is the "brain" of the pipeline:
- **DISK**: A deep-learning based local feature detector that extracts robust keypoints from solar textures.
- **LightGlue**: A high-speed attention-based matcher that finds correspondences between the paper image and the SDO observation.
- **RANSAC**: An algorithm that filters out "noise" matches and estimates the **Affine Transform** (rotation, scale, and translation) needed to align the images.

### 4. Visualization & Reporting (`visualizer.py`)
Generates the final outputs in the `outputs/` folder:
- **Aligned Overlay**: The paper image transformed to sit exactly on top of the SDO observation.
- **Summary Plot**: A side-by-side comparison showing the paper image, the SDO map, matched keypoints, and the resulting alignment.

---

## 📁 Output Artifacts

After a successful run, you will find:
- `outputs/lightglue_summary_final.png`: The main summary figure.
- `data/sunpy_images/`: The downloaded raw FITS files from NASA/SDO.
