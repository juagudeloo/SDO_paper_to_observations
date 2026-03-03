# SDO Paper to Observation Alignment Pipeline

This project replicates the matching logic from `1-sunpy_images.ipynb` but packages it into a cleaner, more modular Python pipeline using `.py` files. It bridges the gap between static *paper pictures* and raw *SDO/HMI Continuum observations* using deep learning feature matching (**DISK + LightGlue**).

## Structure
- `src/data_fetcher.py`: Automates acquiring FITS files from the JSOC/VSO database using Sunpy `Fido`.
- `src/preprocessor.py`: Normalizes raw array data and formats paper figures. Fits images to multiples of 16 for deep learning models.
- `src/matcher.py`: Uses `kornia` (DISK + LightGlue) to aggressively identify matched feature points between vastly different contrast scales.
- `src/visualizer.py`: Visualizes the inlier features matches, plots overlays, bounds the matching active area (AR), and dumps a 4-panel visual report.
- `src/pipeline.py`: The CLI wrapper uniting all these into an automated task.

## Prerequisites
You need the following installed:
```bash
pip install torch opencv-python numpy sunpy astropy matplotlib scikit-image kornia
```
Ensure you have PyTorch set up to use GPU/CUDA if you desire faster feature matching. LightGlue works on CPU, but is much faster on CUDA.

## Usage

Use the command line to execute the pipeline:

```bash
python src/pipeline.py --paper "path/to/your/paper_figure.png" --date "2012-07-04T09:54:53Z"
```

To display the summary figure in a window while running:

```bash
python src/pipeline.py --paper "path/to/your/paper_figure.png" --date "2012-07-04T09:54:53Z" --show
```

### Arguments

- `--paper` (Required): The path to the image taken from the publication.
- `--date` (Required): A valid UTC target date to download corresponding HMI FITS data (`%Y-%m-%dT%H:%M:%SZ`).
- `--outdir` (Optional): The destination directory for output graphs. Default is `./outputs`.
- `--conf` (Optional): The confidence threshold for the LightGlue features. Default `0.1`.

## Outputs
- `matches_lightglue.jpg`: Displays only the top RANSAC inlier features connecting the paper and observation.
- `rect_lightglue.jpg`: Raw observation marked with the localized Region of Interest (ROI).
- `ar_crop_lightglue.jpg`: The fully cropped observation representing the active region isolated.
- `overlay_lightglue.jpg`: Alpha-blended overlay between transformed paper image and observation.
- `lightglue_summary_final.png`: A cleanly rendered 4-panel subplot describing the progression.

By default, all outputs are saved under `./outputs` (or your `--outdir` path), and the pipeline prints each saved visualization path in the console.
