# SDO — Paper to Observations

A pipeline that matches solar observations from NASA's SDO (Solar Dynamics Observatory)
satellite with images published in scientific papers. Given a corpus of SDO papers, it:

1. finds the papers,
2. extracts the solar-observation figures,
3. links each figure to its caption and classifies the solar structure,
4. reads the observational metadata (time, wavelength, coordinates) out of the surrounding
   text with an LLM, and
5. re-queries the SDO/VSO archive to reproduce the matching observation as a cropped submap.

## Requirements

- A conda environment named **`pytorch_jupyter`** (override with the `CONDA_ENV` variable).
- Python packages: `pip install -r requirements_extract.txt`
  - Stage 4 (`metadata`) also needs `bitsandbytes` and `accelerate`, plus a GPU.
  - Stage 5 (`query`) also needs `sunpy` and `astropy`.
  - The launcher checks these per command and prints an install hint if anything is missing.
- The **NASA ADS SDO API** running (needed for `list` and `extract`). It lives in the sibling
  repository `../NASA_ADS_SDO`
  ([juagudeloo/NASA_ADS_SDO](https://github.com/juagudeloo/NASA_ADS_SDO)):

  ```bash
  cd ../NASA_ADS_SDO && ./run_api.sh
  ```

  By default the tools talk to `http://localhost:8000`; override with the `SDO_API_URL` variable.

## Running the tools

Everything runs through a single launcher, `tools/extract_plots.sh`, which activates the conda
environment and validates dependencies for you. The pipeline has five stages that run in order,
each consuming the output of the previous one (all output lands under `output/`).

```bash
# 1. List SDO papers in a date range  ->  output/searched_papers/
./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01

# 2. Extract solar images from one paper (by its API id)  ->  output/papers/<paper>/
#    Use --keep-pdf: the later stages need the PDF.
./tools/extract_plots.sh extract --id 2620529 --keep-pdf

# 3. Link images to figure captions + classify the solar structure  ->  output/images/, output/labels/
./tools/extract_plots.sh label --paper-dir "output/papers/2012-01 - Labrosse, N"

# 4. Extract observation metadata (time, wavelength, coordinates) with an LLM  ->  output/metadata/
./tools/extract_plots.sh metadata --paper-dir "output/papers/2012-01 - Labrosse, N" --output_dir output/metadata

# 5. Query the SDO/VSO archive and produce cropped submaps  ->  output/matched/
./tools/extract_plots.sh query --metadata_dir output/metadata/ --fits_dir output/fits/ --output_dir output/matched/
```

Run any command with `--help` (or the launcher with no arguments) to see all options:

```bash
./tools/extract_plots.sh --help
./tools/extract_plots.sh extract --help
```

### Useful variations

```bash
# Save the paper list as Markdown instead of CSV
./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01 --format md

# Extract from arXiv rather than the publisher, and lower the solar-image score threshold
./tools/extract_plots.sh extract --id 2620529 --keep-pdf --source arxiv --min-score 0.25

# Run stage 4 over every paper folder at once
./tools/extract_plots.sh metadata --pdf_dir output/papers/ --output_dir output/metadata
```

### Environment variables

| Variable      | Purpose                              | Default                 |
|---------------|--------------------------------------|-------------------------|
| `SDO_API_URL` | Base URL of the NASA ADS SDO API     | `http://localhost:8000` |
| `CONDA_ENV`   | Conda environment the launcher uses  | `pytorch_jupyter`       |

## Learn more

Each stage has a detailed design document in [`docs/`](docs/) explaining the algorithms and
heuristics behind it:

- [`docs/EXTRACT_PLOTS.md`](docs/EXTRACT_PLOTS.md) — listing papers and extracting solar images
- [`docs/IMAGE_CAPTION_PIPELINE.md`](docs/IMAGE_CAPTION_PIPELINE.md) — caption matching and structure classification
- [`docs/METADATA_EXTRACTION.md`](docs/METADATA_EXTRACTION.md) — LLM metadata extraction
- [`docs/SDO_QUERY.md`](docs/SDO_QUERY.md) — SDO/VSO querying and submap cropping

If you want to understand or modify how a stage works, start with its doc.
