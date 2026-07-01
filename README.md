# SDO — Paper to Observations

A pipeline that matches solar observations from NASA's SDO (Solar Dynamics Observatory)
satellite with images published in scientific papers. Given a corpus of SDO papers, it:

1. finds the papers,
2. extracts the solar-observation figures (recording each figure's position on the page),
3. reads the observational metadata (time, wavelength, coordinates) — matching each figure to
   its caption and citing paragraphs — out of the surrounding text with an LLM, and
4. re-queries the SDO/VSO archive to reproduce the matching observation as a cropped submap.

## Requirements

- A conda environment named **`pytorch_jupyter`** (override with the `CONDA_ENV` variable).
- Python packages: `pip install -r requirements_extract.txt`
  - The `metadata` stage also needs `bitsandbytes` and `accelerate`, plus a GPU.
  - The `query` stage also needs `sunpy` and `astropy`.
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
environment and validates dependencies for you. The pipeline has four stages that run in order,
each consuming the previous stage's output. Everything lands under a canonical layout inside the
output root (default `output/`):

```
output/
  papers/    <name>.pdf                         # the kept PDF
  images/    <name>/*.png + extraction_log.json # extracted solar images + log
  metadata/  <name>.json                        # observation metadata
  matched/   ...                                # cropped submaps
```

where `<name>` is the paper's canonical name, e.g. `2012-01 - Labrosse, N`. After `extract`, the
later stages address a paper by that name (`--paper-name`) or process them all (`--all`).

```bash
# 1. List SDO papers in a date range  ->  output/searched_papers/
./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01

# 2. Extract solar images from one paper (by its API id)  ->  output/images/<name>/, output/papers/<name>.pdf
#    The PDF is kept by default (later stages need it); pass --no-keep-pdf to skip it.
./tools/extract_plots.sh extract --id 2620529

# 3. Extract observation metadata (time, wavelength, coordinates) with an LLM  ->  output/metadata/
./tools/extract_plots.sh metadata --paper-name "2012-01 - Labrosse, N"

# 4. Query the SDO/VSO archive and produce cropped submaps  ->  output/matched/
./tools/extract_plots.sh query --paper-name "2012-01 - Labrosse, N"
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
./tools/extract_plots.sh extract --id 2620529 --source arxiv --min-score 0.25

# Process every extracted paper at once
./tools/extract_plots.sh metadata --all
./tools/extract_plots.sh query --all
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
- [`docs/IMAGE_CAPTION_PIPELINE.md`](docs/IMAGE_CAPTION_PIPELINE.md) — figure ↔ caption matching
- [`docs/METADATA_EXTRACTION.md`](docs/METADATA_EXTRACTION.md) — LLM metadata extraction
- [`docs/SDO_QUERY.md`](docs/SDO_QUERY.md) — SDO/VSO querying and submap cropping

If you want to understand or modify how a stage works, start with its doc.

> Note: the design docs under `docs/` still describe the earlier five-stage flow (with a separate
> `label` stage and BART structure classifier). The algorithms they cover largely still apply, but
> the stage wiring now follows the four-stage pipeline above.
