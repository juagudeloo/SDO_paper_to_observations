#!/usr/bin/env bash
# =============================================================================
# tools/extract_plots.sh — SDO Plot Extraction Pipeline
# =============================================================================
#
# Entry point for the three-stage pipeline that:
#   1. Lists SDO papers from the NASA ADS SDO API in a date range (-> CSV + Markdown)
#   2. Downloads a paper PDF and extracts solar observation images
#   3. Links extracted images to figure captions and classifies solar structures
#
# USAGE
#   List papers:
#     ./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01
#     ./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01 --format md --output papers
#
#   Extract images from a specific paper:
#     ./tools/extract_plots.sh extract --id 15004866
#     ./tools/extract_plots.sh extract --id 15004866 --output-dir ./output
#     ./tools/extract_plots.sh extract --id 15004866 --source arxiv --keep-pdf
#
#   Label images with captions and solar structure:
#     ./tools/extract_plots.sh label --paper-dir output/papers/2012-01\ -\ Labrosse,\ N
#     ./tools/extract_plots.sh label --paper-dir output/papers/2012-01\ -\ Labrosse,\ N --output-dir output
#
#   Run unit tests:
#     ./tools/extract_plots.sh test
#
# ENVIRONMENT VARIABLES
#   SDO_API_URL   Override API base URL (default: http://localhost:8000)
#   CONDA_ENV     Override conda environment name (default: pytorch_jupyter)
#
# PREREQUISITES
#   - Conda environment 'pytorch_jupyter' with: cv2, requests, PIL, numpy
#   - pymupdf      (pip install pymupdf)
#   - transformers (pip install transformers>=4.30.0)  [required for label stage]
#   - NASA ADS SDO API running:
#       cd ../NASA_ADS_SDO && ./run_api.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
CONDA_ENV="${CONDA_ENV:-pytorch_jupyter}"

# On systems where the OS libstdc++ is older than what conda packages require
# (e.g. GLIBCXX_3.4.29 missing), conda run spawns a fresh subprocess that
# picks up the system library before the conda env's copy.  Prepend the env's
# lib/ directory so its libstdc++ wins.  CONDA_PREFIX is set automatically
# when the env is activated; fall back to the standard path otherwise.
_CONDA_LIB="${CONDA_PREFIX:-${HOME}/.conda/envs/${CONDA_ENV}}/lib"
if [[ -d "$_CONDA_LIB" ]]; then
    export LD_LIBRARY_PATH="${_CONDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
SDO Plot Extraction Pipeline

COMMANDS:
  list      List papers in a date range and save to CSV
  extract   Extract solar observation images from a specific paper
  label     Link extracted images to captions and classify solar structures
  test      Run unit tests

USAGE:
  list:
    ./tools/extract_plots.sh list --start YYYY-MM-DD --end YYYY-MM-DD \
        [--output FILE] [--format csv|md|both] [--api-url URL] [--verbose]

  extract:
    ./tools/extract_plots.sh extract --id PAPER_ID \
        [--output-dir DIR] [--api-url URL] \
        [--source arxiv|publisher] [--keep-pdf] \
        [--save-all] [--min-score 0.25] [--verbose]

  label:
    ./tools/extract_plots.sh label --paper-dir PAPER_DIR \
        [--output-dir DIR] [--verbose]

EXAMPLES:
  # List papers from 2012-01-02 to 2013-03-01 (saves CSV + Markdown to output/searched_papers/)
  ./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01

  # Save only Markdown
  ./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01 --format md

  # Extract images from paper 15004866 (--keep-pdf required for the label stage)
  ./tools/extract_plots.sh extract --id 15004866 --keep-pdf

  # Extract with custom output directory
  ./tools/extract_plots.sh extract --id 15004866 --output-dir ./my_output --keep-pdf

  # Label extracted images with captions and solar structure classification
  ./tools/extract_plots.sh label --paper-dir "output/papers/2012-01 - Labrosse, N"

ENVIRONMENT VARIABLES:
  SDO_API_URL   API base URL  (default: http://localhost:8000)
  CONDA_ENV     Conda env     (default: pytorch_jupyter)

Before running list/extract, start the API:
  cd ../NASA_ADS_SDO && ./run_api.sh
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

check_system_tools() {
    if ! conda run -n "$CONDA_ENV" python3 -c "import fitz" 2>/dev/null; then
        echo "ERROR: PyMuPDF (fitz) is not installed in conda env '$CONDA_ENV'." >&2
        echo "       Install with:" >&2
        echo "         conda activate $CONDA_ENV" >&2
        echo "         pip install pymupdf" >&2
        exit 1
    fi
}

check_transformers() {
    if ! conda run -n "$CONDA_ENV" python3 -c "import transformers" 2>/dev/null; then
        echo "ERROR: 'transformers' is not installed in conda env '$CONDA_ENV'." >&2
        echo "       Install with:" >&2
        echo "         conda activate $CONDA_ENV" >&2
        echo "         pip install transformers>=4.30.0" >&2
        exit 1
    fi
}

check_conda_env() {
    # Try to run a minimal import test in the target environment
    if ! conda run -n "$CONDA_ENV" python3 -c \
        "import cv2, requests; from PIL import Image" \
        2>/dev/null; then
        echo "ERROR: Conda environment '$CONDA_ENV' is missing required packages." >&2
        echo "       Activate it and run:" >&2
        echo "         conda activate $CONDA_ENV" >&2
        echo "         pip install -r $PROJECT_ROOT/requirements_extract.txt" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    usage
fi

COMMAND="$1"
shift

# Validate environment before dispatching
check_system_tools
check_conda_env

case "$COMMAND" in
    list)
        conda run -n "$CONDA_ENV" python3 "$SCRIPTS_DIR/list_papers.py" "$@"
        ;;
    extract)
        conda run -n "$CONDA_ENV" python3 "$SCRIPTS_DIR/extract_plots.py" "$@"
        ;;
    label)
        check_transformers
        conda run -n "$CONDA_ENV" python3 "$SCRIPTS_DIR/label_plots.py" "$@"
        ;;
    test)
        echo "Running unit tests in conda env '$CONDA_ENV' ..."
        conda run -n "$CONDA_ENV" python3 -m unittest discover \
            -s "$PROJECT_ROOT/scripts/tests" \
            -p "test_*.py" \
            -v
        ;;
    *)
        echo "ERROR: Unknown command '$COMMAND'." >&2
        echo "       Use: list | extract | label | test" >&2
        echo "       Run './tools/extract_plots.sh --help' for usage." >&2
        exit 1
        ;;
esac
