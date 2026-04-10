#!/usr/bin/env bash
# =============================================================================
# tools/extract_plots.sh — SDO Plot Extraction Pipeline
# =============================================================================
#
# Entry point for the two-stage pipeline that:
#   1. Lists SDO papers from the NASA ADS SDO API in a date range (-> CSV + Markdown)
#   2. Downloads a paper PDF and extracts solar observation images
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
#   Run unit tests:
#     ./tools/extract_plots.sh test
#
# ENVIRONMENT VARIABLES
#   SDO_API_URL   Override API base URL (default: http://localhost:8000)
#   CONDA_ENV     Override conda environment name (default: pytorch_jupyter)
#
# PREREQUISITES
#   - Conda environment 'pytorch_jupyter' with: cv2, requests, PIL, numpy
#   - pymupdf  (pip install pymupdf)
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

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
SDO Plot Extraction Pipeline

COMMANDS:
  list      List papers in a date range and save to CSV
  extract   Extract solar observation images from a specific paper
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

EXAMPLES:
  # List papers from 2012-01-02 to 2013-03-01 (saves CSV + Markdown to output/searched_papers/)
  ./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01

  # Save only Markdown
  ./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01 --format md

  # Extract images from paper 15004866
  ./tools/extract_plots.sh extract --id 15004866

  # Extract with custom output directory and keep the PDF
  ./tools/extract_plots.sh extract --id 15004866 --output-dir ./my_output --keep-pdf

ENVIRONMENT VARIABLES:
  SDO_API_URL   API base URL  (default: http://localhost:8000)
  CONDA_ENV     Conda env     (default: pytorch_jupyter)

Before running, start the API:
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
    test)
        echo "Running unit tests in conda env '$CONDA_ENV' ..."
        conda run -n "$CONDA_ENV" python3 -m unittest discover \
            -s "$PROJECT_ROOT/scripts/tests" \
            -p "test_*.py" \
            -v
        ;;
    *)
        echo "ERROR: Unknown command '$COMMAND'." >&2
        echo "       Use: list | extract | test" >&2
        echo "       Run './tools/extract_plots.sh --help' for usage." >&2
        exit 1
        ;;
esac
