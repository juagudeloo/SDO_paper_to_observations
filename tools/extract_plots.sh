#!/usr/bin/env bash
# =============================================================================
# tools/extract_plots.sh — SDO Plot Extraction Pipeline
# =============================================================================
#
# Entry point for the four-stage pipeline that:
#   1. Lists SDO papers from the NASA ADS SDO API in a date range (-> CSV + Markdown)
#   2. Downloads a paper PDF and extracts solar observation images into the
#      canonical output layout (images/, papers/)
#   3. Extracts observation metadata (time, wavelength, coordinates) via an LLM
#   4. Queries the SDO/VSO archive and produces cropped submaps
#
# USAGE
#   List papers:
#     ./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01
#     ./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01 --format md --output papers
#
#   Extract images from a specific paper (PDF kept in papers/ by default):
#     ./tools/extract_plots.sh extract --id 2620529
#     ./tools/extract_plots.sh extract --id 2620529 --output-dir ./output
#     ./tools/extract_plots.sh extract --id 2620529 --source arxiv --no-keep-pdf
#
#   Extract observation metadata (one paper, or all):
#     ./tools/extract_plots.sh metadata --paper-name "2012-01 - Labrosse, N"
#     ./tools/extract_plots.sh metadata --all
#
#   Query SDO/VSO and produce cropped submaps:
#     ./tools/extract_plots.sh query --paper-name "2012-01 - Labrosse, N"
#     ./tools/extract_plots.sh query --all
#
#   Run unit tests:
#     ./tools/extract_plots.sh test
#
#   Manage the NASA ADS SDO API (needed by list/extract):
#     ./tools/extract_plots.sh api start
#     ./tools/extract_plots.sh api status
#     ./tools/extract_plots.sh api stop
#
# ENVIRONMENT VARIABLES
#   SDO_API_URL   Override API base URL (default: http://localhost:8000)
#   CONDA_ENV     Override conda environment name (default: pytorch_jupyter)
#
# PREREQUISITES
#   - Conda environment 'pytorch_jupyter' with: cv2, requests, PIL, numpy
#   - pymupdf      (pip install pymupdf)
#   - transformers + bitsandbytes  [required for the metadata stage LLM]
#   - sunpy + astropy              [required for the query stage]
#   - NASA ADS SDO API running:
#       ./tools/extract_plots.sh api start
#     (one-time setup: cd nasa_ads_sdo && ./setup.sh — separate venv, not conda)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
CONDA_ENV="${CONDA_ENV:-pytorch_jupyter}"

# The NASA ADS SDO API lives in-repo under nasa_ads_sdo/ with its own isolated
# venv (fastapi/uvicorn/sqlmodel) — unrelated to $CONDA_ENV.
NASA_ADS_SDO_DIR="$PROJECT_ROOT/nasa_ads_sdo"
API_PIDFILE="$NASA_ADS_SDO_DIR/.api.pid"
API_LOGFILE="$NASA_ADS_SDO_DIR/api_server.log"
API_URL="${SDO_API_URL:-http://localhost:8000}"

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
  extract   Extract solar observation images into the canonical output layout
  metadata  Extract structured observation metadata from a paper via LLM
  query     Query SDO/VSO archive and produce cropped submaps
  api       Manage the NASA ADS SDO API service (start | stop | status)
  test      Run unit tests

USAGE:
  list:
    ./tools/extract_plots.sh list --start YYYY-MM-DD --end YYYY-MM-DD \
        [--output FILE] [--format csv|md|both] [--api-url URL] [--verbose]

  extract:
    ./tools/extract_plots.sh extract --id PAPER_ID \
        [--output-dir DIR] [--api-url URL] \
        [--source arxiv|publisher] [--no-keep-pdf] \
        [--save-all] [--min-score 0.25] [--verbose] \
        [--if-exists ask|overwrite|skip] [--purge-downstream]

  metadata:
    ./tools/extract_plots.sh metadata (--paper-name NAME | --all) \
        [--output-dir DIR] [--model Qwen/Qwen2.5-14B-Instruct] [--verbose]

  query:
    ./tools/extract_plots.sh query (--paper-name NAME | --all) \
        [--output-dir DIR] [--fits-dir DIR]

  api:
    ./tools/extract_plots.sh api start    # launch the API in the background
    ./tools/extract_plots.sh api stop     # stop it
    ./tools/extract_plots.sh api status   # check whether it's running
    (one-time setup: cd nasa_ads_sdo && ./setup.sh)

EXAMPLES:
  # List papers from 2012-01-02 to 2013-03-01 (saves CSV + Markdown to output/searched_papers/)
  ./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01

  # Save only Markdown
  ./tools/extract_plots.sh list --start 2012-01-02 --end 2013-03-01 --format md

  # Extract images from the Labrosse paper (PDF kept in output/papers/ by default)
  ./tools/extract_plots.sh extract --id 2620529

  # Extract with a custom output root
  ./tools/extract_plots.sh extract --id 2620529 --output-dir ./my_output

  # Extract observation metadata from the Labrosse paper
  ./tools/extract_plots.sh metadata --paper-name "2012-01 - Labrosse, N"

  # Query SDO/VSO and produce cropped submaps for every processed paper
  ./tools/extract_plots.sh query --all

ENVIRONMENT VARIABLES:
  SDO_API_URL   API base URL  (default: http://localhost:8000)
  CONDA_ENV     Conda env     (default: pytorch_jupyter)

Before running list/extract, start the API:
  ./tools/extract_plots.sh api start
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

check_bitsandbytes() {
    if ! conda run -n "$CONDA_ENV" python3 -c "import bitsandbytes" 2>/dev/null; then
        echo "ERROR: 'bitsandbytes' is not installed in conda env '$CONDA_ENV'." >&2
        echo "       Install with:" >&2
        echo "         conda activate $CONDA_ENV" >&2
        echo "         pip install bitsandbytes accelerate" >&2
        exit 1
    fi
}

check_sunpy() {
    if ! conda run -n "$CONDA_ENV" python3 -c "import sunpy, astropy" 2>/dev/null; then
        echo "ERROR: 'sunpy' or 'astropy' is not installed in conda env '$CONDA_ENV'." >&2
        echo "       Install with:" >&2
        echo "         conda activate $CONDA_ENV" >&2
        echo "         pip install sunpy astropy" >&2
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
# API service lifecycle (nasa_ads_sdo/) — isolated from $CONDA_ENV entirely,
# so this never goes through check_system_tools/check_conda_env.
# ---------------------------------------------------------------------------

_api_pid_alive() {
    [[ -f "$API_PIDFILE" ]] || return 1
    local pid
    pid="$(cat "$API_PIDFILE" 2>/dev/null)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

cmd_api() {
    local sub="${1:-}"
    case "$sub" in
        start)
            if _api_pid_alive; then
                echo "NASA ADS SDO API is already running (pid $(cat "$API_PIDFILE")) at $API_URL"
                exit 0
            fi
            rm -f "$API_PIDFILE"

            if [[ ! -x "$NASA_ADS_SDO_DIR/run_api.sh" ]]; then
                echo "ERROR: nasa_ads_sdo/run_api.sh not found (one-time setup required)." >&2
                echo "       Run:" >&2
                echo "         cd nasa_ads_sdo && ./setup.sh" >&2
                exit 1
            fi

            # `set -m` in a subshell gives the backgrounded job its own process
            # group (PGID == PID), so `stop` can later signal the whole tree
            # (run_api.sh's shell + the uvicorn --reload supervisor/worker it
            # execs in the foreground) rather than just the wrapper process.
            ( set -m
              nohup "$NASA_ADS_SDO_DIR/run_api.sh" >"$API_LOGFILE" 2>&1 < /dev/null &
              echo $! > "$API_PIDFILE"
            )

            local pid waited=0
            pid="$(cat "$API_PIDFILE")"
            while (( waited < 30 )); do
                if curl -sf "$API_URL/" >/dev/null 2>&1; then
                    echo "NASA ADS SDO API is up at $API_URL (pid $pid)"
                    exit 0
                fi
                sleep 0.5
                waited=$((waited + 1))
            done
            echo "ERROR: API did not respond within 15s — check $API_LOGFILE" >&2
            exit 1
            ;;
        stop)
            if [[ ! -f "$API_PIDFILE" ]]; then
                echo "NASA ADS SDO API is not running (no pidfile)."
                exit 0
            fi
            local pid
            pid="$(cat "$API_PIDFILE" 2>/dev/null)"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                echo "NASA ADS SDO API was not running (stale pidfile removed)."
                rm -f "$API_PIDFILE"
                exit 0
            fi

            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
            local waited=0
            while (( waited < 10 )) && kill -0 "$pid" 2>/dev/null; do
                sleep 0.5
                waited=$((waited + 1))
            done
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
            fi
            # Defensive cleanup in case anything escaped the process group.
            # Matched on the venv's own uvicorn path specifically (not a loose
            # "nasa_ads_sdo.*uvicorn" substring) — that pattern is broad enough
            # to accidentally match unrelated commands merely mentioning both
            # words (e.g. a ps/grep diagnostic), which would kill the wrong
            # process.
            pkill -f "$NASA_ADS_SDO_DIR/venv/bin/uvicorn" 2>/dev/null || true
            rm -f "$API_PIDFILE"
            echo "NASA ADS SDO API stopped (pid $pid)."
            exit 0
            ;;
        status)
            if _api_pid_alive; then
                local pid
                pid="$(cat "$API_PIDFILE")"
                if curl -sf "$API_URL/" >/dev/null 2>&1; then
                    echo "NASA ADS SDO API: running (pid $pid) at $API_URL"
                else
                    echo "NASA ADS SDO API: process running (pid $pid) but not responding at $API_URL"
                fi
                exit 0
            else
                echo "NASA ADS SDO API: not running"
                exit 1
            fi
            ;;
        *)
            echo "Usage: ./tools/extract_plots.sh api {start|stop|status}" >&2
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    usage
fi

COMMAND="$1"
shift

# api never touches $CONDA_ENV (it has its own isolated venv), so it's
# dispatched before check_system_tools/check_conda_env — cmd_api always
# exits itself.
if [[ "$COMMAND" == "api" ]]; then
    cmd_api "$@"
    exit $?  # defensive: cmd_api's branches always exit themselves, but never
             # let a future bug fall through into the ML-pipeline checks below
fi

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
    metadata)
        check_transformers
        check_bitsandbytes
        conda run -n "$CONDA_ENV" python3 "$SCRIPTS_DIR/metadata_extraction.py" "$@"
        ;;
    query)
        check_sunpy
        conda run -n "$CONDA_ENV" python3 "$SCRIPTS_DIR/sdo_query.py" "$@"
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
        echo "       Use: list | extract | metadata | query | api | test" >&2
        echo "       Run './tools/extract_plots.sh --help' for usage." >&2
        exit 1
        ;;
esac
