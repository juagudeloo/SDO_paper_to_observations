"""
api_client.py — NASA ADS SDO API communication.

Handles health checking, pagination, document retrieval, and PDF download.
The API base URL is read from:
  1. SDO_API_URL environment variable
  2. Default: http://localhost:8000
"""

import os
import sys
import logging
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NASA_ADS_SDO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "NASA_ADS_SDO"
)
_START_HINT = (
    f"Start the API with:\n"
    f"  cd {os.path.abspath(NASA_ADS_SDO_DIR)}\n"
    f"  ./run_api.sh"
)


def get_api_base_url() -> str:
    """Return API base URL from env var or default."""
    return os.environ.get("SDO_API_URL", "http://localhost:8000").rstrip("/")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_api_health(base_url: str) -> bool:
    """
    Check that the API is reachable and returns a valid response.

    Raises SystemExit with instructions if the API cannot be reached.
    Returns True on success.
    """
    try:
        resp = requests.get(f"{base_url}/", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if "message" not in data:
            raise ValueError("Unexpected API response format")
        logger.debug("API healthy at %s (version: %s)", base_url, data.get("version"))
        return True
    except requests.ConnectionError:
        print(
            f"\nERROR: Cannot connect to NASA ADS SDO API at {base_url}\n"
            f"{_START_HINT}\n",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.Timeout:
        print(
            f"\nERROR: Timeout connecting to NASA ADS SDO API at {base_url}\n"
            f"{_START_HINT}\n",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(
            f"\nERROR: API health check failed at {base_url}: {exc}\n"
            f"{_START_HINT}\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Document retrieval
# ---------------------------------------------------------------------------

def get_documents_for_year(
    base_url: str, year: int, max_page_size: int = 1000
) -> list:
    """
    Fetch all documents for a given year with automatic pagination.

    Args:
        base_url: API base URL.
        year: Publication year to filter.
        max_page_size: Number of documents per API request (max 1000).

    Returns:
        List of document dicts for that year.
    """
    results = []
    skip = 0
    while True:
        resp = requests.get(
            f"{base_url}/documents/",
            params={"year": year, "skip": skip, "limit": max_page_size},
            timeout=30,
        )
        resp.raise_for_status()
        page = resp.json()
        results.extend(page)
        logger.debug(
            "Year %d: fetched %d documents (skip=%d)", year, len(page), skip
        )
        if len(page) < max_page_size:
            break
        skip += max_page_size
    return results


def get_document_by_id(base_url: str, doc_id: int) -> dict:
    """
    Fetch a single document by ID.

    Raises:
        ValueError: If the document is not found (HTTP 404).
        requests.HTTPError: For other HTTP errors.
    """
    resp = requests.get(f"{base_url}/documents/{doc_id}", timeout=15)
    if resp.status_code == 404:
        raise ValueError(f"Document with ID {doc_id} not found in the database.")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------

def download_pdf(
    base_url: str,
    doc_id: int,
    dest_path: str,
    source: str = None,
) -> str:
    """
    Download the PDF for a document and save it to dest_path.

    Tries arXiv first, then publisher (unless source is specified).
    If neither source has a PDF, prints a warning and raises RuntimeError.

    Args:
        base_url: API base URL.
        doc_id: Document ID.
        dest_path: Local file path to save the PDF.
        source: Optional preferred source ('arxiv' or 'publisher').

    Returns:
        dest_path on success.

    Raises:
        RuntimeError: If no PDF is available from any source.
    """
    params = {}
    if source:
        params["source"] = source

    url = f"{base_url}/documents/{doc_id}/download-pdf"
    logger.debug("Downloading PDF from %s (params=%s)", url, params)

    try:
        resp = requests.get(url, params=params, timeout=60, stream=True)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Connection error downloading PDF for document {doc_id}: {exc}"
        ) from exc

    if resp.status_code == 404:
        detail = resp.json().get("detail", "No PDF available")
        raise RuntimeError(
            f"PDF not available for document {doc_id}: {detail}\n"
            f"Try accessing the paper at the ADS abstract page."
        )
    resp.raise_for_status()

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    total_bytes = 0
    with open(dest_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)
                total_bytes += len(chunk)

    size_mb = total_bytes / (1024 * 1024)
    logger.info("PDF downloaded: %s (%.2f MB)", dest_path, size_mb)
    return dest_path


# ---------------------------------------------------------------------------
# Date range helper (used by list_papers.py)
# ---------------------------------------------------------------------------

def pub_date_in_range(pub_date: str, start_date: str, end_date: str) -> bool:
    """
    Check whether a publication date falls within [start_date, end_date].

    Handles the DB format 'YYYY-MM-00' where day is always 00.
    Comparison is done at month granularity.

    Args:
        pub_date: Publication date string from DB (e.g. '2012-07-00').
        start_date: Range start as 'YYYY-MM-DD'.
        end_date: Range end as 'YYYY-MM-DD'.

    Returns:
        True if pub_date falls within the range.
    """
    try:
        year = int(pub_date[:4])
        month_str = pub_date[5:7] if len(pub_date) >= 7 else "01"
        month = int(month_str) if month_str != "00" else 1

        start_year = int(start_date[:4])
        start_month = int(start_date[5:7])
        end_year = int(end_date[:4])
        end_month = int(end_date[5:7])

        pub_ym = (year, month)
        start_ym = (start_year, start_month)
        end_ym = (end_year, end_month)

        return start_ym <= pub_ym <= end_ym
    except (ValueError, IndexError):
        return False
