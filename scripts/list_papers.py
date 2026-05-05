#!/usr/bin/env python3
"""
list_papers.py — List SDO papers in a date range from the NASA ADS SDO API.

Outputs a CSV and/or Markdown file. Files are saved under output/searched_papers/
by default.

Usage:
  python3 list_papers.py --start 2012-01-02 --end 2013-03-01
  python3 list_papers.py --start 2012-01-02 --end 2013-03-01 --output my_papers
  python3 list_papers.py --start 2012-01-02 --end 2013-03-01 --format csv
  python3 list_papers.py --start 2012-01-02 --end 2013-03-01 --api-url http://localhost:8000
"""

import argparse
import csv
import logging
import os
import sys

# Allow running as a script from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.api_client import (
    check_api_health,
    get_api_base_url,
    get_documents_for_year,
    pub_date_in_range,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "id",
    "title",
    "authors",
    "publication_date",
    "doi",
    "bibcode",
    "citation_count",
    "ads_url",
]

TITLE_MAX_LEN = 80

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "output", "searched_papers"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List SDO papers from the NASA ADS SDO API in a date range."
    )
    parser.add_argument(
        "--start",
        required=True,
        metavar="YYYY-MM-DD",
        help="Start date (inclusive), e.g. 2012-01-02",
    )
    parser.add_argument(
        "--end",
        required=True,
        metavar="YYYY-MM-DD",
        help="End date (inclusive), e.g. 2013-03-01",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help=(
            "Output file base path, without extension "
            "(default: output/searched_papers/papers_YYYYMMDD_YYYYMMDD). "
            "Extensions are added automatically based on --format."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["csv", "md", "both"],
        default="both",
        help="Output format: csv, md, or both (default: both)",
    )
    parser.add_argument(
        "--api-url",
        metavar="URL",
        default=None,
        help="API base URL (default: $SDO_API_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def validate_date(date_str: str, name: str) -> None:
    """Validate YYYY-MM-DD format."""
    parts = date_str.split("-")
    if len(parts) != 3:
        print(f"ERROR: {name} must be in YYYY-MM-DD format, got: {date_str}", file=sys.stderr)
        sys.exit(1)
    try:
        int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        print(f"ERROR: {name} has non-numeric parts: {date_str}", file=sys.stderr)
        sys.exit(1)


def truncate_title(title: str, max_len: int = TITLE_MAX_LEN) -> str:
    if len(title) > max_len:
        return title[: max_len - 3] + "..."
    return title


def write_markdown(docs: list, path: str, start_date: str, end_date: str) -> None:
    """Write papers to a human-readable Markdown file, including abstracts."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# SDO Papers — {start_date} to {end_date}\n\n")
        fh.write(f"**Total:** {len(docs)} papers\n\n---\n\n")
        for doc in docs:
            title = doc.get("title") or "(no title)"
            authors = doc.get("authors") or "(see paper)"
            pub_date = doc.get("publication_date", "")
            doi = doc.get("doi") or ""
            bibcode = doc.get("bibcode") or ""
            citations = doc.get("citation_count")
            ads_url = doc.get("ads_url") or ""
            abstract = (doc.get("abstract") or "").strip()

            fh.write(f"## {title}\n\n")
            fh.write(f"- **ID:** {doc.get('id', '')}\n")
            fh.write(f"- **Authors:** {authors}\n")
            fh.write(f"- **Date:** {pub_date}\n")
            if doi:
                fh.write(f"- **DOI:** [{doi}](https://doi.org/{doi})\n")
            if bibcode:
                fh.write(f"- **Bibcode:** `{bibcode}`\n")
            if citations is not None:
                fh.write(f"- **Citations:** {citations}\n")
            if ads_url:
                fh.write(f"- **ADS:** [{ads_url}]({ads_url})\n")
            if abstract:
                fh.write(f"\n> {abstract}\n")
            fh.write("\n---\n\n")


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    validate_date(args.start, "--start")
    validate_date(args.end, "--end")

    start_year = int(args.start[:4])
    end_year = int(args.end[:4])
    if start_year > end_year:
        print("ERROR: --start must not be later than --end", file=sys.stderr)
        sys.exit(1)

    api_url = args.api_url or get_api_base_url()
    check_api_health(api_url)

    # Determine output base path (no extension)
    if args.output:
        output_base = args.output
    else:
        start_compact = args.start.replace("-", "")
        end_compact = args.end.replace("-", "")
        output_base = os.path.join(
            os.path.abspath(DEFAULT_OUTPUT_DIR),
            f"papers_{start_compact}_{end_compact}",
        )

    os.makedirs(os.path.dirname(output_base), exist_ok=True)

    # Fetch documents year by year
    print(f"Querying API at {api_url} ...")
    all_docs = []
    for year in range(start_year, end_year + 1):
        docs = get_documents_for_year(api_url, year)
        logger.debug("Year %d: %d documents fetched", year, len(docs))
        # Filter to date range
        in_range = [
            d for d in docs
            if pub_date_in_range(d.get("publication_date", ""), args.start, args.end)
        ]
        logger.debug("Year %d: %d in range", year, len(in_range))
        all_docs.extend(in_range)

    # Sort by publication date, then by id
    all_docs.sort(key=lambda d: (d.get("publication_date", ""), d.get("id", 0)))

    print(f"Found {len(all_docs)} papers between {args.start} and {args.end}")

    if args.format in ("csv", "both"):
        csv_path = output_base + ".csv"
        # Write CSV (UTF-8 with BOM for Excel compatibility)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for doc in all_docs:
                row = {
                    "id": doc.get("id", ""),
                    "title": truncate_title(doc.get("title", "")),
                    "authors": doc.get("authors") or "(see paper)",
                    "publication_date": doc.get("publication_date", ""),
                    "doi": doc.get("doi") or "",
                    "bibcode": doc.get("bibcode") or "",
                    "citation_count": doc.get("citation_count") or "",
                    "ads_url": doc.get("ads_url") or "",
                }
                writer.writerow(row)
        print(f"Saved CSV to: {csv_path}")

    if args.format in ("md", "both"):
        md_path = output_base + ".md"
        write_markdown(all_docs, md_path, args.start, args.end)
        print(f"Saved Markdown to: {md_path}")


if __name__ == "__main__":
    main()
