#!/usr/bin/env python3
"""
extract_plots.py — Download a paper PDF and extract solar observation images.

Given a paper ID from the NASA ADS SDO database:
  1. Fetch document metadata from the API.
  2. Download the PDF.
  3. Extract all embedded images via pdfimages.
  4. Classify each image as a solar observation or not.
  5. Save solar images to: output_dir/YYYY-MM - LastName, F/
  6. Write an extraction_log.json with full classification details.

Usage:
  python3 extract_plots.py --id 15004866
  python3 extract_plots.py --id 15004866 --output-dir ./output --source arxiv
  python3 extract_plots.py --id 15004866 --keep-pdf --min-score 0.25
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile

# Allow running as a script from repo root
sys.path.insert(0, os.path.dirname(__file__))

from api_client import check_api_health, get_api_base_url, get_document_by_id, download_pdf
from folder_naming import build_folder_name, parse_first_author
from pdf_extractor import extract_pdf_images, extract_first_page_text
from solar_classifier import classify_image

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract solar observation images from an SDO paper PDF."
    )
    parser.add_argument(
        "--id",
        required=True,
        type=int,
        metavar="PAPER_ID",
        help="Paper ID in the NASA ADS SDO database",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Base output directory (default: ./output)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        metavar="URL",
        help="API base URL (default: $SDO_API_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--source",
        choices=["arxiv", "publisher"],
        default=None,
        help="Preferred PDF source (default: auto, tries arXiv then publisher)",
    )
    parser.add_argument(
        "--keep-pdf",
        action="store_true",
        help="Keep the downloaded PDF and extracted PNGs in a temp folder",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save all extracted images without applying the solar classifier",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.25,
        metavar="FLOAT",
        help="Minimum confidence score (0-1) to save an image (default: 0.25); ignored when --save-all is set",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    api_url = args.api_url or get_api_base_url()
    check_api_health(api_url)

    # --- 1. Fetch document metadata ---
    try:
        doc = get_document_by_id(api_url, args.id)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing paper ID {args.id}")
    print(f"  Title     : {doc.get('title', '(no title)')[:80]}")
    print(f"  Published : {doc.get('publication_date', '(unknown)')}")
    print(f"  Bibcode   : {doc.get('bibcode', '(none)')}")

    # --- 2. Download PDF to temp dir ---
    tmp_dir = tempfile.mkdtemp(prefix="sdo_extract_")
    tmp_pdf = os.path.join(tmp_dir, f"paper_{args.id}.pdf")

    try:
        try:
            pdf_path = download_pdf(api_url, args.id, tmp_pdf, source=args.source)
        except RuntimeError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            print(
                f"  ADS page: https://ui.adsabs.harvard.edu/abs/{doc.get('bibcode', '')}",
                file=sys.stderr,
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
            sys.exit(1)

        pdf_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"  PDF size  : {pdf_size_mb:.2f} MB  -> {pdf_path}")

        # --- 3. Parse first author from PDF text ---
        page1_text = extract_first_page_text(pdf_path)
        last_name, first_initial = parse_first_author(
            doc.get("authors", ""),
            doc.get("bibcode", ""),
            page1_text,
        )
        print(f"  Author    : {last_name}, {first_initial}")

        # --- 4. Build output folder ---
        folder_name = build_folder_name(
            doc.get("publication_date", "0000-00-00"),
            last_name,
            first_initial,
        )
        output_folder = os.path.join(args.output_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        print(f"  Output    : {output_folder}")

        # --- 5. Extract images ---
        image_list = extract_pdf_images(pdf_path, tmp_dir)
        print(f"\n  Found {len(image_list)} embedded images in PDF")

        # --- 6. Classify images ---
        solar_images = []
        log_entries = []

        for img_meta in image_list:
            result = classify_image(img_meta)
            entry = {
                "index": img_meta.index,
                "page": img_meta.page,
                "size": f"{img_meta.width}x{img_meta.height}",
                "color_space": img_meta.color_space,
                "encoding": img_meta.encoding,
                "is_solar": result.is_solar,
                "score": round(result.score, 3),
                "signals": result.signals,
                "image_type": result.image_type,
            }
            log_entries.append(entry)

            if args.save_all:
                keep = img_meta.file_path is not None and "too_small" not in result.signals
                status = "save " if keep else "skip "
            else:
                keep = result.is_solar and result.score >= args.min_score
                status = "SOLAR" if keep else "skip "

            print(
                f"    [{status}] img-{img_meta.index:03d} p{img_meta.page}"
                f" {img_meta.width}x{img_meta.height}"
                f" {img_meta.color_space}"
                f" score={result.score:.2f}"
                f" [{', '.join(result.signals[:3])}]"
            )

            if keep:
                solar_images.append((img_meta, result))

        # --- 7. Save solar images ---
        saved_count = 0
        img_prefix = "img" if args.save_all else "solar"
        for i, (img_meta, result) in enumerate(solar_images):
            dest_name = (
                f"{img_prefix}_{i + 1:03d}"
                f"_p{img_meta.page}"
                f"_{result.image_type}.png"
            )
            dest_path = os.path.join(output_folder, dest_name)
            if img_meta.file_path and os.path.isfile(img_meta.file_path):
                shutil.copy(img_meta.file_path, dest_path)
                saved_count += 1
                print(f"  Saved: {dest_name}")
            else:
                logger.warning(
                    "Image file not found for index %d: %s",
                    img_meta.index,
                    img_meta.file_path,
                )

        # --- 8. Write extraction log ---
        log_data = {
            "paper_id": args.id,
            "title": doc.get("title", ""),
            "publication_date": doc.get("publication_date", ""),
            "bibcode": doc.get("bibcode"),
            "doi": doc.get("doi"),
            "ads_url": doc.get("ads_url"),
            "first_author": f"{last_name}, {first_initial}",
            "total_images_found": len(image_list),
            "solar_images_saved": saved_count,
            "min_score_threshold": args.min_score,
            "images": log_entries,
        }
        log_path = os.path.join(output_folder, "extraction_log.json")
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(log_data, fh, indent=2, ensure_ascii=False)

        mode_label = "saved (no filter)" if args.save_all else "classified as solar observations"
        print(
            f"\nExtraction complete: {saved_count}/{len(image_list)} images {mode_label}"
        )
        print(f"Output folder: {output_folder}")
        print(f"Log: {log_path}")

        # --- Keep or clean up temp files ---
        if args.keep_pdf:
            dest_pdf = os.path.join(output_folder, f"paper_{args.id}.pdf")
            shutil.copy(pdf_path, dest_pdf)
            print(f"PDF kept at: {dest_pdf}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
