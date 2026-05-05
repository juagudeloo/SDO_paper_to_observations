#!/usr/bin/env python3
"""
label_plots.py — Link extracted solar images to figure captions and classify solar structures.

Given a paper folder produced by the extract stage (output/papers/<folder_name>/):
  1. Reads extraction_log.json to get the list of solar observation images.
  2. Extracts figure captions from the paper PDF using PyMuPDF.
  3. Matches each solar image to its nearest caption by vertical proximity.
  4. Classifies the solar structure described in each caption (NLP model).
  5. Copies files to a canonical folder layout:
       output/papers/<folder_name>.pdf       (flat PDF copy)
       output/images/<folder_name>/          (solar image copies)
       output/labels/<folder_name>.csv       (per-paper CSV)

Usage:
  python3 label_plots.py --paper-dir "output/papers/2012-01 - Labrosse, N"
  python3 label_plots.py --paper-dir "output/papers/2012-01 - Labrosse, N" --output-dir output --verbose
"""

import argparse
import csv
import glob
import json
import logging
import os
import shutil
import sys
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

# Allow running as a script from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.caption_extractor import Caption, extract_all_captions, get_all_image_bboxes, match_image_to_caption
from utils.structure_classifier_nlp import classify_structure

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "paper_filename",
    "paper_path",
    "image_path",
    "figure_label",
    "caption_text",
    "structure_label",
    "structure_confidence",
    "caption_match_confidence",
    "image_type",
    "classifier_score",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link solar images to captions and classify solar structures."
    )
    parser.add_argument(
        "--paper-dir",
        required=True,
        metavar="DIR",
        help="Paper folder produced by the extract stage "
             "(e.g. output/papers/2012-01 - Labrosse, N)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Base output directory (default: ./output)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_extraction_log(paper_dir: str) -> dict:
    log_path = os.path.join(paper_dir, "extraction_log.json")
    if not os.path.isfile(log_path):
        raise FileNotFoundError(
            f"extraction_log.json not found in '{paper_dir}'.\n"
            "Run './tools/extract_plots.sh extract --id PAPER_ID' first."
        )
    with open(log_path, encoding="utf-8") as fh:
        return json.load(fh)


def _find_pdf(paper_dir: str) -> str:
    """Return the path to the first paper_*.pdf found in paper_dir."""
    matches = glob.glob(os.path.join(paper_dir, "paper_*.pdf"))
    if not matches:
        raise FileNotFoundError(
            f"No paper_*.pdf found in '{paper_dir}'.\n"
            "Re-run the extract stage with the --keep-pdf flag:\n"
            "  ./tools/extract_plots.sh extract --id PAPER_ID --keep-pdf"
        )
    return matches[0]


def _solar_image_filename(entry: dict, solar_rank: int) -> str:
    """Reconstruct the filename for the Nth solar image (1-based rank)."""
    return f"solar_{solar_rank:03d}_p{entry['page']}_{entry['image_type']}.png"


def _posix_rel(output_dir: str, *parts: str) -> str:
    """Build a forward-slash relative path from output_dir down through parts."""
    return str(PurePosixPath(output_dir).joinpath(*parts))


def _copy_idempotent(src: str, dst: str) -> None:
    """Copy src to dst; skip if dst already exists with the same size."""
    if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return
    shutil.copy2(src, dst)


def _build_global_bbox_list(bboxes_by_page: dict) -> list:
    """
    Flatten bboxes_by_page into a global list ordered by page then encounter order.

    The Nth item in this list corresponds to the log entry with index=N because
    both extract_pdf_images() and get_all_image_bboxes() iterate pages and images
    in the same order with the same seen_xrefs deduplication.
    """
    flat = []
    for page_num in sorted(bboxes_by_page.keys()):
        flat.extend(bboxes_by_page[page_num])  # list of (xref, fitz.Rect)
    return flat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    paper_dir = os.path.normpath(args.paper_dir)
    output_dir = args.output_dir
    folder_name = os.path.basename(paper_dir)

    # --- 1. Validate inputs and load log ---
    if not os.path.isdir(paper_dir):
        print(f"ERROR: paper directory not found: '{paper_dir}'", file=sys.stderr)
        sys.exit(1)

    try:
        log = _load_extraction_log(paper_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLabeling paper: {folder_name}")
    print(f"  Title       : {log.get('title', '(no title)')[:80]}")
    print(f"  First author: {log.get('first_author', '(unknown)')}")

    # All images in global index order (needed for bbox correlation)
    all_entries: List[dict] = sorted(log.get("images", []), key=lambda e: e["index"])

    # Solar images only, preserving index order (rank = position in this list + 1)
    solar_entries: List[dict] = [e for e in all_entries if e.get("is_solar")]

    if not solar_entries:
        print("  WARNING: No solar images found in extraction log. Writing empty CSV.")

    print(f"  Solar images: {len(solar_entries)} / {len(all_entries)} extracted")

    # --- 2. Find PDF ---
    try:
        pdf_path = _find_pdf(paper_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Create output directories ---
    images_dir = os.path.join(output_dir, "images", folder_name)
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    # output/papers/ is assumed to already exist (created by the extract stage)
    os.makedirs(os.path.join(output_dir, "papers"), exist_ok=True)

    # --- 4. Copy PDF to output/papers/<folder_name>.pdf ---
    dest_pdf = os.path.join(output_dir, "papers", folder_name + ".pdf")
    _copy_idempotent(pdf_path, dest_pdf)
    print(f"  PDF copied  : {dest_pdf}")

    # --- 5. Copy solar images to output/images/<folder_name>/ ---
    solar_rank = 0  # tracks position among solar images for filename reconstruction
    img_src_paths: Dict[int, str] = {}  # entry["index"] → source path

    for entry in all_entries:
        if not entry.get("is_solar"):
            continue
        solar_rank += 1
        filename = _solar_image_filename(entry, solar_rank)
        src = os.path.join(paper_dir, filename)
        dst = os.path.join(images_dir, filename)
        if os.path.isfile(src):
            _copy_idempotent(src, dst)
            img_src_paths[entry["index"]] = dst
        else:
            logger.warning("Solar image file missing: %s", src)
            img_src_paths[entry["index"]] = ""

    print(f"  Images dir  : {images_dir}")

    # --- 6. Extract captions and image bboxes from the PDF ---
    print("  Extracting captions from PDF...")
    captions_by_page = extract_all_captions(pdf_path)
    total_captions = sum(len(v) for v in captions_by_page.values())
    print(f"    Found {total_captions} figure caption(s) across {len(captions_by_page)} page(s)")

    bboxes_by_page = get_all_image_bboxes(pdf_path)
    global_bboxes = _build_global_bbox_list(bboxes_by_page)

    # --- 7. Match each solar image to its caption and classify structure ---
    print("  Loading NLP model for structure classification...")
    csv_rows = []
    solar_rank = 0

    for entry in all_entries:
        if not entry.get("is_solar"):
            continue
        solar_rank += 1
        global_idx = entry["index"]
        filename = _solar_image_filename(entry, solar_rank)

        # Look up bbox for this image using its global index
        img_rect = None
        if global_idx < len(global_bboxes):
            _xref, img_rect = global_bboxes[global_idx]

        # Match to caption
        caption: Optional[Caption] = None
        match_confidence = "none"
        if img_rect is not None:
            caption, match_confidence = match_image_to_caption(
                entry["page"], img_rect, captions_by_page
            )
        else:
            logger.warning(
                "No bbox found for image index %d (page %d); skipping caption match.",
                global_idx,
                entry["page"],
            )

        # Classify solar structure from caption text
        caption_text = caption.text if caption else ""
        structure_label, structure_confidence = classify_structure(caption_text or None)

        # Build relative paths (forward-slash)
        paper_filename = folder_name + ".pdf"
        paper_path_rel = _posix_rel(output_dir, "papers", paper_filename)
        image_path_rel = (
            _posix_rel(output_dir, "images", folder_name, filename)
            if img_src_paths.get(global_idx)
            else ""
        )

        row = {
            "paper_filename": paper_filename,
            "paper_path": paper_path_rel,
            "image_path": image_path_rel,
            "figure_label": caption.figure_label if caption else "",
            "caption_text": caption_text,
            "structure_label": structure_label,
            "structure_confidence": structure_confidence,
            "caption_match_confidence": match_confidence,
            "image_type": entry.get("image_type", "unknown"),
            "classifier_score": entry.get("score", 0.0),
        }
        csv_rows.append(row)

        if args.verbose:
            print(
                f"    solar_{solar_rank:03d} p{entry['page']}"
                f"  caption={caption.figure_label if caption else '(none)'!r}"
                f"  [{match_confidence}]"
                f"  → {structure_label} ({structure_confidence:.2f})"
            )

    # --- 8. Write CSV ---
    csv_path = os.path.join(labels_dir, folder_name + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nLabeling complete: {len(csv_rows)} row(s) written")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
