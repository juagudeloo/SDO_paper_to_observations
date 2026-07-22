#!/usr/bin/env python3
"""
extract_plots.py — Download a paper PDF and extract solar observation images.

Given a paper ID from the NASA ADS SDO database:
  1. Fetch document metadata from the API.
  2. Download the PDF.
  3. Extract all embedded images (with their page placement bbox).
  4. Classify each image as a solar observation or not.
  5. Lay down the canonical output layout, keyed by the paper name
     'YYYY-MM - LastName, F':
       <root>/images/<name>/*.png                   (solar images)
       <root>/images/<name>/extraction_log.json     (per-image log, incl. bbox)
       <root>/papers/<name>.pdf                      (kept unless --no-keep-pdf)

The bbox recorded per image lets the downstream `metadata` stage match each
image to its figure caption without re-parsing the PDF.

Usage:
  python3 extract_plots.py --id 2620529
  python3 extract_plots.py --id 2620529 --output-dir ./output --source arxiv
  python3 extract_plots.py --id 2620529 --no-keep-pdf --min-score 0.25
  python3 extract_plots.py --id 2620529 --if-exists overwrite --purge-downstream
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from glob import glob

# Allow running as a script from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.api_client import check_api_health, get_api_base_url, get_document_by_id, download_pdf
from utils import folder_naming as fn
from utils.folder_naming import build_folder_name, parse_first_author
from utils.pdf_extractor import extract_pdf_images, extract_first_page_text
from utils.solar_classifier import classify_image

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
        "--no-keep-pdf",
        dest="keep_pdf",
        action="store_false",
        help="Do not copy the PDF into <root>/papers/ (later stages need it, so"
             " it is kept by default)",
    )
    parser.set_defaults(keep_pdf=True)
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
    parser.add_argument(
        "--if-exists",
        choices=["ask", "overwrite", "skip"],
        default="ask",
        metavar="{ask,overwrite,skip}",
        help="What to do if this paper ID was already extracted: prompt "
             "interactively (default; aborts if stdin is not a TTY), overwrite "
             "the existing images, or skip extraction entirely",
    )
    parser.add_argument(
        "--purge-downstream",
        action="store_true",
        help="When overwriting, also delete this paper's existing metadata JSON "
             "and matched/ outputs (regenerating them is expensive — an ~8 min "
             "model load and fresh VSO downloads). Without this flag they are "
             "left in place with a warning that they may now be stale.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Existing-paper detection and overwrite/skip handling
# ---------------------------------------------------------------------------
#
# Mirrors sdo_query.py's _safe() (kept independent rather than imported, since
# that stage is owned/evolved separately): sanitizes a canonical paper name for
# matching the "<safe_name>__*" prefix its output files are saved under.
_SAFE_RE = re.compile(r"[^\w\-]")


def _find_existing_paper_by_id(root: str, paper_id: int) -> tuple[str, dict] | None:
    """
    Scan existing extraction logs for a paper with the given ID.

    Reads only <root>/images/*/extraction_log.json (no network calls), since
    paper_id is already recorded there by a previous `extract` run.

    Returns:
        (canonical_name, log_data) if found, else None.
    """
    for name in fn.iter_paper_names(root):
        try:
            with open(fn.log_path(root, name), encoding="utf-8") as fh:
                log = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if log.get("paper_id") == paper_id:
            return name, log
    return None


def _find_downstream_artifacts(root: str, name: str) -> tuple[str | None, list[str]]:
    """Return (metadata_json_path or None, matched_output_paths) for a paper name."""
    meta_path = fn.metadata_json(root, name)
    meta = meta_path if os.path.isfile(meta_path) else None

    safe_name = _SAFE_RE.sub("_", name)
    matched = sorted(glob(os.path.join(fn.matched_dir(root), f"{safe_name}__*")))

    return meta, matched


def _warn_stale(meta_path: str | None, matched_files: list[str]) -> None:
    parts = []
    if meta_path:
        parts.append("metadata JSON")
    if matched_files:
        parts.append(f"{len(matched_files)} matched output(s)")
    if parts:
        print(
            f"WARNING: existing {' and '.join(parts)} for this paper were NOT "
            "deleted and may now be stale (they reference the old extraction). "
            "Re-run metadata/query for this paper if needed, or pass "
            "--purge-downstream next time.",
            file=sys.stderr,
        )


def _purge_downstream(meta_path: str | None, matched_files: list[str]) -> None:
    if meta_path and os.path.isfile(meta_path):
        os.remove(meta_path)
        print(f"  Removed stale metadata: {meta_path}")
    for f in matched_files:
        os.remove(f)
    if matched_files:
        print(f"  Removed {len(matched_files)} stale matched output(s)")


def _resolve_overwrite(
    root: str,
    existing_name: str,
    existing_log: dict,
    if_exists: str,
    purge_downstream: bool,
) -> str:
    """
    Decide whether to overwrite or skip an already-extracted paper.

    Prompts interactively when if_exists == "ask" and stdin is a TTY; exits
    the process with an error if confirmation is required but stdin is not
    interactive (e.g. running under a script/cron without --if-exists set
    explicitly).

    Returns "overwrite" or "skip".
    """
    n_saved = existing_log.get("solar_images_saved", "?")

    if if_exists == "skip":
        return "skip"

    if if_exists == "overwrite":
        do_overwrite = True
    else:  # "ask"
        if not sys.stdin.isatty():
            print(
                f"ERROR: paper id already extracted as '{existing_name}' "
                f"({n_saved} image(s) saved) and stdin is not interactive. "
                "Pass --if-exists overwrite or --if-exists skip explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        answer = input(
            f"Paper '{existing_name}' already exists ({n_saved} image(s) saved). "
            "Overwrite? [y/N]: "
        )
        do_overwrite = answer.strip().lower() in ("y", "yes")

    if not do_overwrite:
        return "skip"

    meta_path, matched_files = _find_downstream_artifacts(root, existing_name)
    if meta_path or matched_files:
        if purge_downstream:
            _purge_downstream(meta_path, matched_files)
        elif if_exists == "ask" and sys.stdin.isatty():
            desc = []
            if meta_path:
                desc.append("metadata.json")
            if matched_files:
                desc.append(f"{len(matched_files)} matched output(s)")
            answer = input(
                f"Also delete existing {' and '.join(desc)} for this paper? [y/N]: "
            )
            if answer.strip().lower() in ("y", "yes"):
                _purge_downstream(meta_path, matched_files)
            else:
                _warn_stale(meta_path, matched_files)
        else:
            _warn_stale(meta_path, matched_files)

    return "overwrite"


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- 0. Check for an existing extraction of this paper ID ---
    # Done before any network call: paper_id is already recorded in each
    # paper's extraction_log.json, so a skip costs nothing but a local scan.
    found = _find_existing_paper_by_id(args.output_dir, args.id)
    if found is not None:
        existing_name, existing_log = found
        decision = _resolve_overwrite(
            args.output_dir, existing_name, existing_log, args.if_exists, args.purge_downstream
        )
        if decision == "skip":
            print(f"Skipping — paper id {args.id} already extracted as '{existing_name}'.")
            return
        # decision == "overwrite": clear the old images dir + PDF first, so
        # stale files from a previous run (e.g. a different --min-score) don't
        # linger alongside the new ones.
        shutil.rmtree(fn.images_dir(args.output_dir, existing_name), ignore_errors=True)
        old_pdf = fn.pdf_path(args.output_dir, existing_name)
        if os.path.isfile(old_pdf):
            os.remove(old_pdf)
        print(f"Overwriting existing extraction of '{existing_name}'.\n")

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

        # --- 4. Build canonical output layout ---
        folder_name = build_folder_name(
            doc.get("publication_date", "0000-00-00"),
            last_name,
            first_initial,
        )
        output_folder = fn.images_dir(args.output_dir, folder_name)
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
                "bbox": list(img_meta.bbox) if img_meta.bbox else None,
                "size": f"{img_meta.width}x{img_meta.height}",
                "color_space": img_meta.color_space,
                "encoding": img_meta.encoding,
                "is_solar": result.is_solar,
                "score": round(result.score, 3),
                "signals": result.signals,
                "image_type": result.image_type,
                "filename": None,  # set below for images that get saved
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
                solar_images.append((img_meta, result, entry))

        # --- 7. Save solar images ---
        saved_count = 0
        img_prefix = "img" if args.save_all else "solar"
        for i, (img_meta, result, entry) in enumerate(solar_images):
            dest_name = (
                f"{img_prefix}_{i + 1:03d}"
                f"_p{img_meta.page}"
                f"_{result.image_type}.png"
            )
            dest_path = os.path.join(output_folder, dest_name)
            if img_meta.file_path and os.path.isfile(img_meta.file_path):
                shutil.copy(img_meta.file_path, dest_path)
                entry["filename"] = dest_name
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
        log_file = fn.log_path(args.output_dir, folder_name)
        with open(log_file, "w", encoding="utf-8") as fh:
            json.dump(log_data, fh, indent=2, ensure_ascii=False)

        mode_label = "saved (no filter)" if args.save_all else "classified as solar observations"
        print(
            f"\nExtraction complete: {saved_count}/{len(image_list)} images {mode_label}"
        )
        print(f"Output folder: {output_folder}")
        print(f"Log: {log_file}")

        # --- Keep the PDF in the canonical papers/ dir (needed by later stages) ---
        if args.keep_pdf:
            dest_pdf = fn.pdf_path(args.output_dir, folder_name)
            os.makedirs(os.path.dirname(dest_pdf), exist_ok=True)
            shutil.copy(pdf_path, dest_pdf)
            print(f"PDF kept at: {dest_pdf}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
