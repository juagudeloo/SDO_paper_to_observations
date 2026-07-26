#!/usr/bin/env python3
"""
metadata_extraction.py — Extract structured solar observation metadata from the canonical output layout.

For each paper (addressed by its canonical name 'YYYY-MM - LastName, F'):
  1. Reads <root>/images/<name>/extraction_log.json for the solar images and their
     page-placement bboxes (recorded at extract time — no need to re-parse for them).
  2. Extracts figure captions from <root>/papers/<name>.pdf and matches each solar
     image to the nearest caption, then groups images by figure (for shared context)
     while keeping each REAL saved image as its own observation record — sorted into
     reading order (top-to-bottom, left-to-right) via its logged bbox.
  3. Collects body-text paragraphs that explicitly cite each figure, tables linked to
     it, and any "on YYYY-MM-DD at HH:MM" event date/time mentioned in the body text.
  4. Sends the combined context, plus a classifier hint per image, to a local LLM
     (Qwen2.5-14B-Instruct, 8-bit quantised) — one call per figure, returning exactly
     one metadata object per real image (never more, never fewer).
  5. Writes <root>/metadata/<name>.json containing one observation per real image.

Usage:
  python scripts/metadata_extraction.py --paper-name "2012-01 - Labrosse, N"
  python scripts/metadata_extraction.py --all
  python scripts/metadata_extraction.py --all --output-dir output --model Qwen/Qwen2.5-14B-Instruct
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF — build a Rect from the logged bbox for caption matching
import torch

# Redirect HuggingFace cache to the project's models/ directory.
# Must be set before importing transformers.
os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parent.parent / "models"),
)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import folder_naming as fn
from utils.caption_extractor import (
    Caption,
    Table,
    extract_all_captions,
    extract_all_tables,
    extract_event_datetimes,
    extract_figure_body_refs,
    extract_figure_table_links,
    match_image_to_caption,
    _LABEL_NUM_RE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert solar physicist assistant. A scientific figure is described by its \
caption, the body-text paragraphs that cite it, any referenced tables, and a list of the \
REAL images actually saved for this figure — in reading order (top-to-bottom, then \
left-to-right) — each with a classifier hint (a rough automatic guess at its content, plus \
pixel size). Extract structured metadata as ONE JSON object with an "images" array \
containing EXACTLY one entry per listed image, in that same order. Return ONLY valid JSON \
— no prose, no markdown fences, no extra keys.

Figure-level fields (use null when absent):
  phenomenon            — concise label for the solar structure/event (e.g. "Prominence", "White-light flare")
  active_region         — NOAA active-region number as a string e.g. "11515", or null (shared by all images)
  images                — array of image objects, exactly one per listed image, same order (see below)

Each image object (use null when the information is absent):
  description           — one concise phrase describing this specific image
  image_kind            — one of "intensity", "magnetogram", "difference", "ratio", "lightcurve", "other"
                          ("difference"/"ratio" = derived from two frames; "lightcurve" = a time-series plot)
  instrument            — "AIA", "HMI", "EIT", "LASCO", "XRT", etc., or null
  wavelength_angstrom   — integer e.g. 171, 193, 304, 1600, or null
  limb_position         — one of "NW","SW","NE","SE","N","S","E","W","disk", or null
  fov_arcsec            — [width_float, height_float] or null
  center_tx_arcsec      — float (Heliprojective Tx) or null
  center_ty_arcsec      — float (Heliprojective Ty) or null
  heliographic_location — heliographic position e.g. "S17W08", or null (may differ per image)
  confidence            — "high" if Tx/Ty explicitly given, "medium" if limb+fov known, "low" otherwise

Rules:
- The image count is fixed by what was actually saved during extraction. NEVER add extra
  entries for sub-panels the caption mentions that are not in the provided image list, and
  never omit a listed image — the output array length must equal the input image count.
- Treat each image's classifier hint as a strong prior for image_kind/instrument, but correct
  it using the caption/table text when they clearly disagree (the caption is more reliable for
  content; the hint is more reliable for confirming distinct real images exist).
- Mine the referenced tables and body text for active_region, heliographic_location, and
  instrument — tables often list this observational data explicitly.

Example output (a figure with 2 real saved images):
{"phenomenon": "White-light flare", "active_region": "11515",
 "images": [
   {"description": "AIA 1600 image at peak", "image_kind": "intensity", "instrument": "AIA",
    "wavelength_angstrom": 1600, "limb_position": "disk", "fov_arcsec": null,
    "center_tx_arcsec": null, "center_ty_arcsec": null, "heliographic_location": "S18W29",
    "confidence": "low"},
   {"description": "HMI line-of-sight magnetogram", "image_kind": "magnetogram", "instrument": "HMI",
    "wavelength_angstrom": null, "limb_position": "disk", "fov_arcsec": null,
    "center_tx_arcsec": null, "center_ty_arcsec": null, "heliographic_location": "S18W29",
    "confidence": "low"}
 ]}\
"""

USER_TEMPLATE = """\
Extract the figure metadata (exactly one entry per image listed below, in order).
Return ONLY a JSON object.

Figure: {figure_label}

Caption:
{caption}

Body text paragraphs referencing this figure:
{paragraphs}

Referenced tables (caption + raw contents):
{tables}

Event date/time (extracted separately from the paper's text with high confidence, use as-is
if it matches this figure's event — do not second-guess it): {event_datetime}

Images in this figure, in reading order — return exactly {n_images} entries, same order:
{image_hints}\
"""


# Per-image keys emitted in the output (normalised from the LLM response).
_IMAGE_OBS_KEYS = (
    "description",
    "image_kind",
    "instrument",
    "wavelength_angstrom",
    "limb_position",
    "fov_arcsec",
    "center_tx_arcsec",
    "center_ty_arcsec",
    "heliographic_location",
    "confidence",
)


def _normalize_image_obs(raw: dict) -> dict:
    """Coerce a raw LLM image-object dict to the canonical key set, filling missing keys."""
    obs = {k: raw.get(k) for k in _IMAGE_OBS_KEYS}
    obs["confidence"] = raw.get("confidence") or "low"
    return obs


def _reading_order_key(entry: dict, row_bucket: float = 20.0) -> tuple:
    """
    Sort key for row-major reading order (top-to-bottom, then left-to-right)
    from a logged image's bbox. Bucketing y0 absorbs small sub-pixel
    differences between images in the same visual row while still separating
    genuinely different rows (rows in observed multi-panel figures are tens
    of points apart). PyMuPDF bboxes have their origin at the page's
    top-left, so smaller y0 is higher on the page.
    """
    bbox = entry.get("bbox")
    if not bbox:
        return (float("inf"), float("inf"))
    x0, y0 = bbox[0], bbox[1]
    return (round(y0 / row_bucket), x0)


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _load_log(root: str, name: str) -> dict:
    path = fn.log_path(root, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"extraction_log.json not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _pdf_authors(pdf_path: str) -> str | None:
    """Read author string from PDF document metadata, if available."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        meta = doc.metadata
        doc.close()
        return meta.get("author") or None
    except Exception:
        return None


def _figure_number(figure_label: str) -> int | None:
    """Return the leading integer from a label like 'Figure 2a' → 2."""
    if not figure_label:
        return None
    m = re.search(r"(\d+)", figure_label)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    logger.info("Loading model with 8-bit quantization …")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    logger.info("Model loaded.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _query_model(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    figure_label: str,
    caption_text: str,
    paragraphs: list[str],
    tables_text: str,
    event_datetime: str,
    image_hints: list[str],
) -> str:
    paras_str = "\n".join(f"- {p}" for p in paragraphs) if paragraphs else "(none)"
    hints_str = "\n".join(
        f"{i}. {h}" for i, h in enumerate(image_hints, 1)
    ) if image_hints else "(none)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                figure_label=figure_label or "(unknown)",
                caption=caption_text or "(none)",
                paragraphs=paras_str,
                tables=tables_text or "(none)",
                event_datetime=event_datetime or "(unknown)",
                n_images=len(image_hints),
                image_hints=hints_str,
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,  # a multi-image figure can emit many image objects
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)


def _parse_llm_output(raw: str) -> dict:
    """Parse a JSON object from the model's raw response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip("`\n ")
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return {}


# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------

def process_paper(
    name: str,
    root: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> str:
    """
    Process one paper (by canonical name) and write <root>/metadata/<name>.json.

    Returns one of "success" or "failed". Callers are expected to pre-filter
    papers that already have metadata JSON (see main()).
    """
    out_json = fn.metadata_json(root, name)

    # Load extraction log (carries the solar images + their page-placement bboxes)
    logger.info("Loading extraction log for %s", name)
    try:
        log = _load_log(root, name)
    except FileNotFoundError as exc:
        logger.warning("%s", exc)
        return "failed"

    paper_title = log.get("title", "")
    first_author = log.get("first_author", "")

    # Locate the PDF in the canonical papers/ directory
    pdf_path = fn.pdf_path(root, name)
    pdf_filename = os.path.basename(pdf_path)
    if not os.path.isfile(pdf_path):
        logger.warning("PDF not found: %s (run extract with the PDF kept)", pdf_path)
        _write_result(out_json, pdf_filename, paper_title, first_author, [], "failed")
        return "failed"

    paper_authors = _pdf_authors(pdf_path) or first_author

    # Captions + body refs + tables + event dates come from the PDF; image bboxes
    # come from the log
    try:
        logger.info("Extracting captions from %s", pdf_filename)
        captions_by_page = extract_all_captions(pdf_path)
        logger.info("Extracting figure body references")
        body_refs_by_fig = extract_figure_body_refs(pdf_path)
        logger.info("Extracting tables and figure->table links")
        tables_by_num = extract_all_tables(pdf_path)
        fig_table_links = extract_figure_table_links(pdf_path)
        logger.info("Extracting event date/time mentions")
        event_datetimes = extract_event_datetimes(pdf_path)
        logger.debug(
            "Found %d table(s); figure->table links: %s; event dates: %s",
            len(tables_by_num), fig_table_links, event_datetimes,
        )
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", name, exc)
        _write_result(out_json, pdf_filename, paper_title, paper_authors, [], "failed")
        return "failed"

    # Match every saved image to its caption, then group images by figure so
    # context (caption/tables/paragraphs/event date) is shared, while each REAL
    # saved image still becomes its own observation record — ordered into
    # reading order within its figure via its logged bbox.
    all_entries: list[dict] = sorted(log.get("images", []), key=lambda e: e["index"])
    saved_entries = [e for e in all_entries if e.get("filename")]

    groups = _group_images_by_figure(saved_entries, captions_by_page)
    observations: list[dict] = []

    for i, group in enumerate(groups, 1):
        caption_text = group["caption_text"]
        figure_label = group["figure_label"]
        fig_num = group["fig_num"]
        ordered_entries = sorted(group["entries"], key=_reading_order_key)

        paragraphs = body_refs_by_fig.get(fig_num, []) if fig_num else []

        # Resolve tables linked to this figure and build the table context text
        linked_tables: list[Table] = [
            tables_by_num[n]
            for n in fig_table_links.get(fig_num, [])
            if n in tables_by_num
        ] if fig_num else []
        tables_text = (
            "\n\n".join(f"{t.label}: {t.body_text}" for t in linked_tables)
            if linked_tables else ""
        )
        referenced_tables = [{"label": t.label, "caption": t.caption} for t in linked_tables]

        # An event date is only trustworthy for this figure if that literal
        # date string appears in its own caption (e.g. "...the 2010-06-13
        # prominence eruption...") — otherwise leave it unset rather than
        # guessing which event a figure belongs to.
        event_date = next((d for d in event_datetimes if d in caption_text), None)
        event_time = event_datetimes.get(event_date) if event_date else None
        event_datetime_str = f"{event_date}T{event_time}:00" if event_date and event_time else None

        image_hints = [
            f"classifier guess: {e.get('image_type', 'unknown')}, {e.get('size', '?')}px"
            for e in ordered_entries
        ]

        logger.info(
            "  [%d/%d] %s (%d image(s))%s%s : querying LLM",
            i, len(groups), figure_label or "(no caption)",
            len(ordered_entries),
            f" + {', '.join(t.label for t in linked_tables)}" if linked_tables else "",
            f" + event {event_datetime_str}" if event_datetime_str else "",
        )

        # Query LLM — skip when there is no text context at all
        llm_meta: dict = {}
        if caption_text or paragraphs or tables_text:
            try:
                raw = _query_model(
                    tokenizer, model, figure_label, caption_text, paragraphs,
                    tables_text, event_datetime_str, image_hints,
                )
                llm_meta = _parse_llm_output(raw)
            except Exception as exc:
                logger.warning("LLM query failed for %s / %s: %s", name, figure_label, exc)

        # Normalise the images array — force its length to match the real
        # image count exactly (truncate extras, pad missing with nulls) so
        # every observation below is backed by a real saved image and no
        # image is ever left without one, regardless of what the LLM returned.
        raw_images = llm_meta.get("images")
        if not isinstance(raw_images, list):
            raw_images = []
        raw_images = raw_images[:len(ordered_entries)]
        raw_images += [{}] * (len(ordered_entries) - len(raw_images))
        image_objs = [_normalize_image_obs(o if isinstance(o, dict) else {}) for o in raw_images]

        for seq, (entry, obs) in enumerate(zip(ordered_entries, image_objs)):
            observations.append({
                "observation_filename": entry.get("filename"),
                "figure": _figure_number(figure_label),
                "figure_label": figure_label,
                "sequence_index": seq,
                "is_solar": entry.get("is_solar"),
                "classifier_score": entry.get("score"),
                "caption": caption_text,
                "paragraphs": paragraphs,
                "referenced_tables": referenced_tables,
                "phenomenon": llm_meta.get("phenomenon"),
                "active_region": llm_meta.get("active_region"),
                "timestamp_start": event_datetime_str,
                **obs,
            })

    status = "success" if observations else "failed"
    _write_result(out_json, pdf_filename, paper_title, paper_authors, observations, status)
    return status


def _group_images_by_figure(
    saved_entries: list[dict],
    captions_by_page: dict,
) -> list[dict]:
    """
    Group saved images by their matched figure caption.

    Each saved image is matched to its nearest caption (via the bbox recorded at
    extract time). Images that share a figure number are grouped together; images
    that match no caption each form their own single-image group (keyed on filename)
    so nothing is dropped.

    Returns a list of group dicts, in first-appearance order, each with keys:
    ``fig_num`` (str, "" when no caption), ``figure_label``, ``caption_text``,
    and ``entries`` (the log entries in the group).
    """
    groups: dict[str, dict] = {}
    order: list[str] = []

    for entry in saved_entries:
        caption: Caption | None = None
        bbox = entry.get("bbox")
        if bbox:
            img_rect = fitz.Rect(*bbox)
            caption, _conf = match_image_to_caption(
                entry["page"], img_rect, captions_by_page
            )
        caption_text = caption.text if caption else ""
        figure_label = caption.figure_label if caption else ""

        num_match = _LABEL_NUM_RE.search(figure_label) if figure_label else None
        fig_num = num_match.group(1) if num_match else ""

        # Key: figure number when known, else a unique key per uncaptioned image
        key = fig_num if fig_num else f"__img__{entry.get('filename')}"
        if key not in groups:
            groups[key] = {
                "fig_num": fig_num,
                "figure_label": figure_label,
                "caption_text": caption_text,
                "entries": [],
            }
            order.append(key)
        groups[key]["entries"].append(entry)

    return [groups[k] for k in order]


def _write_result(
    path: str,
    paper: str,
    paper_title: str,
    paper_authors: str,
    observations: list[dict],
    status: str,
) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "paper": paper,
                "paper_authors": paper_authors,
                "paper_title": paper_title,
                "observations": observations,
                "status": status,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract solar observation metadata from the canonical output layout."
    )
    parser.add_argument(
        "--paper-name",
        default=None,
        metavar="NAME",
        help="Canonical paper name to process (e.g. '2012-01 - Labrosse, N')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every paper found under <root>/images/",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Root of the canonical output layout (default: output). Reads "
             "images/ and papers/, writes metadata/.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-14B-Instruct",
        metavar="MODEL",
        help="HuggingFace model identifier (default: Qwen/Qwen2.5-14B-Instruct)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable step-by-step DEBUG logging (caption/table extraction, per-image LLM queries)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.paper_name and not args.all:
        print("ERROR: one of --paper-name or --all is required", file=sys.stderr)
        sys.exit(1)

    root = args.output_dir
    os.makedirs(fn.metadata_dir(root), exist_ok=True)

    # Collect paper names to process
    if args.paper_name:
        names = [args.paper_name]
    else:
        names = fn.iter_paper_names(root)

    if not names:
        print(f"No papers found under {fn.images_root(root)}")
        return

    print(f"Found {len(names)} paper(s) to process", flush=True)

    # Split into pending vs already-done BEFORE loading the model — a paper that
    # already has metadata JSON should never trigger the (~8 min) model load.
    pending: list[str] = []
    n_skipped = 0
    for name in names:
        if os.path.exists(fn.metadata_json(root, name)):
            n_skipped += 1
            print(f"  [skip]  {name}", flush=True)
        else:
            pending.append(name)

    if not pending:
        print(
            f"\nSummary: 0 processed, {n_skipped} skipped, 0 failed"
            f"  (total: {len(names)}) — nothing to do, model not loaded",
            flush=True,
        )
        return

    tokenizer, model = load_model(args.model)

    n_processed = n_failed = 0

    for i, name in enumerate(pending, 1):
        logger.info("Processing paper %d/%d: %s", i, len(pending), name)
        status = process_paper(name, root, tokenizer, model)

        if status == "success":
            n_processed += 1
            try:
                with open(fn.metadata_json(root, name), encoding="utf-8") as fh:
                    data = json.load(fh)
                n_obs = len(data.get("observations", []))
            except Exception:
                n_obs = 0
            print(f"  [ok]    {name}  ({n_obs} observation(s))", flush=True)
        else:
            n_failed += 1
            print(f"  [fail]  {name}", flush=True)

    print(
        f"\nSummary: {n_processed} processed, {n_skipped} skipped, {n_failed} failed"
        f"  (total: {len(names)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
