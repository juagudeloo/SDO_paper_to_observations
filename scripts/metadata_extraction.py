#!/usr/bin/env python3
"""
metadata_extraction.py — Extract structured solar observation metadata from the canonical output layout.

For each paper (addressed by its canonical name 'YYYY-MM - LastName, F'):
  1. Reads <root>/images/<name>/extraction_log.json for the solar images and their
     page-placement bboxes (recorded at extract time — no need to re-parse for them).
  2. Extracts figure captions from <root>/papers/<name>.pdf and matches each solar
     image to the nearest caption by vertical proximity, using the logged bbox.
  3. Collects body-text paragraphs that explicitly cite each figure.
  4. Sends the combined caption + paragraph text per image to a local LLM
     (Qwen2.5-14B-Instruct, 8-bit quantised) to extract observational metadata.
  5. Writes <root>/metadata/<name>.json containing all per-image observations.

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
caption, the body-text paragraphs that cite it, and any referenced tables. The figure \
may contain several sub-panels labelled (a), (b), (c), … — each a DISTINCT observation \
with its own instrument, wavelength, time and role. Extract structured metadata for the \
whole figure as ONE JSON object with a "panels" array. Return ONLY valid JSON — no prose, \
no markdown fences, no extra keys.

Figure-level fields (use null when absent):
  figure_label          — the figure label e.g. "Figure 5"
  phenomenon            — concise label for the solar structure/event (e.g. "Prominence", "White-light flare")
  active_region         — NOAA active-region number as a string e.g. "11515", or null (shared by all panels)
  panels                — array of panel objects (see below)

Each panel object (use null when the information is absent):
  panel                 — panel letter e.g. "a", or null if the figure has no sub-panels
  description           — one concise phrase describing this panel
  image_kind            — one of "intensity", "magnetogram", "difference", "ratio", "lightcurve", "other"
                          ("difference"/"ratio" = derived from two frames; "lightcurve" = a time-series plot)
  instrument            — "AIA", "HMI", "EIT", "LASCO", "XRT", etc., or null
  wavelength_angstrom   — integer e.g. 171, 193, 304, 1600, or null
  timestamp_start       — ISO UTC string e.g. "2012-07-04T09:54:53" or null
  timestamp_end         — ISO UTC string or null
  limb_position         — one of "NW","SW","NE","SE","N","S","E","W","disk", or null
  fov_arcsec            — [width_float, height_float] or null
  center_tx_arcsec      — float (Heliprojective Tx) or null
  center_ty_arcsec      — float (Heliprojective Ty) or null
  heliographic_location — heliographic position e.g. "S17W08", or null (per-panel; may differ)
  derived_from          — for "difference"/"ratio" panels, the source panel letters e.g. ["d","e"], else null
  confidence            — "high" if Tx/Ty explicitly given, "medium" if limb+fov known, "low" otherwise

Rules:
- Enumerate EVERY panel. Expand ranges and lists: "(b)-(e)" → b, c, d, e; "(b) and (d)" → b, d.
- If the caption describes no sub-panels, return exactly one panel with "panel": null.
- Mine the referenced tables and body text for active_region, heliographic_location, timestamps
  and instrument — tables often list this observational data explicitly.

Example output:
{"figure_label": "Figure 5", "phenomenon": "White-light flare", "active_region": "11515",
 "panels": [
   {"panel": "a", "description": "AIA 1600 image at peak", "image_kind": "intensity",
    "instrument": "AIA", "wavelength_angstrom": 1600, "timestamp_start": null, "timestamp_end": null,
    "limb_position": "disk", "fov_arcsec": null, "center_tx_arcsec": null, "center_ty_arcsec": null,
    "heliographic_location": "S18W29", "derived_from": null, "confidence": "low"},
   {"panel": "f", "description": "difference image (peak - beginning)", "image_kind": "difference",
    "instrument": "HMI", "wavelength_angstrom": null, "timestamp_start": null, "timestamp_end": null,
    "limb_position": "disk", "fov_arcsec": null, "center_tx_arcsec": null, "center_ty_arcsec": null,
    "heliographic_location": "S18W29", "derived_from": ["d","e"], "confidence": "low"}
 ]}\
"""

USER_TEMPLATE = """\
Extract the figure metadata (with one entry per panel) for the figure described below.
Return ONLY a JSON object.

Figure: {figure_label}

Caption:
{caption}

Body text paragraphs referencing this figure:
{paragraphs}

Referenced tables (caption + raw contents):
{tables}\
"""


# Per-panel keys emitted in the output (normalised from the LLM response).
_PANEL_KEYS = (
    "panel",
    "description",
    "image_kind",
    "instrument",
    "wavelength_angstrom",
    "timestamp_start",
    "timestamp_end",
    "limb_position",
    "fov_arcsec",
    "center_tx_arcsec",
    "center_ty_arcsec",
    "heliographic_location",
    "derived_from",
    "confidence",
)


def _normalize_panel(raw: dict) -> dict:
    """Coerce a raw LLM panel dict to the canonical key set, filling missing keys."""
    panel = {k: raw.get(k) for k in _PANEL_KEYS}
    panel["confidence"] = raw.get("confidence") or "low"
    return panel


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
    tables_text: str = "",
) -> str:
    paras_str = "\n".join(f"- {p}" for p in paragraphs) if paragraphs else "(none)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                figure_label=figure_label or "(unknown)",
                caption=caption_text or "(none)",
                paragraphs=paras_str,
                tables=tables_text or "(none)",
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
            max_new_tokens=2048,  # a multi-panel figure can emit many panel objects
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

    # Captions + body refs + tables come from the PDF; image bboxes come from the log
    try:
        logger.info("Extracting captions from %s", pdf_filename)
        captions_by_page = extract_all_captions(pdf_path)
        logger.info("Extracting figure body references")
        body_refs_by_fig = extract_figure_body_refs(pdf_path)
        logger.info("Extracting tables and figure->table links")
        tables_by_num = extract_all_tables(pdf_path)
        fig_table_links = extract_figure_table_links(pdf_path)
        logger.debug(
            "Found %d table(s); figure->table links: %s",
            len(tables_by_num), fig_table_links,
        )
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", name, exc)
        _write_result(out_json, pdf_filename, paper_title, paper_authors, [], "failed")
        return "failed"

    # Match every saved image to its caption, then group images by figure so each
    # figure (not each raster) becomes one observation record with a panels[] array.
    all_entries: list[dict] = sorted(log.get("images", []), key=lambda e: e["index"])
    saved_entries = [e for e in all_entries if e.get("filename")]

    groups = _group_images_by_figure(saved_entries, captions_by_page)
    observations: list[dict] = []

    for i, group in enumerate(groups, 1):
        caption_text = group["caption_text"]
        figure_label = group["figure_label"]
        fig_num = group["fig_num"]

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

        logger.info(
            "  [%d/%d] %s (%d image(s))%s : querying LLM",
            i, len(groups), figure_label or "(no caption)",
            len(group["entries"]),
            f" + {', '.join(t.label for t in linked_tables)}" if linked_tables else "",
        )

        # Query LLM — skip when there is no text context at all
        llm_meta: dict = {}
        if caption_text or paragraphs or tables_text:
            try:
                raw = _query_model(
                    tokenizer, model, figure_label, caption_text, paragraphs, tables_text
                )
                llm_meta = _parse_llm_output(raw)
            except Exception as exc:
                logger.warning("LLM query failed for %s / %s: %s", name, figure_label, exc)

        # Normalise panels; guarantee at least one (single-panel fallback)
        raw_panels = llm_meta.get("panels")
        if not isinstance(raw_panels, list) or not raw_panels:
            raw_panels = [{}]
        panels = [_normalize_panel(p if isinstance(p, dict) else {}) for p in raw_panels]

        source_images = [
            {
                "filename": e.get("filename"),
                "is_solar": e.get("is_solar"),
                "classifier_score": e.get("score"),
            }
            for e in group["entries"]
        ]

        observations.append({
            "figure": _figure_number(figure_label),
            "figure_label": figure_label,
            "caption": caption_text,
            "paragraphs": paragraphs,
            "referenced_tables": referenced_tables,
            "source_images": source_images,
            "phenomenon": llm_meta.get("phenomenon"),
            "active_region": llm_meta.get("active_region"),
            "panels": panels,
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
