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
    extract_all_captions,
    extract_figure_body_refs,
    match_image_to_caption,
    _LABEL_NUM_RE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert solar physicist assistant. Extract structured metadata for a \
single solar observation from the figure caption and body-text excerpts provided. \
Return ONLY a valid JSON object — no prose, no markdown fences, no extra keys.

Required fields (use null when the information is absent):
  timestamp_start     — ISO UTC string e.g. "2012-07-04T09:54:53" or null
  timestamp_end       — ISO UTC string or null
  instrument          — "AIA", "HMI", "EIT", "LASCO", "XRT", etc., or null
  wavelength_angstrom — integer e.g. 171, 193, 304, or null
  limb_position       — one of "NW","SW","NE","SE","N","S","E","W","disk", or null
  fov_arcsec          — [width_float, height_float] or null
  center_tx_arcsec    — float (Heliprojective Tx) or null
  center_ty_arcsec    — float (Heliprojective Ty) or null
  phenomenon          — concise label for the solar structure/event (e.g. "Prominence", "Active Region")
  confidence          — "high" if Tx/Ty explicitly given, "medium" if limb+fov known, "low" otherwise\
"""

USER_TEMPLATE = """\
Extract the observation metadata for the solar image described below.
Return ONLY a JSON object.

Figure: {figure_label}

Caption:
{caption}

Body text paragraphs referencing this figure:
{paragraphs}\
"""


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
            max_new_tokens=512,
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

    Returns one of "skipped", "success", or "failed".
    """
    out_json = fn.metadata_json(root, name)

    if os.path.exists(out_json):
        return "skipped"

    # Load extraction log (carries the solar images + their page-placement bboxes)
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

    # Captions + body refs still come from the PDF; image bboxes come from the log
    try:
        captions_by_page = extract_all_captions(pdf_path)
        body_refs_by_fig = extract_figure_body_refs(pdf_path)
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", name, exc)
        _write_result(out_json, pdf_filename, paper_title, paper_authors, [], "failed")
        return "failed"

    # Iterate the solar images recorded by extract, in index order
    all_entries: list[dict] = sorted(log.get("images", []), key=lambda e: e["index"])
    observations: list[dict] = []

    for entry in all_entries:
        if not entry.get("is_solar"):
            continue
        obs_filename = entry.get("filename") or f"index_{entry['index']}"

        # Match to nearest caption using the bbox recorded at extract time
        caption: Caption | None = None
        bbox = entry.get("bbox")
        if bbox:
            img_rect = fitz.Rect(*bbox)
            caption, _conf = match_image_to_caption(
                entry["page"], img_rect, captions_by_page
            )
        caption_text = caption.text if caption else ""
        figure_label = caption.figure_label if caption else ""

        # Get body paragraphs for this figure number
        num_match = _LABEL_NUM_RE.search(figure_label) if figure_label else None
        fig_num = num_match.group(1) if num_match else ""
        paragraphs = body_refs_by_fig.get(fig_num, [])

        # Query LLM — skip when there is no text context at all
        llm_meta: dict = {}
        if caption_text or paragraphs:
            try:
                raw = _query_model(tokenizer, model, figure_label, caption_text, paragraphs)
                llm_meta = _parse_llm_output(raw)
            except Exception as exc:
                logger.warning("LLM query failed for %s / %s: %s", name, obs_filename, exc)

        observations.append({
            "observation_filename": obs_filename,
            "figure": _figure_number(figure_label),
            "caption": caption_text,
            "paragraphs": paragraphs,
            "timestamp_start": llm_meta.get("timestamp_start"),
            "timestamp_end": llm_meta.get("timestamp_end"),
            "instrument": llm_meta.get("instrument"),
            "wavelength_angstrom": llm_meta.get("wavelength_angstrom"),
            "limb_position": llm_meta.get("limb_position"),
            "fov_arcsec": llm_meta.get("fov_arcsec"),
            "center_tx_arcsec": llm_meta.get("center_tx_arcsec"),
            "center_ty_arcsec": llm_meta.get("center_ty_arcsec"),
            "phenomenon": llm_meta.get("phenomenon"),
            "confidence": llm_meta.get("confidence") or "low",
        })

    status = "success" if observations else "failed"
    _write_result(out_json, pdf_filename, paper_title, paper_authors, observations, status)
    return status


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    print(f"Found {len(names)} paper(s) to process")

    tokenizer, model = load_model(args.model)

    n_processed = n_skipped = n_failed = 0

    for name in names:
        status = process_paper(name, root, tokenizer, model)

        if status == "skipped":
            n_skipped += 1
            print(f"  [skip]  {name}")
        elif status == "success":
            n_processed += 1
            try:
                with open(fn.metadata_json(root, name), encoding="utf-8") as fh:
                    data = json.load(fh)
                n_obs = len(data.get("observations", []))
            except Exception:
                n_obs = 0
            print(f"  [ok]    {name}  ({n_obs} observation(s))")
        else:
            n_failed += 1
            print(f"  [fail]  {name}")

    print(
        f"\nSummary: {n_processed} processed, {n_skipped} skipped, {n_failed} failed"
        f"  (total: {len(names)})"
    )


if __name__ == "__main__":
    main()
