#!/usr/bin/env python3
"""
stage1_metadata_extraction.py — Extract structured solar observation metadata from paper directories.

For each paper directory produced by the extract stage (output/papers/<folder>/):
  1. Reads extraction_log.json for the list of solar observation images.
  2. Extracts figure captions from the paper PDF and matches each solar image to the
     nearest caption by vertical proximity (same approach as label_plots.py).
  3. Collects body-text paragraphs that explicitly cite each figure.
  4. Sends the combined caption + paragraph text per image to a local LLM
     (Qwen2.5-14B-Instruct, 8-bit quantised) to extract observational metadata.
  5. Writes one JSON file per paper containing all per-image observations.

Usage:
  python scripts/stage1_metadata_extraction.py \\
      --paper_dir "output/papers/2012-01 - Labrosse, N" \\
      --output_dir output/metadata/

  python scripts/stage1_metadata_extraction.py \\
      --pdf_dir output/papers/ \\
      --output_dir output/metadata/ \\
      --model Qwen/Qwen2.5-14B-Instruct
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch

# Redirect HuggingFace cache to the project's models/ directory.
# Must be set before importing transformers.
os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parent.parent / "models"),
)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.caption_extractor import (
    Caption,
    extract_all_captions,
    extract_figure_body_refs,
    get_all_image_bboxes,
    match_image_to_caption,
    _LABEL_NUM_RE,
)
from utils.structure_classifier_nlp import classify_structure

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

def _load_extraction_log(paper_dir: str) -> dict:
    log_path = os.path.join(paper_dir, "extraction_log.json")
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"extraction_log.json not found in '{paper_dir}'")
    with open(log_path, encoding="utf-8") as fh:
        return json.load(fh)


def _find_pdf(paper_dir: str) -> str:
    matches = glob.glob(os.path.join(paper_dir, "paper_*.pdf"))
    if not matches:
        raise FileNotFoundError(f"No paper_*.pdf found in '{paper_dir}'")
    return matches[0]


def _build_global_bbox_list(bboxes_by_page: dict) -> list:
    """Flatten bboxes_by_page into a list ordered by page (mirrors extract_plots index order)."""
    flat = []
    for page_num in sorted(bboxes_by_page.keys()):
        flat.extend(bboxes_by_page[page_num])
    return flat


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

def process_paper_dir(
    paper_dir: str,
    output_dir: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> str:
    """
    Process one paper directory and write the metadata JSON.

    Returns one of "skipped", "success", or "failed".
    """
    folder_name = os.path.basename(os.path.normpath(paper_dir))
    out_json = os.path.join(output_dir, f"{folder_name}.json")

    if os.path.exists(out_json):
        return "skipped"

    # Load extraction log
    try:
        log = _load_extraction_log(paper_dir)
    except FileNotFoundError as exc:
        logger.warning("%s", exc)
        return "failed"

    paper_title = log.get("title", "")
    first_author = log.get("first_author", "")

    # Find PDF
    try:
        pdf_path = _find_pdf(paper_dir)
    except FileNotFoundError as exc:
        logger.warning("%s", exc)
        _write_result(out_json, folder_name + ".pdf", paper_title, first_author, [], "failed")
        return "failed"

    pdf_filename = os.path.basename(pdf_path)
    paper_authors = _pdf_authors(pdf_path) or first_author

    # Extract captions, image bboxes, and body refs from the PDF
    try:
        captions_by_page = extract_all_captions(pdf_path)
        bboxes_by_page = get_all_image_bboxes(pdf_path)
        global_bboxes = _build_global_bbox_list(bboxes_by_page)
        body_refs_by_fig = extract_figure_body_refs(pdf_path)
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", folder_name, exc)
        _write_result(out_json, pdf_filename, paper_title, paper_authors, [], "failed")
        return "failed"

    # Iterate solar images in global index order (same as label_plots.py)
    all_entries: list[dict] = sorted(log.get("images", []), key=lambda e: e["index"])
    observations: list[dict] = []
    solar_rank = 0

    for entry in all_entries:
        if not entry.get("is_solar"):
            continue
        solar_rank += 1
        global_idx = entry["index"]
        obs_filename = f"solar_{solar_rank:03d}_p{entry['page']}_{entry['image_type']}"

        # Get image bounding box
        img_rect = None
        if global_idx < len(global_bboxes):
            _xref, img_rect = global_bboxes[global_idx]

        # Match to nearest caption
        caption: Caption | None = None
        if img_rect is not None:
            caption, _conf = match_image_to_caption(
                entry["page"], img_rect, captions_by_page
            )
        caption_text = caption.text if caption else ""
        figure_label = caption.figure_label if caption else ""

        # Get body paragraphs for this figure number
        num_match = _LABEL_NUM_RE.search(figure_label) if figure_label else None
        fig_num = num_match.group(1) if num_match else ""
        paragraphs = body_refs_by_fig.get(fig_num, [])

        # NLP classification as fallback for phenomenon/confidence
        nlp_label, nlp_confidence = classify_structure(caption_text or None)

        # Query LLM — skip when there is no text context at all
        llm_meta: dict = {}
        if caption_text or paragraphs:
            try:
                raw = _query_model(tokenizer, model, figure_label, caption_text, paragraphs)
                llm_meta = _parse_llm_output(raw)
            except Exception as exc:
                logger.warning("LLM query failed for %s / %s: %s", folder_name, obs_filename, exc)

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
            "phenomenon": llm_meta.get("phenomenon") or nlp_label,
            "confidence": llm_meta.get("confidence") or nlp_confidence,
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
        description="Stage 1: extract solar observation metadata from paper directories."
    )
    parser.add_argument(
        "--paper_dir",
        default=None,
        metavar="DIR",
        help="Single paper directory produced by the extract stage "
             "(e.g. output/papers/2012-01 - Labrosse, N)",
    )
    parser.add_argument(
        "--paper-dir",
        dest="paper_dir",
        default=None,
        metavar="DIR",
        help="Alias for --paper_dir",
    )
    parser.add_argument(
        "--pdf_dir",
        default=None,
        metavar="DIR",
        help="Parent directory; all subdirectories that contain "
             "extraction_log.json are processed (e.g. output/papers/)",
    )
    parser.add_argument(
        "--output_dir",
        default="output/metadata",
        metavar="DIR",
        help="Output directory for JSON metadata files (default: output/metadata)",
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

    paper_dir = args.paper_dir
    pdf_dir = args.pdf_dir

    if not paper_dir and not pdf_dir:
        print("ERROR: one of --paper_dir or --pdf_dir is required", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Collect paper directories to process
    paper_dirs: list[str] = []
    if paper_dir:
        paper_dirs = [os.path.normpath(paper_dir)]
    else:
        for entry in sorted(os.scandir(pdf_dir), key=lambda e: e.name):
            if entry.is_dir() and os.path.isfile(
                os.path.join(entry.path, "extraction_log.json")
            ):
                paper_dirs.append(entry.path)

    if not paper_dirs:
        print(f"No paper directories found in {pdf_dir or paper_dir}")
        return

    print(f"Found {len(paper_dirs)} paper director(ies) to process")

    tokenizer, model = load_model(args.model)

    n_processed = n_skipped = n_failed = 0

    for pdir in paper_dirs:
        folder_name = os.path.basename(pdir)
        status = process_paper_dir(pdir, output_dir, tokenizer, model)

        if status == "skipped":
            n_skipped += 1
            print(f"  [skip]  {folder_name}")
        elif status == "success":
            n_processed += 1
            out_json = os.path.join(output_dir, f"{folder_name}.json")
            try:
                with open(out_json, encoding="utf-8") as fh:
                    data = json.load(fh)
                n_obs = len(data.get("observations", []))
            except Exception:
                n_obs = 0
            print(f"  [ok]    {folder_name}  ({n_obs} observation(s))")
        else:
            n_failed += 1
            print(f"  [fail]  {folder_name}")

    print(
        f"\nSummary: {n_processed} processed, {n_skipped} skipped, {n_failed} failed"
        f"  (total: {len(paper_dirs)})"
    )


if __name__ == "__main__":
    main()
