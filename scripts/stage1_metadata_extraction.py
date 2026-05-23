#!/usr/bin/env python3
"""
stage1_metadata_extraction.py — Extract structured solar observation metadata from paper PDFs.

For each PDF in an input directory, extracts the full text, sends it to
Qwen2.5-14B-Instruct via HuggingFace transformers (8-bit quantized), and
parses the model's JSON output into per-paper metadata files.

Usage:
  python scripts/stage1_metadata_extraction.py \
      --paper-dir "output/papers/2012-01 - Labrosse, N" \
      --output_dir output/metadata/ \
      --model Qwen/Qwen2.5-14B-Instruct
"""

import argparse
import json
import logging
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Any

import torch

# Redirect HuggingFace cache to the project's models/ directory instead of ~/.cache/huggingface.
# Must be set before importing transformers so the library picks it up at initialisation.
os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parent.parent / "models"),
)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pdf_extractor import extract_full_text
from utils.caption_extractor import build_figure_contexts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert solar physicist assistant. Extract structured metadata from \
solar physics paper text. Return ONLY a valid JSON array — no prose, no markdown \
fences. Each element represents one distinct solar observation event described \
in the paper. If no events are found return an empty array [].

Required fields per event:
  timestamp_start     — ISO UTC string e.g. "2010-06-13T04:15:00" (required)
  timestamp_end       — ISO UTC string or null
  instrument          — "AIA", "HMI", "EIT", "LASCO", etc., or null
  wavelength_angstrom — integer e.g. 171, 193, 304, or null
  limb_position       — one of "NW","SW","NE","SE","N","S","E","W","disk", or null
  fov_arcsec          — [width_float, height_float] or null
  center_tx_arcsec    — float (Heliprojective Tx in arcseconds) or null
  center_ty_arcsec    — float (Heliprojective Ty in arcseconds) or null
  phenomenon          — short string describing the solar event
  confidence          — "high" if explicit Tx/Ty coords given, \
"medium" if limb+fov known, "low" otherwise\
"""

USER_TEMPLATE = """\
The content below is organised as one block per figure in the paper.
Each block shows the figure caption followed by the body-text passages \
that explicitly reference that figure — these passages contain the detailed \
observational context (timestamps, instruments, coordinates, phenomena).
Extract ALL solar observation events across all figures and return ONLY a JSON array.

--- PAPER FIGURES ---
{text}
--- END ---\
"""


# ---------------------------------------------------------------------------
# PDF context extraction
# ---------------------------------------------------------------------------

def _format_figure_contexts(contexts: list, max_chars: int = 12000) -> str:
    """
    Render per-figure context dicts into a prompt-ready string.

    Each block contains the figure label, its caption, and all body paragraphs
    that cite that figure. The result is truncated to max_chars.
    """
    parts = []
    for ctx in contexts:
        block = f"[{ctx['figure_label']}]\nCaption: {ctx['caption']}"
        if ctx["body_refs"]:
            block += "\nBody text references:"
            for ref in ctx["body_refs"]:
                block += f"\n  - {ref}"
        parts.append(block)
    return "\n\n".join(parts)[:max_chars]


def extract_pdf_context(pdf_path: str, max_chars: int = 12000) -> str:
    """
    Extract a prompt-ready context string from a PDF.

    Preferred path: build per-figure blocks (caption + body references) using
    extract_all_captions() and extract_figure_body_refs() so the LLM sees the
    observational details from the main text alongside each figure caption.

    Fallback: raw full-text dump if no figure captions are found.

    Args:
        pdf_path: Path to the PDF file.
        max_chars: Maximum characters to return.

    Returns:
        Formatted string ready to drop into the USER_TEMPLATE.
    """
    try:
        contexts = build_figure_contexts(pdf_path)
    except Exception as exc:
        logger.warning("build_figure_contexts failed (%s); falling back to full text", exc)
        contexts = []

    if contexts:
        return _format_figure_contexts(contexts, max_chars=max_chars)

    logger.info("No figure captions found in %s; using raw full-text extraction", pdf_path)
    return extract_full_text(pdf_path, max_chars=max_chars)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load Qwen2.5-14B-Instruct with 8-bit bitsandbytes quantization.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Tuple of (tokenizer, model) ready for inference.
    """
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

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

def build_messages(text: str) -> list[dict[str, str]]:
    """
    Build the chat messages list for Qwen's chat template.

    Args:
        text: Paper full text (already truncated to max_chars).

    Returns:
        List of {"role": ..., "content": ...} dicts.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]


def query_model(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    messages: list[dict[str, str]],
) -> str:
    """
    Run inference and return the model's raw text response.

    Args:
        tokenizer: Loaded tokenizer.
        model: Loaded model.
        messages: Chat messages list.

    Returns:
        Decoded string with only the newly generated tokens.
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_model_output(raw: str) -> list[dict[str, Any]]:
    """
    Parse the model's raw text into a list of observation dicts.

    Handles markdown code fences. Falls back to regex extraction of the
    first JSON array found in the text.

    Args:
        raw: Raw string returned by the model.

    Returns:
        List of observation dicts.

    Raises:
        ValueError: If no valid JSON array can be extracted.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.strip("`\n ")

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # Model sometimes returns {"observations": [...]}
            for key in ("observations", "events", "data"):
                if isinstance(result.get(key), list):
                    return result[key]
        raise ValueError(f"Unexpected JSON type: {type(result)}")
    except json.JSONDecodeError:
        pass

    # Regex fallback: find the outermost JSON array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract a JSON array from model output")


# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: str,
    output_dir: str,
    failed_dir: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> str:
    """
    Extract observation metadata from a single PDF and write the result JSON.

    Args:
        pdf_path: Absolute path to the PDF file.
        output_dir: Directory for successful output JSON files.
        failed_dir: Directory for raw model outputs when parsing fails.
        tokenizer: Loaded tokenizer.
        model: Loaded model.

    Returns:
        One of "skipped", "success", or "failed".
    """
    filename = os.path.basename(pdf_path)
    stem = Path(pdf_path).stem
    out_json = os.path.join(output_dir, f"{stem}.json")

    if os.path.exists(out_json):
        return "skipped"

    # Extract per-figure context (caption + body references); fall back to raw text
    text = extract_pdf_context(pdf_path, max_chars=12000)
    if not text.strip():
        logger.warning("%s — no extractable text", filename)
        _write_result(out_json, filename, [], "failed")
        return "failed"

    # Run inference
    raw_output = ""
    observations: list[dict[str, Any]] = []
    try:
        messages = build_messages(text)
        raw_output = query_model(tokenizer, model, messages)
        observations = parse_model_output(raw_output)
        _write_result(out_json, filename, observations, "success")
        return "success"
    except Exception as exc:
        logger.warning("%s — extraction failed: %s", filename, exc)
        if raw_output:
            failed_path = os.path.join(failed_dir, f"{stem}_raw.txt")
            try:
                with open(failed_path, "w", encoding="utf-8") as fh:
                    fh.write(raw_output)
            except OSError:
                pass
        _write_result(out_json, filename, [], "failed")
        return "failed"


def _write_result(
    path: str, paper: str, observations: list[dict], status: str
) -> None:
    """Write the per-paper result JSON to disk."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"paper": paper, "observations": observations, "status": status},
            fh,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: extract solar observation metadata from PDFs."
    )
    parser.add_argument(
        "--pdf_dir",
        default=None,
        metavar="DIR",
        help="Directory containing input PDF files",
    )
    parser.add_argument(
        "--paper-dir",
        default=None,
        metavar="DIR",
        help="Alias for --pdf_dir (consistent with the label command interface)",
    )
    parser.add_argument(
        "--output_dir",
        default="output/metadata",
        metavar="DIR",
        help="Directory for output JSON metadata files (default: ./output)",
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

    pdf_dir = args.pdf_dir or args.paper_dir
    if not pdf_dir:
        print("ERROR: one of --pdf_dir or --paper-dir is required", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    failed_dir = os.path.join(output_dir, "failed")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    pdf_files = sorted(glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")

    tokenizer, model = load_model(args.model)

    n_processed = n_skipped = n_failed = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        status = process_pdf(pdf_path, output_dir, failed_dir, tokenizer, model)

        if status == "skipped":
            n_skipped += 1
            print(f"  [skip]  {filename}")
        elif status == "success":
            n_processed += 1
            # Read back to report event count
            stem = Path(pdf_path).stem
            out_json = os.path.join(output_dir, f"{stem}.json")
            try:
                with open(out_json, encoding="utf-8") as fh:
                    data = json.load(fh)
                n_events = len(data.get("observations", []))
            except Exception:
                n_events = 0
            print(f"  [ok]    {filename}  ({n_events} event(s))")
        else:
            n_failed += 1
            print(f"  [fail]  {filename}")

    print(
        f"\nSummary: {n_processed} processed, {n_skipped} skipped, {n_failed} failed"
        f"  (total PDFs: {len(pdf_files)})"
    )


if __name__ == "__main__":
    main()
