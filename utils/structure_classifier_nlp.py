"""
structure_classifier_nlp.py — Classify the solar structure shown in a figure caption.

Uses HuggingFace zero-shot classification (facebook/bart-large-mnli) to map
caption text to one of a controlled vocabulary of solar structures.

The NLP pipeline is loaded lazily on the first call to classify_structure()
and reused across all subsequent calls in the same process (singleton pattern).

Requires:
    pip install transformers>=4.30.0
    torch  (already available in the pytorch_jupyter conda environment)
"""

import re
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------

STRUCTURE_LABELS: List[str] = [
    "Active Region",
    "Flare",
    "Prominence",
    "Coronal Hole",
    "Sunspot",
    "Filament",
    "Plage",
    "Faculae",
    "Granulation",
    "Supergranulation",
    "Polarity Inversion Line",
    "Filament Channel",
    "Coronal Loops",
    "Coronal Cavities",
    "Helmet Streamer",
    "Pseudostreamer",
    "Polar Crown Filament",
    "Sigmoid",
    "Post-Flare Loops",
    "Other",
]

# "Other" is a fallback, not a candidate for the zero-shot classifier
_CANDIDATE_LABELS: List[str] = [lbl for lbl in STRUCTURE_LABELS if lbl != "Other"]

# Minimum score for the top candidate to be accepted; below this → "Other"
SCORE_THRESHOLD: float = 0.10

# Hypothesis template passed to the zero-shot pipeline
_HYPOTHESIS_TEMPLATE = "This solar image shows a {}."


# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------

_PIPELINE = None  # module-level; shared across all calls in a process


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415
        return torch.cuda.is_available()
    except Exception:
        return False


def _get_pipeline():
    """Load and cache the HuggingFace zero-shot classification pipeline."""
    global _PIPELINE
    if _PIPELINE is None:
        logger.info("Loading NLP model 'facebook/bart-large-mnli' (may take a moment)...")
        from transformers import pipeline  # noqa: PLC0415
        device = 0 if _cuda_available() else -1
        _PIPELINE = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
        )
        logger.info("NLP model loaded (device=%s).", "cuda:0" if device == 0 else "cpu")
    return _PIPELINE


# ---------------------------------------------------------------------------
# Caption preprocessing
# ---------------------------------------------------------------------------

def _preprocess_caption(text: str) -> str:
    """
    Strip the figure label prefix and normalize whitespace.

    "Figure 1. AIA 304 Å image..." → "AIA 304 Å image..."
    Truncated to 512 characters to stay within BART's practical token limit.
    """
    text = re.sub(r"^Figure\s+\d+[a-z]?[.:\s]+", "", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    return text[:512]


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------

def classify_structure(caption_text: Optional[str]) -> Tuple[str, float]:
    """
    Classify the solar structure described in a figure caption.

    Args:
        caption_text: Raw caption text (including "Figure N." prefix) or None.

    Returns:
        (structure_label, confidence_score) where:
          - structure_label is one of STRUCTURE_LABELS (exact strings).
          - confidence_score is the NLP model's top probability (0.0–1.0).
            Returns 0.0 when caption is absent or score is below the threshold
            (i.e., when the fallback "Other" label is used).
    """
    if not caption_text or not caption_text.strip():
        return ("Other", 0.0)

    cleaned = _preprocess_caption(caption_text)
    if not cleaned:
        return ("Other", 0.0)

    pipe = _get_pipeline()
    result = pipe(
        cleaned,
        candidate_labels=_CANDIDATE_LABELS,
        hypothesis_template=_HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    # result["labels"] is sorted by score descending
    top_label: str = result["labels"][0]
    top_score: float = result["scores"][0]

    if top_score >= SCORE_THRESHOLD:
        return (top_label, round(top_score, 4))
    else:
        # Score below threshold → fall back to "Other" but still return the raw score
        return ("Other", round(top_score, 4))
