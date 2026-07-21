"""
caption_extractor.py — Extract figure captions from PDF files and match them to images.

Uses PyMuPDF (fitz) to:
  - Scan page text blocks and identify figure captions by the "Figure N" / "Fig. N" prefix.
  - Retrieve bounding boxes of all embedded images on each page.
  - Match each image to its nearest caption by vertical proximity.

Requires:
    pip install pymupdf
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False
    logger.warning("PyMuPDF (fitz) not available. Install with: pip install pymupdf")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matches the start of a figure caption: "Figure 1", "Fig. 2a", "Figure 10b."
CAPTION_RE = re.compile(r"^(?:Figure|Fig\.)\s*\d+[a-z]?[.:\s]", re.IGNORECASE)

# Used to extract only the label part, e.g. "Figure 1", "Fig. 2a"
LABEL_RE = re.compile(r"((?:Figure|Fig\.)\s*\d+[a-z]?)", re.IGNORECASE)

# Matches the start of a table caption: "Table 1", "Tab. 2."
TABLE_CAPTION_RE = re.compile(r"^(?:Table|Tab\.)\s*\d+[.:\s]", re.IGNORECASE)

# Extracts the table number from a caption, e.g. "Table 1" -> "1"
TABLE_LABEL_RE = re.compile(r"(?:Table|Tab\.)\s*(\d+)", re.IGNORECASE)

# Matches inline table citations in body text: "Table 1", "Tab. 2", "Tables 1 and 2"
TABLE_BODY_REF_RE = re.compile(r"\b(?:Tables?|Tab\.)\s*(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Caption:
    """A figure caption extracted from a PDF page."""
    figure_label: str                           # e.g. "Figure 1", "Figure 2a"
    text: str                                   # Full merged caption text (whitespace-normalized)
    page: int                                   # 1-based page number
    bbox: Tuple[float, float, float, float]     # (x0, y0, x1, y1) in PDF points; union if multi-block

    def __post_init__(self):
        self.text = " ".join(self.text.split())


@dataclass
class Table:
    """A table (caption + body text) extracted from a PDF page."""
    label: str                                  # e.g. "Table 1"
    number: str                                 # e.g. "1"
    caption: str                                # caption line, whitespace-normalized
    body_text: str                              # caption + header + rows + footnotes (raw text)
    page: int                                   # 1-based page number
    bbox: tuple[float, float, float, float]     # (x0, y0, x1, y1); union of merged blocks

    def __post_init__(self):
        self.caption = " ".join(self.caption.split())
        self.body_text = " ".join(self.body_text.split())


# ---------------------------------------------------------------------------
# Caption extraction
# ---------------------------------------------------------------------------

def _assemble_block_text(block: dict) -> str:
    """Concatenate all span texts in a text block into a single string."""
    parts = []
    for line in block.get("lines", []):
        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
        parts.append(line_text)
    return " ".join(parts).strip()


def _union_bbox(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def extract_all_captions(pdf_path: str) -> Dict[int, List[Caption]]:
    """
    Scan every page of a PDF and extract figure captions.

    A caption is any text block whose assembled text starts with the
    pattern "Figure N[a-z]?[.:]" / "Fig. N[a-z]?[.:]" (case-insensitive).  Consecutive
    non-caption blocks that immediately follow a caption block on the
    same page are merged into it (handles multi-paragraph captions).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict mapping 1-based page number → sorted list of Caption objects
        (sorted top-to-bottom by bbox y0).

    Raises:
        RuntimeError: If PyMuPDF is not installed.
    """
    if not _FITZ_AVAILABLE:
        raise RuntimeError("PyMuPDF is required. Install with: pip install pymupdf")

    captions_by_page: Dict[int, List[Caption]] = {}
    current: Optional[Caption] = None   # caption being accumulated
    current_page: int = -1

    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_number = page_num + 1  # 1-based

            # Finalize any caption carried over from the previous page
            if current is not None and current_page != page_number:
                captions_by_page.setdefault(current_page, []).append(current)
                current = None

            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])

            for block in blocks:
                if block.get("type") != 0:  # 0 = text block; 1 = image block
                    continue

                text = _assemble_block_text(block)
                if not text:
                    continue

                bbox: Tuple[float, float, float, float] = tuple(block["bbox"])  # type: ignore[assignment]

                if CAPTION_RE.match(text):
                    # Finalize the previous caption before starting a new one
                    if current is not None:
                        captions_by_page.setdefault(current_page, []).append(current)

                    label_match = LABEL_RE.match(text)
                    figure_label = label_match.group(1) if label_match else "Figure ?"
                    current = Caption(
                        figure_label=figure_label,
                        text=text,
                        page=page_number,
                        bbox=bbox,
                    )
                    current_page = page_number

                elif current is not None and current_page == page_number:
                    # Continuation block: merge into the current caption
                    current = Caption(
                        figure_label=current.figure_label,
                        text=current.text + " " + text,
                        page=current.page,
                        bbox=_union_bbox(current.bbox, bbox),
                    )

                else:
                    # Non-caption block with no active caption — skip
                    pass

        # Finalize the last caption after iterating all pages
        if current is not None:
            captions_by_page.setdefault(current_page, []).append(current)

    finally:
        doc.close()

    # Sort each page's captions top-to-bottom
    for page_number in captions_by_page:
        captions_by_page[page_number].sort(key=lambda c: c.bbox[1])

    logger.debug(
        "Extracted %d captions across %d pages from %s",
        sum(len(v) for v in captions_by_page.values()),
        len(captions_by_page),
        pdf_path,
    )
    return captions_by_page


# ---------------------------------------------------------------------------
# Image-to-caption matching
# ---------------------------------------------------------------------------
#
# Note: image page-placement bboxes are captured once, at extract time, by
# utils/pdf_extractor.py and stored in extraction_log.json. Downstream stages
# read them from the log and pass them to match_image_to_caption() below — they
# do not re-derive image bboxes from the PDF.

def match_image_to_caption(
    img_page: int,
    img_rect: "fitz.Rect",
    captions_by_page: Dict[int, List[Caption]],
) -> Tuple[Optional[Caption], str]:
    """
    Find the caption closest to an image by vertical distance.

    Search strategy:
      1. Same page — pick the nearest caption by vertical gap.
      2. Adjacent pages (next, then previous) — take the first caption found.
      3. No caption found → return (None, "none").

    The vertical gap between a caption and an image is:
        min(|caption.y0 - image.y1|, |caption.y1 - image.y0|)
    This measures the distance from the closest edge of each object.

    Args:
        img_page: 1-based page number of the image.
        img_rect: Bounding rectangle of the image on its page.
        captions_by_page: Output of extract_all_captions().

    Returns:
        (Caption | None, confidence_string)
        where confidence_string is one of: "same_page", "adjacent_page", "none".
    """
    def _gap(caption: Caption) -> float:
        return min(
            abs(caption.bbox[1] - img_rect.y1),  # caption top vs image bottom
            abs(caption.bbox[3] - img_rect.y0),  # caption bottom vs image top
        )

    # 1. Same-page search
    same_page_captions = captions_by_page.get(img_page, [])
    if same_page_captions:
        best = min(same_page_captions, key=_gap)
        return (best, "same_page")

    # 2. Adjacent-page search — prefer next page, then previous
    for adjacent in (img_page + 1, img_page - 1):
        adjacent_captions = captions_by_page.get(adjacent, [])
        if adjacent_captions:
            return (adjacent_captions[0], "adjacent_page")

    # 3. No caption found
    return (None, "none")


# ---------------------------------------------------------------------------
# Per-figure context builder (used by metadata_extraction)
# ---------------------------------------------------------------------------

# Matches inline figure citations in body text: "Fig. 2", "Figs. 2", "Figure 2a", etc.
_FIG_BODY_REF_RE = re.compile(
    r"\b(?:Figs?\.?\s*|Figures?\s+)(\d+[a-zA-Z]?)",
    re.IGNORECASE,
)

# Extracts the trailing number+letter from a label like "Figure 2a" → "2a"
_LABEL_NUM_RE = re.compile(r"(\d+[a-zA-Z]?)$")


def extract_figure_body_refs(pdf_path: str) -> Dict[str, List[str]]:
    """
    Return a mapping from figure number to body paragraphs that cite it.

    Scans all non-caption text blocks across the document. Any block containing
    a reference such as "Fig. 2", "Figure 2a", or "Figs. 2 and 3" is collected
    under each figure number found in that block.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict mapping figure-number string (e.g. "2", "2a") → list of body
        paragraph texts (whitespace-normalised, deduplicated, in document order).
    """
    if not _FITZ_AVAILABLE:
        raise RuntimeError("PyMuPDF is required. Install with: pip install pymupdf")

    refs: Dict[str, List[str]] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                text = _assemble_block_text(block)
                if not text or CAPTION_RE.match(text):
                    continue
                clean = " ".join(text.split())
                for m in _FIG_BODY_REF_RE.finditer(clean):
                    fig_num = m.group(1)
                    bucket = refs.setdefault(fig_num, [])
                    if clean not in bucket:
                        bucket.append(clean)
    finally:
        doc.close()
    return refs


# ---------------------------------------------------------------------------
# Table extraction (caption + body) and figure -> table linking
# ---------------------------------------------------------------------------
#
# find_tables() reconstructs poorly on vector-drawn tables (returns 0 on many
# A&A PDFs), but get_text still streams the caption, header, rows and footnotes
# as ordinary text blocks in reading order. We capture that raw text so the
# metadata LLM can read values (dates, GOES class, NOAA AR, heliographic
# locations like "S17W08") straight out of the table.

# A block is treated as a caption/next-table/next-figure boundary, or as
# running prose (starts with a lowercase word) — either ends the table body.
_PROSE_START_RE = re.compile(r"^[a-z]")


def extract_all_tables(pdf_path: str, max_body_chars: int = 4000) -> dict[str, "Table"]:
    """
    Extract tables (caption + body text) from a PDF, keyed by table number.

    A table starts at a block matching ``TABLE_CAPTION_RE`` ("Table N", "Tab. N").
    Its body is that caption block plus the following same-page text blocks
    (header row, data rows, footnotes) merged in vertical order, stopping at the
    next caption (table or figure), at a block that begins running prose, or when
    ``max_body_chars`` is reached.

    Args:
        pdf_path: Path to the PDF file.
        max_body_chars: Soft cap on captured body text length per table.

    Returns:
        Dict mapping table-number string (e.g. "1") -> Table. When a number
        appears more than once, the first occurrence wins.
    """
    if not _FITZ_AVAILABLE:
        raise RuntimeError("PyMuPDF is required. Install with: pip install pymupdf")

    tables: dict[str, Table] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page_number = page_num + 1
            blocks = [
                b for b in doc[page_num].get_text("dict").get("blocks", [])
                if b.get("type") == 0 and _assemble_block_text(b)
            ]
            blocks.sort(key=lambda b: b["bbox"][1])  # top-to-bottom

            i = 0
            while i < len(blocks):
                text = _assemble_block_text(blocks[i])
                if not TABLE_CAPTION_RE.match(text):
                    i += 1
                    continue

                num_match = TABLE_LABEL_RE.match(text)
                number = num_match.group(1) if num_match else "?"
                label = f"Table {number}"
                caption_text = text
                bbox = tuple(blocks[i]["bbox"])
                body_parts = [text]

                # Merge following blocks into the table body
                j = i + 1
                while j < len(blocks):
                    nxt = _assemble_block_text(blocks[j])
                    if (
                        TABLE_CAPTION_RE.match(nxt)
                        or CAPTION_RE.match(nxt)
                        or _PROSE_START_RE.match(nxt)
                        or sum(len(p) for p in body_parts) > max_body_chars
                    ):
                        break
                    body_parts.append(nxt)
                    bbox = _union_bbox(bbox, tuple(blocks[j]["bbox"]))
                    j += 1

                if number not in tables:
                    tables[number] = Table(
                        label=label,
                        number=number,
                        caption=caption_text,
                        body_text=" ".join(body_parts),
                        page=page_number,
                        bbox=bbox,  # type: ignore[arg-type]
                    )
                i = j
    finally:
        doc.close()

    logger.debug("Extracted %d table(s) from %s", len(tables), pdf_path)
    return tables


def _ordered_body_paragraphs(pdf_path: str) -> list[dict]:
    """
    Return non-caption body paragraphs in document order.

    Each entry is ``{"page": int, "text": str, "figs": set[str], "tables": set[str]}``
    where ``figs``/``tables`` are the figure/table numbers cited in that paragraph.
    """
    if not _FITZ_AVAILABLE:
        raise RuntimeError("PyMuPDF is required. Install with: pip install pymupdf")

    paragraphs: list[dict] = []
    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            for block in doc[page_num].get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                text = _assemble_block_text(block)
                if not text or CAPTION_RE.match(text) or TABLE_CAPTION_RE.match(text):
                    continue
                clean = " ".join(text.split())
                figs = {m.group(1) for m in _FIG_BODY_REF_RE.finditer(clean)}
                tabs = {m.group(1) for m in TABLE_BODY_REF_RE.finditer(clean)}
                if not figs and not tabs:
                    continue
                paragraphs.append({
                    "page": page_num + 1,
                    "text": clean,
                    "figs": figs,
                    "tables": tabs,
                })
    finally:
        doc.close()
    return paragraphs


def extract_figure_table_links(pdf_path: str, window: int = 4) -> dict[str, list[str]]:
    """
    Map each figure number to the table numbers it is linked to (hybrid strategy).

    Primary (reference-driven): any table cited in a paragraph that also cites
    the figure is linked to that figure.
    Fallback (positional): for a figure with no co-cited table, scan a window of
    ``+/- window`` paragraphs around each paragraph that cites the figure and
    link any table found there.

    Args:
        pdf_path: Path to the PDF file.
        window: Number of neighbouring paragraphs to scan in the fallback.

    Returns:
        Dict mapping figure-number string -> sorted list of table-number strings.
    """
    paragraphs = _ordered_body_paragraphs(pdf_path)

    # Figures that appear anywhere, and primary (co-citation) links
    all_figs: set[str] = set()
    links: dict[str, set[str]] = {}
    for para in paragraphs:
        for fig in para["figs"]:
            all_figs.add(fig)
            if para["tables"]:
                links.setdefault(fig, set()).update(para["tables"])

    # Fallback: positional window for figures with no primary link
    for fig in all_figs:
        if links.get(fig):
            continue
        found: set[str] = set()
        for idx, para in enumerate(paragraphs):
            if fig not in para["figs"]:
                continue
            lo = max(0, idx - window)
            hi = min(len(paragraphs), idx + window + 1)
            for neighbour in paragraphs[lo:hi]:
                found.update(neighbour["tables"])
        if found:
            links[fig] = found

    return {fig: sorted(nums) for fig, nums in links.items()}


def build_figure_contexts(pdf_path: str) -> List[Dict]:
    """
    Build one context dict per figure combining its caption and body references.

    For each figure caption found in the PDF, assembles the body-text paragraphs
    that explicitly cite that figure number ("Fig. N", "Figure N", etc.) so the
    LLM receives both the brief caption label and the richer observational
    description from the main text.

    Each returned dict has keys:
      - ``figure_label``: e.g. "Figure 2"
      - ``caption``: full caption text
      - ``body_refs``: list of body paragraphs that reference this figure

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts sorted by page then vertical position, one per figure.
        Returns an empty list if no figure captions are found.
    """
    captions_by_page = extract_all_captions(pdf_path)
    if not captions_by_page:
        return []

    body_refs = extract_figure_body_refs(pdf_path)

    contexts: List[Dict] = []
    for page_num in sorted(captions_by_page.keys()):
        for caption in captions_by_page[page_num]:
            num_match = _LABEL_NUM_RE.search(caption.figure_label)
            fig_num = num_match.group(1) if num_match else ""
            contexts.append({
                "figure_label": caption.figure_label,
                "caption": caption.text,
                "body_refs": body_refs.get(fig_num, []),
            })

    return contexts
