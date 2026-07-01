"""
pdf_extractor.py — Extract images and text from PDF files using PyMuPDF.

Requires:
    pip install pymupdf

PyMuPDF handles embedded raster images, Form XObjects, and indirect image
streams that poppler's pdfimages silently skips.
"""

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False
    logger.warning("PyMuPDF (fitz) not available. Install with: pip install pymupdf")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ImageMetadata:
    """Metadata for a single image embedded in a PDF."""
    index: int              # Sequential number across the whole document
    page: int               # PDF page number (1-based)
    width: int              # Image width in pixels
    height: int             # Image height in pixels
    color_space: str        # 'rgb', 'cmyk', 'gray', etc.
    encoding: str           # 'jpeg', 'png', 'jp2', 'raw', etc.
    file_path: str | None = field(default=None)  # Set after extraction
    # Placement rectangle on the page in PDF points (x0, y0, x1, y1), origin
    # top-left. Used downstream to match the image to its figure caption by
    # vertical proximity without re-parsing the PDF. None if the image is
    # embedded but not placed on any page.
    bbox: tuple[float, float, float, float] | None = field(default=None)


# ---------------------------------------------------------------------------
# Color space helpers
# ---------------------------------------------------------------------------

_CS_MAP = {
    fitz.CS_RGB:  "rgb",
    fitz.CS_GRAY: "gray",
    fitz.CS_CMYK: "cmyk",
} if _FITZ_AVAILABLE else {}


def _colorspace_name(cs) -> str:
    if cs is None:
        return "unknown"
    return _CS_MAP.get(cs.n, cs.name.lower() if hasattr(cs, "name") else "unknown")


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def _largest_image_bbox(page, xref: int) -> tuple[float, float, float, float] | None:
    """
    Return the placement rectangle of image `xref` on `page` as (x0, y0, x1, y1).

    An xref can be placed multiple times on a page; the largest-area rect is the
    one worth matching to a caption. Returns None if the image is not placed.
    """
    try:
        rects = page.get_image_rects(xref)
    except Exception as exc:
        logger.debug("Could not get image rects for xref %d: %s", xref, exc)
        return None
    if not rects:
        return None
    biggest = max(rects, key=lambda r: r.get_area())
    return (float(biggest.x0), float(biggest.y0), float(biggest.x1), float(biggest.y1))


def extract_pdf_images(pdf_path: str, output_dir: str) -> list[ImageMetadata]:
    """
    Extract all embedded images from a PDF as PNG files using PyMuPDF.

    Iterates every page and collects images via page.get_images(full=True),
    which includes Form XObjects and indirectly referenced images that
    pdfimages misses. Each image is saved as img-NNN.png in output_dir.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory where extracted PNG files will be saved.

    Returns:
        List of ImageMetadata with file_path filled in for each extracted image.

    Raises:
        RuntimeError: If PyMuPDF is not installed.
        FileNotFoundError: If pdf_path does not exist.
    """
    if not _FITZ_AVAILABLE:
        raise RuntimeError(
            "PyMuPDF is required for image extraction.\n"
            "Install it with: pip install pymupdf"
        )
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    metadata_list: list[ImageMetadata] = []
    index = 0

    # Track xrefs already processed to avoid duplicates from shared XObjects
    seen_xrefs: set = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
            except Exception as exc:
                logger.debug("Could not extract xref %d on page %d: %s", xref, page_num + 1, exc)
                index += 1
                continue

            width = base_image["width"]
            height = base_image["height"]
            colorspace = base_image.get("colorspace", 0)
            ext = base_image.get("ext", "png")
            image_bytes = base_image["image"]

            # Map colorspace integer to name
            cs_map = {1: "gray", 3: "rgb", 4: "cmyk"}
            cs_name = cs_map.get(colorspace, f"cs{colorspace}")

            # Save as PNG via PIL to normalise all formats to a single type
            dest_path = os.path.join(output_dir, f"img-{index:03d}.png")
            try:
                _save_as_png(image_bytes, ext, dest_path)
                file_path = dest_path
            except Exception as exc:
                logger.debug("Could not save image %d as PNG: %s", index, exc)
                file_path = None

            metadata_list.append(ImageMetadata(
                index=index,
                page=page_num + 1,
                width=width,
                height=height,
                color_space=cs_name,
                encoding=ext,
                file_path=file_path,
                bbox=_largest_image_bbox(page, xref),
            ))
            index += 1

    doc.close()
    logger.debug("PyMuPDF: extracted %d images from %s", len(metadata_list), pdf_path)
    return metadata_list


def _save_as_png(image_bytes: bytes, ext: str, dest_path: str) -> None:
    """Save raw image bytes as a PNG file, converting via PIL if needed."""
    import io
    from PIL import Image

    if ext.lower() == "png":
        with open(dest_path, "wb") as fh:
            fh.write(image_bytes)
        return

    img = Image.open(io.BytesIO(image_bytes))
    # Convert CMYK to RGB so cv2 can read it
    if img.mode == "CMYK":
        img = img.convert("RGB")
    img.save(dest_path, "PNG")


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_full_text(pdf_path: str, max_chars: int = 8000) -> str:
    """
    Extract and concatenate text from all pages of a PDF, stopping once
    max_chars is reached.

    Args:
        pdf_path: Path to the PDF file.
        max_chars: Maximum characters to return (default 8000).

    Returns:
        Concatenated page text, truncated to max_chars, or empty string on failure.
    """
    if not _FITZ_AVAILABLE:
        return ""
    if not os.path.isfile(pdf_path):
        return ""
    try:
        doc = fitz.open(pdf_path)
        chunks: list = []
        total = 0
        for page in doc:
            t = page.get_text()
            chunks.append(t)
            total += len(t)
            if total >= max_chars:
                break
        doc.close()
        return "".join(chunks)[:max_chars]
    except Exception as exc:
        logger.warning("PyMuPDF full-text extraction error: %s", exc)
        return ""


def extract_first_page_text(pdf_path: str) -> str:
    """
    Extract text from the first page of a PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Text content of the first page, or empty string on failure.
    """
    if not _FITZ_AVAILABLE:
        logger.warning("PyMuPDF not available; author detection will be limited")
        return ""

    if not os.path.isfile(pdf_path):
        return ""

    try:
        doc = fitz.open(pdf_path)
        text = doc[0].get_text() if len(doc) > 0 else ""
        doc.close()
        return text
    except Exception as exc:
        logger.warning("PyMuPDF text extraction error: %s", exc)
        return ""
