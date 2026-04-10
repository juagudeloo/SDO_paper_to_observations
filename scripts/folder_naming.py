"""
folder_naming.py — Build output folder names from paper metadata.

Handles:
- Publication date formatting (DB uses YYYY-MM-00 with day always 00)
- First-author name extraction from PDF text (DB authors field is empty)
- Filesystem-safe folder name construction
"""

import re
import logging

logger = logging.getLogger(__name__)

# Characters forbidden in folder names on common filesystems
_FORBIDDEN_CHARS_RE = re.compile(r'[/\\:*?"<>|]')


# ---------------------------------------------------------------------------
# Date formatting
# ---------------------------------------------------------------------------

def format_publication_date(pub_date: str) -> str:
    """
    Convert DB publication date to a human-readable string.

    The DB stores dates as 'YYYY-MM-00' (day is always 00).
    Year-only entries appear as 'YYYY-00-00'.

    Examples:
        '2018-06-00' -> '2018-06'
        '2010-00-00' -> '2010'
        '2015-04-23' -> '2015-04-23'
        '2012-07-00' -> '2012-07'
    """
    if len(pub_date) < 4:
        return pub_date

    year = pub_date[:4]

    if len(pub_date) < 7:
        return year

    month = pub_date[5:7]
    if month == "00":
        return year

    if len(pub_date) < 10:
        return f"{year}-{month}"

    day = pub_date[8:10]
    if day == "00":
        return f"{year}-{month}"

    return f"{year}-{month}-{day}"


# ---------------------------------------------------------------------------
# Author parsing
# ---------------------------------------------------------------------------

def parse_first_author(
    authors_field: str,
    bibcode: str,
    pdf_text_page1: str,
) -> tuple:
    """
    Extract (last_name, first_initial) for the first author.

    Strategy (in order):
      1. Parse authors_field if non-empty.
         Supports:
           - "Song, Y. L.; Tian, H." -> ('Song', 'Y')
           - "Y. L. Song, H. Tian"   -> ('Song', 'Y')
      2. Regex on pdf_text_page1 for common author formats.
      3. Fallback: last uppercase letter of bibcode -> ('Unknown', 'X')

    Returns:
        Tuple of (last_name, first_initial) strings.
    """
    # Strategy 1: parse authors_field
    if authors_field and authors_field.strip():
        result = _parse_authors_field(authors_field.strip())
        if result:
            logger.debug("Author from authors_field: %s", result)
            return result

    # Strategy 2: parse PDF page 1 text
    if pdf_text_page1 and pdf_text_page1.strip():
        result = _parse_pdf_text(pdf_text_page1)
        if result:
            logger.debug("Author from PDF text: %s", result)
            return result

    # Strategy 3: bibcode fallback
    result = _parse_bibcode(bibcode)
    logger.debug("Author from bibcode fallback: %s", result)
    return result


def _parse_authors_field(authors: str) -> tuple:
    """Parse 'LastName, F. I.; ...' or 'F. I. LastName, ...' format."""
    # Try "LastName, F." format (semicolon-separated)
    # e.g. "Song, Y. L.; Tian, H."
    match = re.match(r"([A-Z][a-zA-Z'\-]+),\s+([A-Z])", authors)
    if match:
        return match.group(1), match.group(2)

    # Try "F. I. LastName" format (comma or semicolon separated)
    # e.g. "Y. L. Song, H. Tian" or "Y.L. Song"
    match = re.match(r"([A-Z])[\.\s]+(?:[A-Z][\.\s]+)*([A-Z][a-z]{2,})", authors)
    if match:
        return match.group(2), match.group(1)

    return None


def _parse_pdf_text(text: str) -> tuple:
    """
    Extract first author name from PDF page 1 text.

    Tries several common patterns used in astronomy journals.
    """
    lines = text[:3000]  # limit to first portion of page text

    # Pattern 1: "F. I. LastName1,2" (e.g., "Y. L. Song1,2")
    match = re.search(
        r"\b([A-Z])\.(?:\s*[A-Z]\.)?\s+([A-Z][a-z]{2,})\s*[\d,\s]",
        lines,
    )
    if match:
        return match.group(2), match.group(1)

    # Pattern 2: "LastName, F. I." (e.g., "Song, Y. L.")
    match = re.search(r"\b([A-Z][a-z]{2,}),\s+([A-Z])\.", lines)
    if match:
        return match.group(1), match.group(2)

    # Pattern 3: "FirstName LastName" (e.g., "John Smith")
    match = re.search(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]{2,})\b", lines)
    if match:
        return match.group(2), match.group(1)[0]

    return None


def _parse_bibcode(bibcode: str) -> tuple:
    """
    Fallback: extract author initial from the last character of the bibcode.

    ADS bibcodes end with the first letter of the first author's last name,
    e.g. '2018A&A...613A..69S' -> 'S'.
    """
    if bibcode:
        last_char = bibcode.strip()[-1]
        if last_char.isupper():
            return "Unknown", last_char
    return "Unknown", "X"


# ---------------------------------------------------------------------------
# Folder name construction
# ---------------------------------------------------------------------------

def build_folder_name(
    pub_date: str, last_name: str, first_initial: str
) -> str:
    """
    Build a filesystem-safe folder name from date and author.

    Format: 'YYYY-MM - LastName, F'
    Examples:
        ('2018-06-00', 'Song', 'Y')     -> '2018-06 - Song, Y'
        ('2015-04-23', 'Doe', 'J')      -> '2015-04-23 - Doe, J'
        ('2010-00-00', 'Unknown', 'S')  -> '2010 - Unknown, S'

    Returns:
        Sanitized folder name string.
    """
    date_str = format_publication_date(pub_date)
    author_str = f"{last_name}, {first_initial}"
    folder = f"{date_str} - {author_str}"
    # Sanitize forbidden characters
    folder = _FORBIDDEN_CHARS_RE.sub("_", folder)
    # Collapse multiple spaces
    folder = re.sub(r" {2,}", " ", folder).strip()
    return folder
