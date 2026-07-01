#!/usr/bin/env python3
"""
stage2_sdo_query.py — Query SDO/VSO archive and produce cropped submaps from Stage 1 metadata.

For each observation event produced by stage1_metadata_extraction.py, downloads
the closest FITS file from the VSO, crops the map according to available coordinate
metadata, and saves a normalised uint8 PNG alongside a companion JSON.

Strategies (in priority order):
  A — explicit Heliprojective Tx/Ty + FOV (confidence="high")
  B — limb position known, approximate bounding box (confidence="medium")
  C — full-disk map saved for downstream CV matching (confidence="low")

Usage:
  python scripts/stage2_sdo_query.py \
      --metadata_dir papers/metadata/ \
      --fits_dir papers/sdo_fits/ \
      --output_dir papers/matched/
"""

import argparse
import json
import logging
import os
import re
import warnings
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.fits.verify import VerifyWarning

import sunpy.map
from sunpy.net import Fido, attrs as a

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approximate Heliprojective bounding boxes for limb/disk positions (arcsec).
# Each value is (min, max) for Tx and Ty.
LIMB_BOXES: dict[str, Optional[dict[str, tuple[float, float]]]] = {
    "NW":   {"tx": (-800.0, -100.0), "ty": (200.0,   800.0)},
    "SW":   {"tx": (-800.0, -100.0), "ty": (-800.0, -200.0)},
    "NE":   {"tx": (100.0,   800.0), "ty": (200.0,   800.0)},
    "SE":   {"tx": (100.0,   800.0), "ty": (-800.0, -200.0)},
    "N":    {"tx": (-400.0,  400.0), "ty": (400.0,   900.0)},
    "S":    {"tx": (-400.0,  400.0), "ty": (-900.0, -400.0)},
    "E":    {"tx": (400.0,   900.0), "ty": (-400.0,  400.0)},
    "W":    {"tx": (-900.0, -400.0), "ty": (-400.0,  400.0)},
    "disk": None,  # full-disk → Strategy C
}

_SAFE_RE = re.compile(r"[^\w\-]")

# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_all_events(
    metadata_dir: str,
) -> list[tuple[str, int, dict]]:
    """
    Load all observation events from Stage 1 JSON files.

    Args:
        metadata_dir: Directory containing per-paper JSON files.

    Returns:
        List of (paper_stem, event_index, observation_dict) tuples for
        every event in every successful paper record.
    """
    events: list[tuple[str, int, dict]] = []
    for json_path in sorted(glob(os.path.join(metadata_dir, "*.json"))):
        try:
            with open(json_path, encoding="utf-8") as fh:
                record = json.load(fh)
        except Exception as exc:
            logger.warning("Could not read %s: %s", json_path, exc)
            continue

        if record.get("status") != "success":
            continue

        stem = Path(json_path).stem
        for idx, obs in enumerate(record.get("observations", [])):
            events.append((stem, idx, obs))

    return events


# ---------------------------------------------------------------------------
# FITS caching
# ---------------------------------------------------------------------------

def _fits_cache_key(paper_stem: str, event_idx: int, obs: dict) -> str:
    """
    Build a deterministic cache filename for a FITS file.

    Args:
        paper_stem: Stem of the paper metadata JSON filename.
        event_idx: Zero-based index of the event within the paper.
        obs: Observation metadata dict.

    Returns:
        Filename string (not a full path) ending in ".fits".
    """
    instrument = _safe(obs.get("instrument") or "unknown")
    wavelength = str(obs.get("wavelength_angstrom") or "na")
    ts = _safe(obs.get("timestamp_start") or "notime")
    stem = _safe(paper_stem)
    return f"{stem}__{event_idx:03d}__{instrument}__{wavelength}__{ts}.fits"


def _safe(s: str) -> str:
    """Replace characters unsafe for filenames with underscores."""
    return _SAFE_RE.sub("_", s)


def fetch_fits(
    obs: dict,
    fits_dir: str,
    cache_path: str,
) -> Optional[str]:
    """
    Download the closest FITS file from the VSO for an observation event.

    Skips the download if cache_path already exists.

    Args:
        obs: Observation metadata dict (must contain timestamp_start).
        fits_dir: Directory used for both cache and download destination.
        cache_path: Full path to the intended cached FITS file.

    Returns:
        Path to the FITS file on success, None on failure.
    """
    if os.path.exists(cache_path):
        return cache_path

    ts_str = obs.get("timestamp_start")
    if not ts_str:
        logger.warning("No timestamp_start in observation — skipping FITS fetch")
        return None

    try:
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        start = (ts - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")
        end   = (ts + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")

        search_attrs: list = [a.Time(start, end)]

        instrument = obs.get("instrument")
        if instrument:
            search_attrs.append(a.Instrument(instrument))

        wavelength = obs.get("wavelength_angstrom")
        if wavelength:
            search_attrs.append(a.Wavelength(int(wavelength) * u.angstrom))

        results = Fido.search(*search_attrs)
        if results.file_num == 0:
            logger.warning("No VSO results for %s", ts_str)
            return None

        downloaded = Fido.fetch(results[0, 0], path=fits_dir)
        if not downloaded:
            logger.warning("Fido.fetch returned no files for %s", ts_str)
            return None

        src = downloaded[0]
        if src != cache_path:
            os.rename(src, cache_path)

        return cache_path

    except Exception as exc:
        logger.warning("FITS fetch failed for %s: %s", ts_str, exc)
        return None


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

def load_map(fits_path: str) -> Optional[sunpy.map.Map]:
    """
    Load a FITS file as a sunpy Map, suppressing non-standard header warnings.

    Args:
        fits_path: Path to the FITS file.

    Returns:
        Loaded sunpy.map.Map, or None on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=VerifyWarning)
            warnings.simplefilter("ignore")
            return sunpy.map.Map(fits_path)
    except Exception as exc:
        logger.warning("Could not load FITS as sunpy map: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Extraction strategies
# ---------------------------------------------------------------------------

def apply_strategy_a(
    smap: sunpy.map.Map,
    obs: dict,
) -> sunpy.map.Map:
    """
    Extract a submap centred on explicit Heliprojective coordinates.

    Falls back to the full map if the submap operation fails.

    Args:
        smap: Full-disk sunpy Map.
        obs: Observation dict with center_tx_arcsec, center_ty_arcsec, and
             optionally fov_arcsec.

    Returns:
        Cropped sunpy Map (or full map on failure).
    """
    try:
        tx = float(obs["center_tx_arcsec"])
        ty = float(obs["center_ty_arcsec"])
        fov = obs.get("fov_arcsec")
        fov_w, fov_h = (float(fov[0]), float(fov[1])) if fov else (300.0, 300.0)

        bl = SkyCoord(
            (tx - fov_w / 2) * u.arcsec,
            (ty - fov_h / 2) * u.arcsec,
            frame=smap.coordinate_frame,
        )
        tr = SkyCoord(
            (tx + fov_w / 2) * u.arcsec,
            (ty + fov_h / 2) * u.arcsec,
            frame=smap.coordinate_frame,
        )
        return smap.submap(bl, top_right=tr)
    except Exception as exc:
        logger.warning("Strategy A submap failed, using full map: %s", exc)
        return smap


def apply_strategy_b(
    smap: sunpy.map.Map,
    limb: str,
) -> sunpy.map.Map:
    """
    Extract a submap using an approximate bounding box for a limb position.

    Falls back to the full map for "disk" or unknown positions.

    Args:
        smap: Full-disk sunpy Map.
        limb: Limb position key (e.g. "NW", "disk").

    Returns:
        Cropped sunpy Map (or full map on failure / disk).
    """
    box = LIMB_BOXES.get(limb)
    if box is None:
        return smap  # "disk" or unknown → full map
    try:
        tx_min, tx_max = box["tx"]
        ty_min, ty_max = box["ty"]
        bl = SkyCoord(tx_min * u.arcsec, ty_min * u.arcsec, frame=smap.coordinate_frame)
        tr = SkyCoord(tx_max * u.arcsec, ty_max * u.arcsec, frame=smap.coordinate_frame)
        return smap.submap(bl, top_right=tr)
    except Exception as exc:
        logger.warning("Strategy B submap failed for %s, using full map: %s", limb, exc)
        return smap


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """
    Normalise a 2-D array to uint8 in [0, 255] using min-max scaling.

    Applies abs() first to handle signed magnetic data (HMI Stokes V / LOS).

    Args:
        data: Raw 2-D float array from a sunpy Map.

    Returns:
        uint8 numpy array ready for cv2.imwrite.
    """
    return cv2.normalize(np.abs(data), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def save_outputs(
    smap: sunpy.map.Map,
    output_png: str,
    companion: dict,
) -> None:
    """
    Save a normalised PNG and a companion JSON alongside it.

    Args:
        smap: sunpy Map whose data will be normalised and saved.
        output_png: Full path for the output PNG file.
        companion: Dict written as JSON with the same stem as output_png.
    """
    img = normalize_to_uint8(smap.data)
    cv2.imwrite(output_png, img)

    companion_path = str(Path(output_png).with_suffix(".json"))
    with open(companion_path, "w", encoding="utf-8") as fh:
        json.dump(companion, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Per-event processing
# ---------------------------------------------------------------------------

def process_event(
    paper_stem: str,
    event_idx: int,
    obs: dict,
    fits_dir: str,
    output_dir: str,
) -> str:
    """
    Download the FITS file for one observation event and produce a cropped PNG.

    Args:
        paper_stem: Stem of the source paper metadata filename.
        event_idx: Zero-based index of this event within the paper.
        obs: Observation metadata dict.
        fits_dir: Cache directory for downloaded FITS files.
        output_dir: Directory for output PNGs and companion JSONs.

    Returns:
        One of "skipped", "strategy_a", "strategy_b", "strategy_c", or "failed".
    """
    out_name = f"{_safe(paper_stem)}__{event_idx:03d}.png"
    output_png = os.path.join(output_dir, out_name)

    if os.path.exists(output_png):
        return "skipped"

    # --- Download FITS ---
    cache_key = _fits_cache_key(paper_stem, event_idx, obs)
    cache_path = os.path.join(fits_dir, cache_key)
    fits_path = fetch_fits(obs, fits_dir, cache_path)
    if fits_path is None:
        return "failed"

    # --- Load map ---
    smap = load_map(fits_path)
    if smap is None:
        return "failed"

    # --- Choose and apply strategy ---
    confidence = obs.get("confidence", "low")
    strategy = "strategy_c"
    result_map = smap

    if confidence == "high" and obs.get("center_tx_arcsec") is not None:
        result_map = apply_strategy_a(smap, obs)
        strategy = "strategy_a"
    elif confidence == "medium" and obs.get("limb_position"):
        result_map = apply_strategy_b(smap, obs["limb_position"])
        strategy = "strategy_b"

    # --- Build companion metadata ---
    bl = result_map.bottom_left_coord
    tr = result_map.top_right_coord
    companion = {
        "paper": paper_stem,
        "event_index": event_idx,
        "strategy": strategy,
        "observation": obs,
        "bounds_arcsec": {
            "tx_min": float(bl.Tx.arcsec),
            "ty_min": float(bl.Ty.arcsec),
            "tx_max": float(tr.Tx.arcsec),
            "ty_max": float(tr.Ty.arcsec),
        },
        "fits_file": os.path.basename(fits_path),
    }

    try:
        save_outputs(result_map, output_png, companion)
    except Exception as exc:
        logger.warning("Could not save outputs for %s event %d: %s", paper_stem, event_idx, exc)
        return "failed"

    return strategy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: query SDO/VSO archive and produce cropped submaps."
    )
    parser.add_argument(
        "--metadata_dir",
        required=True,
        metavar="DIR",
        help="Directory containing Stage 1 JSON metadata files",
    )
    parser.add_argument(
        "--fits_dir",
        required=True,
        metavar="DIR",
        help="Cache directory for downloaded FITS files",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        metavar="DIR",
        help="Output directory for PNG images and companion JSONs (default: ./output)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.fits_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    events = load_all_events(args.metadata_dir)
    if not events:
        print(f"No events found in {args.metadata_dir}")
        return

    print(f"Loaded {len(events)} observation event(s) from {args.metadata_dir}")

    counts: dict[str, int] = {
        "skipped": 0,
        "strategy_a": 0,
        "strategy_b": 0,
        "strategy_c": 0,
        "failed": 0,
    }

    for paper_stem, event_idx, obs in events:
        label = f"{paper_stem} [{event_idx:03d}]"
        status = process_event(
            paper_stem, event_idx, obs, args.fits_dir, args.output_dir
        )
        counts[status] = counts.get(status, 0) + 1

        ts = obs.get("timestamp_start", "?")
        instr = obs.get("instrument", "?")
        wl = obs.get("wavelength_angstrom", "?")
        print(f"  [{status:12s}]  {label}  {ts}  {instr} {wl}Å")

    total = len(events)
    print(
        f"\nSummary ({total} events):\n"
        f"  Strategy A (high confidence) : {counts['strategy_a']}\n"
        f"  Strategy B (medium confidence): {counts['strategy_b']}\n"
        f"  Strategy C (full disk)        : {counts['strategy_c']}\n"
        f"  Skipped (already done)        : {counts['skipped']}\n"
        f"  Failed                        : {counts['failed']}"
    )


if __name__ == "__main__":
    main()
