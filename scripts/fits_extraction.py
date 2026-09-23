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

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from datetime import datetime, timedelta, timezone
import aiohttp
from parfive import Downloader, SessionConfig

import json
from collections import Counter, defaultdict

import sunpy.map
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import SphericalScreen

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.fits.verify import VerifyWarning
import astropy.units as u

import matplotlib.pyplot as plt
import numpy as np

import argparse
import json
import logging
import os
import re
import warnings

from glob import glob
from pathlib import Path
from typing import Optional

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approximate Heliprojective bounding boxes for limb/disk positions (arcsec).
# Each value is (min, max) for Tx and Ty.

EVENTS = {
  "AR": "Active Region",
  "CE": "CME",
  "CD": "Coronal Dimming",
  "CH": "Coronal Hole",
  "CW": "Coronal Wave",
  "FI": "Filament",
  "FE": "Filament Eruption",
  "FA": "Filament Activation",
  "FL": "Flare",
  "LP": "Loop",
  "OS": "Oscillation",
  "SS": "Sunspot",
  "EF": "Emerging Flux",
  "CJ": "Coronal Jet",
  "PG": "Plage",
  "SG": "Sigmoid",
  "SP": "Spray Surge",
  "CR": "Coronal Rain",
  "CC": "Coronal Cavity",
  "ER": "Eruption",
  "TO": "Topological Object",
  "BU": "UV Burst",
  "EE": "Explosive Event",
  "PB": "Prominence Bubble",
  "PT": "Peacock Tail",
  "EP": "SEPs",
  "IC": "ICMEs",
  "SR": "SIRs",
  "PR": "Prominence"
}

VALID_REGIONS = ["N","S","NE","SE","NW","SW","W","E"]

_SAFE_RE = re.compile(r"[^\w\-]")

# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_all_events(
    metadata_dir: str,
    ) -> list[dict]:
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

        observations = record["observations"]
        # 1. Bucket by time window first -- that's what actually defines "the same event"
        by_time = defaultdict(list)
        for o in observations:
            time_key = (o["timestamp_start"], o["timestamp_end"])
            by_time[time_key].append(dict(o))

        # 2. Fill missing instrument/wavelength using the LOCAL mode (same time window only)
        filled = []
        for time_key, obs_list in by_time.items():
            instrument_counts = Counter(o["instrument"] for o in obs_list if o.get("instrument") is not None)
            wavelength_counts = Counter(o["wavelength_angstrom"] for o in obs_list if o.get("wavelength_angstrom") is not None and isinstance(o.get("wavelength_angstrom"),int))
            local_instrument = instrument_counts.most_common(1)[0][0] if instrument_counts else None
            local_wavelength = wavelength_counts.most_common(1)[0][0] if wavelength_counts else None

            for o in obs_list:
                if o.get("instrument") is None:
                    o["instrument"] = local_instrument
                if o.get("wavelength_angstrom") is None and o.get("instrument").lower() != "hmi":
                    o["wavelength_angstrom"] = local_wavelength
                filled.append(o)

        total_obs = []

# 3. Split out observations with list/tuple wavelengths -- these can't
        #    be grouped (unhashable) and should each stand alone as an event.
        groupable = []
        standalone = []
        for o in filled:
            if isinstance(o.get("wavelength_angstrom"), (list, tuple)):
                standalone.append(o)
            else:
                groupable.append(o)

        # 3b. Group the rest strictly by all four values together
        groups = defaultdict(list)
        for o in groupable:
            key = (o["instrument"], o["wavelength_angstrom"], o["timestamp_start"], o["timestamp_end"])
            groups[key].append(o)

        # 4. Trim down to the fields you want
        for key, obs_list in groups.items():
            first = obs_list[0]
            total_obs.append({
                "timestamp_start": first["timestamp_start"],
                "timestamp_end": first["timestamp_end"],
                "instrument": first["instrument"],
                "wavelength_angstrom": first["wavelength_angstrom"],
                "phenomenon": [o["phenomenon"] for o in obs_list],
                "center_tx_arcsec": [o["center_tx_arcsec"] for o in obs_list],
                "center_ty_arcsec": [o["center_ty_arcsec"] for o in obs_list],
                "observation_filenames": [o["observation_filename"] for o in obs_list],
                "limb_position": [o["limb_position"] for o in obs_list],
                "fov_arcsec": [o["fov_arcsec"] for o in obs_list]
            })

        # 4b. Add each standalone (list/tuple wavelength) observation as its own event
        for o in standalone:
            total_obs.append({
                "timestamp_start": o["timestamp_start"],
                "timestamp_end": o["timestamp_end"],
                "instrument": o["instrument"],
                "wavelength_angstrom": o["wavelength_angstrom"],
                "phenomenon": [o["phenomenon"]],
                "center_tx_arcsec": [o["center_tx_arcsec"]],
                "center_ty_arcsec": [o["center_ty_arcsec"]],
                "observation_filenames": [o["observation_filename"]],
                "limb_position": [o["limb_position"]],
                "fov_arcsec": [o["fov_arcsec"]]
            })

    return total_obs

# ---------------------------------------------------------------------------
# FITS caching
# ---------------------------------------------------------------------------

def _safe(s: str) -> str:
    """Replace characters unsafe for filenames with underscores."""
    return _SAFE_RE.sub("_", s)

def fetch_rate_limited(results, path, downloader, batch_size=10, period=60):
    """
    Fetch Fido search results in batches, capping throughput to
    `batch_size` files per `period` seconds (e.g. 3 files/minute).
    """
    table = results["vso"] if "vso" in results.keys() else results[0]
    files = []

    for i in range(0, len(table), batch_size):
        batch_start = time.monotonic()
        batch = table[i:i + batch_size]

        try:
            batch_files = Fido.fetch(batch, path=path, downloader=downloader)
            files.extend(batch_files)
        except Exception:
            logger.warning("Batch fetch failed for rows %d:%d", i, i + batch_size)

        # only sleep if there's more left to fetch
        if i + batch_size < len(table):
            elapsed = time.monotonic() - batch_start
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    return files

def fetch_fits(
    obs: dict,
    fits_dir: str,
) -> Optional[list]:
    """
    Download the closest FITS file from the VSO for an observation event.

    Skips the download if cache_path already exists.

    Args:
        obs: Observation metadata dict (must contain timestamp_start).
        fits_dir: Directory used for both cache and download destination.
        
    Returns:
        Path to the FITS file on success, None on failure.
    """
    
    ts_str = obs.get("timestamp_start")
    te_str = obs.get("timestamp_end")

    if not ts_str and not te_str:
        logger.warning("No timestamp_start in observation — skipping FITS fetch")
        return None
    
    #try:
    if not ts_str and te_str:
        ts = datetime.fromisoformat(te_str).replace(tzinfo=timezone.utc)
        start = (ts - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")
        end   = (ts + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")

    elif ts_str and te_str:
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        te = datetime.fromisoformat(te_str).replace(tzinfo=timezone.utc)

        start = (ts - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")
        end   = (te + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")

    else:
        start = (ts - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")
        end   = (ts + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S")

    search_attrs: list = [a.Time(start, end)]

    instrument = obs.get("instrument")
    if instrument:
        search_attrs.append(a.Instrument(instrument.lower()))

    wavelength = obs.get("wavelength_angstrom")
    if wavelength:
        if isinstance(wavelength, int):
            search_attrs.append(a.Wavelength(int(wavelength) * u.angstrom))
        elif isinstance(wavelength, (list, tuple)):
            wl_attrs = [a.Wavelength(int(w) * u.angstrom) for w in wavelength]
            search_attrs.append(a.AttrOr(wl_attrs))
            
    results = Fido.search(*search_attrs)
    if results.file_num == 0:
        logger.warning("No VSO results for %s", ts_str)
        return None

    timeout = aiohttp.ClientTimeout(total=0, sock_read=1200, sock_connect=90)
    config = SessionConfig(timeouts=timeout)

    downloader = Downloader(max_conn=5, progress=True, overwrite=False, config=config)

    try:
        obs_files = fetch_rate_limited(results, path=fits_dir, downloader=downloader,
                                    batch_size=3, period=60)
    except Exception:
        logger.warning("Fido.fetch did not download correctly at %s", ts_str)
        return None

    return obs_files

    #except Exception as exc:
    #    logger.warning("FITS fetch failed for %s: %s", ts_str, exc)
    #    return None


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
) -> tuple[sunpy.map.Map,bool]:
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

    tx = obs.get("center_tx_arcsec")
    ty = obs.get("center_ty_arcsec")
    if tx and ty:

        fov = obs.get("fov_arcsec")
        fov_w, fov_h = (float(fov[0]), float(fov[1])) if fov and len(fov) == 2 else (300.0, 300.0)

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
        return (smap.submap(bl, top_right=tr), True)
    else:
        #logger.warning("Strategy A submap failed, using events locations map")
        return (smap, False)

def extract_solar_regions(smap):

    coords = sunpy.map.all_coordinates_from_map(smap)
    Tx = coords.Tx
    Ty = coords.Ty

    on_disk = sunpy.map.coordinate_is_on_solar_disk(coords)

    data = smap.data.astype(float)

    masks = {
        'N':  (Ty.value >= 0) & on_disk,
        'S':  (Ty.value <  0) & on_disk,
        'NE': (Tx.value <= 0) & (Ty.value >= 0) & on_disk,
        'SE': (Tx.value <= 0) & (Ty.value <  0) & on_disk,
        'NW': (Tx.value >  0) & (Ty.value >= 0) & on_disk,
        'SW': (Tx.value >  0) & (Ty.value <  0) & on_disk,
        'W': (Tx.value < 0) & on_disk,
        'E': (Tx.value >= 0) & on_disk,
    }

    results = {}
    
    for name, mask in masks.items():
        region_data = data[mask]
        region_Tx = Tx[mask]
        region_Ty = Ty[mask]
    
        # Geometric center: plain average position of pixels in the region
        center_Tx = np.nanmean(region_Tx)
        center_Ty = np.nanmean(region_Ty)
        geometric_center = SkyCoord(center_Tx, center_Ty, frame=smap.coordinate_frame)

        # Bounding extent of the region
        tx_min, tx_max = np.nanmin(region_Tx), np.nanmax(region_Tx)
        ty_min, ty_max = np.nanmin(region_Ty), np.nanmax(region_Ty)

        bottom_left = SkyCoord(tx_min, ty_min, frame=smap.coordinate_frame)
        top_right = SkyCoord(tx_max, ty_max, frame=smap.coordinate_frame)
            
        # Intensity-weighted centroid (shift weights so they're non-negative)
        weights = region_data - np.nanmin(region_data)
        w_sum = np.nansum(weights)
        if w_sum > 0:
            weighted_Tx = np.nansum(region_Tx * weights) / w_sum
            weighted_Ty = np.nansum(region_Ty * weights) / w_sum
        else:
            weighted_Tx, weighted_Ty = center_Tx, center_Ty
        weighted_center = SkyCoord(weighted_Tx, weighted_Ty, frame=smap.coordinate_frame)

        variance = np.nanvar(region_data)
        std = np.nanstd(region_data)
    
        results[name] = {
            'geometric_center': geometric_center,
            'weighted_center': weighted_center,
            'variance': variance,
            'bottom_left': bottom_left,
            'top_right': top_right,
            'tx_min': tx_min, 'tx_max': tx_max,
            'ty_min': ty_min, 'ty_max': ty_max,
            'std': std,
            'n_pixels': int(mask.sum()),
        }

    return results

def search_with_timeout(query, timeout=30):
    """
    Run Fido.search with a hard timeout (seconds).
    Raises TimeoutError if the search doesn't complete in time.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(Fido.search, *query)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            raise TimeoutError(f"Fido.search did not complete within {timeout}s")

def search_with_retry(query, timeout=30, retries=3, backoff=2):
    """
    Retry Fido.search up to `retries` times if it times out.

    Parameters
    ----------
    query : list
        Attrs to pass to Fido.search.
    timeout : float
        Seconds to wait per attempt.
    retries : int
        Number of attempts before giving up.
    backoff : float
        Multiplier for wait time between attempts (e.g. 2 = doubles each retry).
        Set to 1 for constant wait, or 0 to retry immediately.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return search_with_timeout(query, timeout=timeout)
        except TimeoutError as e:
            last_error = e
            print(f"Attempt {attempt}/{retries} timed out.")
            if attempt < retries:
                wait = backoff ** (attempt - 1) if backoff > 0 else 0
                if wait:
                    print(f"Retrying in {wait:.1f}s...")
                    time.sleep(wait)

    print(f"All {retries} attempts failed.")
    raise last_error
        
def extract_events(
    smap: sunpy.map.Map,
    event_type: str,
    margin: u.Quantity = 15 * u.s,
    timeout=30,
    retries=3
) -> list[SkyCoord]:
    """
    Find HEK events of a given type within a time margin of a map's
    timestamp, and return their coordinates in the map's frame.

    Parameters
    ----------
    smap : sunpy.map.Map
        The map whose timestamp defines the search window.
    event_type : str
        HEK event type code (e.g. 'FL', 'CE', 'AR').
    margin : `~astropy.units.Quantity`, optional
        Time margin before/after the map's timestamp to search. Default 1 minute.

    Returns
    -------
    list[SkyCoord]
        Event coordinates transformed into the map's coordinate frame.
    """
    t_start = smap.date - margin
    t_end = smap.date + margin

    query = [a.Time(t_start, t_end), a.hek.EventType(event_type)]

    try:
        result = search_with_retry(query, timeout=timeout, retries=retries)
    except TimeoutError:
        print(f"HEK search timed out after {timeout}s, skipping this map.")
        return []
    
    if len(result) == 0 or len(result["hek"]) == 0:
        return []

    with SphericalScreen(smap.observer_coordinate, only_off_disk=True):
        event_coords = [
            event["event_coord"].transform_to(smap.coordinate_frame)
            for event in result["hek"]
        ]

    return event_coords

def apply_strategy_b(
    smap: sunpy.map.Map,
    obs: dict,
) -> Optional[tuple[list[sunpy.map.Map],bool]]:
    """
    Extract a submap using the pso.

    Falls back to the full map for "disk" or unknown positions.

    Args:
        smap: Full-disk sunpy Map.
        obs: Observation dict with center_tx_arcsec, center_ty_arcsec, and
                optionally fov_arcsec.

    Returns:
        List of cropped sunpy Maps centered on (or full map on failure / disk).
    """
    
    event = obs.get("phenomenon")
    tot_events = []
    
    for key in EVENTS.keys():
        if event and event.strip().upper().startswith(key) or EVENTS[key].lower() in event.lower():
            if key == "PR" and len(tot_events) == 0:
                possible_events = ["FL","PB","ER","EE"]
                for pevent in possible_events:
                    tot_events += extract_events(smap, pevent)
                break

            else:
                tot_events += extract_events(smap, key)
                break
    
    limb = obs.get("limb_position")
    possible_submaps = []
    if limb and limb.upper() in VALID_REGIONS and len(tot_events) > 0:

        fov_w, fov_h = (300.0, 300.0)

        region_description = extract_solar_regions(smap)[limb.upper()]
        
        region_events = [event for event in tot_events 
                        if event.Tx > region_description["tx_min"] 
                        and event.Tx < region_description["tx_max"]
                        and event.Ty > region_description["ty_min"] 
                        and event.Ty < region_description["ty_max"]]

        for event in region_events:
                        
            bl = SkyCoord(
                        event.Tx - fov_w * u.arcsec,
                        event.Ty - fov_h * u.arcsec,
                        frame=smap.coordinate_frame,
                    )
            tr = SkyCoord(
                        event.Tx + fov_w * u.arcsec,
                        event.Ty + fov_h * u.arcsec,
                        frame=smap.coordinate_frame,
                    )

            possible_submaps.append(smap.submap(bl, top_right=tr))

    elif (limb is None or limb.upper() not in VALID_REGIONS) and len(tot_events) > 0:

        fov_w, fov_h = (300.0, 300.0)
        for event in tot_events:
                                
            bl = SkyCoord(
                        event.Tx - fov_w * u.arcsec,
                        event.Ty - fov_h * u.arcsec,
                        frame=smap.coordinate_frame,
                    )
            tr = SkyCoord(
                        event.Tx + fov_w * u.arcsec,
                        event.Ty + fov_h * u.arcsec,
                        frame=smap.coordinate_frame,
                    )

            possible_submaps.append(smap.submap(bl, top_right=tr))
    else:
        #logger.warning("Strategy B submap failed, using zone cropped map")
        return (smap, False)
    
    return (possible_submaps, True)
    
def apply_strategy_c(
    smap: sunpy.map.Map,
    obs: dict,
) -> tuple[list[sunpy.map.Map], bool]:
    """
    Extract a submap using the pso.

    Falls back to the full map for "disk" or unknown positions.

    Args:
        smap: Full-disk sunpy Map.
        obs: Observation dict with center_tx_arcsec, center_ty_arcsec, and
                optionally fov_arcsec.

    Returns:
        List of cropped sunpy Maps centered on (or full map on failure / disk).
    """

    limb = obs.get("limb_position")
    if limb and limb.upper() in VALID_REGIONS:
        
        region_description = extract_solar_regions(smap)[limb.upper()]
        
        return (smap.submap(region_description["bottom_left"], top_right=region_description["upper_right"]), True)

    else:
        #logger.warning("Strategy C submap failed, using full disk map")
        return (smap, False)
    
# ---------------------------------------------------------------------------
# Refine match
# ---------------------------------------------------------------------------

def confidence(img, template):
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    conf = res.max()
    return np.where(res == conf), conf

def normalize_to_uint8(arr, scale_type='log', low_percentile=1, high_percentile=99.9):
    arr = np.nan_to_num(arr)

    if scale_type == 'log':
        min_val = arr.min()
        if min_val < 0:
            arr = arr - min_val
        arr = np.log1p(arr)
    elif scale_type == 'sqrt':
        arr = np.clip(arr, 0, None)
        arr = np.sqrt(arr)
    elif scale_type == 'linear':
        pass  # no transform, just percentile-clip below

    vmin = np.percentile(arr, low_percentile)
    vmax = np.percentile(arr, high_percentile)
    arr_clipped = np.clip(arr, vmin, vmax)
    norm = (arr_clipped - vmin) / (vmax - vmin + 1e-9)
    return (norm * 255).astype(np.uint8)


def find_image(smap, obs, imgs_dir):
    img_name = obs.get("observation_filename")
    if img_name:
        template = cv2.imread(os.path.join(imgs_dir, img_name), cv2.IMREAD_GRAYSCALE)
        if template is None:
            logger.warning("Could not read template image: %s", img_name)
            return None
        h, w = template.shape
    else:
        logger.warning("No previous image associate with the observation")
        return None

    if "hmi" in str(smap.instrument).lower():
        img = normalize_to_uint8(smap.data, scale_type='linear')
    else:
        img = normalize_to_uint8(smap.data)  # default 'log' for AIA-type intensity data

    ([y], [x]), conf = confidence(img, template)

    return ([y], [x]), (h, w), conf

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

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
    if "hmi" in str(smap.instrument):
        img = smap.data
    else:
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


#if __name__ == "__main__":
#    main()
