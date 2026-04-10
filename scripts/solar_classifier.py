"""
solar_classifier.py — Classify images as solar observations or other figures.

Uses classical computer vision:
  1. Metadata pre-filter (size, color space)
  2. Background color analysis (dark space vs. white plot background)
  3. Hough Circle Transform (detect solar disk)
  4. HSV color analysis (AIA false-color palettes)
  5. HMI grayscale detection
  6. Texture analysis (for cropped active regions)
  7. Scientific plot penalty (edge density + white interior)

Decision threshold: raw score >= 5 -> solar observation
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy import cv2 so unit tests can mock it if needed
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("cv2 not available; solar classification will be limited")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Result of classifying an image as a solar observation."""
    is_solar: bool
    score: float                     # 0.0 to 1.0 normalized confidence
    signals: List[str] = field(default_factory=list)
    image_type: str = "unknown"      # 'aia_false_color', 'hmi_grayscale', 'unknown', 'rejected'


# ---------------------------------------------------------------------------
# Thresholds (tune here if needed)
# ---------------------------------------------------------------------------

SOLAR_SCORE_THRESHOLD = 5     # Minimum raw integer score to be classified as solar
SCORE_NORMALIZATION = 20.0    # Divide raw score by this for 0-1 range

MIN_WIDTH = 200
MIN_HEIGHT = 200

# Hough circle params
HOUGH_CIRCLE_MIN_FRACTION = 0.30   # Min radius as fraction of shorter dimension
HOUGH_CIRCLE_MAX_FRACTION = 0.55   # Max radius as fraction of shorter dimension
HOUGH_DP = 1.5
HOUGH_PARAM1 = 100  # Canny high threshold
HOUGH_PARAM2 = 40   # Accumulator threshold

# AIA false-color HSV ranges (OpenCV hue: 0-179)
AIA_ORANGE_HUE = (5, 30)    # 193, 94, 1600 Å -> orange/gold
AIA_BLUE_HUE = (42, 65)     # 171 Å -> blue-green
AIA_RED_HUE_LOW = (0, 5)    # 304 Å -> red
AIA_RED_HUE_HIGH = (170, 179)
AIA_MIN_SATURATION = 80     # /255

# Scoring weights
SCORE_FULL_DISK_CIRCLE = 8
SCORE_OFF_CENTER_CIRCLE = 3
SCORE_AIA_STRONG = 8
SCORE_AIA_WEAK = 4
SCORE_HMI_FULL_DISK = 4
SCORE_HMI_CANDIDATE = 1
SCORE_DARK_BACKGROUND = 2
SCORE_MODERATE_DARK = 1
SCORE_WHITE_BACKGROUND = -4
SCORE_DARK_REGION_STRONG = 2
SCORE_DARK_REGION_MODERATE = 1
SCORE_HIGH_TEXTURE = 2
SCORE_PLOT_STRONG = -5
SCORE_PLOT_MODERATE = -2


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_image(
    img_meta,
    file_path: Optional[str] = None,
) -> ClassificationResult:
    """
    Classify an image as a solar observation or other figure.

    Args:
        img_meta: ImageMetadata object (from pdf_extractor).
        file_path: Override file_path from img_meta (useful in tests).

    Returns:
        ClassificationResult with is_solar, score, signals, image_type.
    """
    path = file_path or (img_meta.file_path if img_meta else None)

    # --- Step 1: Metadata pre-filter ---
    result = _metadata_prefilter(img_meta)
    if result is not None:
        return result

    # --- Load image ---
    if not _CV2_AVAILABLE:
        return ClassificationResult(
            is_solar=False, score=0.0,
            signals=["cv2_unavailable"], image_type="rejected"
        )

    if not path:
        return ClassificationResult(
            is_solar=False, score=0.0,
            signals=["no_file_path"], image_type="rejected"
        )

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        logger.debug("Could not read image: %s", path)
        return ClassificationResult(
            is_solar=False, score=0.0,
            signals=["unreadable"], image_type="rejected"
        )

    return _classify_pixels(img_bgr, path)


def classify_image_array(img_bgr: np.ndarray) -> ClassificationResult:
    """
    Classify a BGR image array directly (used in tests with synthetic images).

    Args:
        img_bgr: NumPy array in BGR color format (H, W, 3).

    Returns:
        ClassificationResult.
    """
    return _classify_pixels(img_bgr, path="<array>")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _metadata_prefilter(img_meta) -> Optional[ClassificationResult]:
    """
    Fast rejection based on image metadata (no pixel reading).

    Returns ClassificationResult if rejected, None if should proceed.
    """
    if img_meta is None:
        return None

    w, h = img_meta.width, img_meta.height
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return ClassificationResult(
            is_solar=False, score=0.0,
            signals=["too_small"], image_type="rejected"
        )

    if img_meta.color_space == "index":
        return ClassificationResult(
            is_solar=False, score=0.0,
            signals=["palette_indexed"], image_type="rejected"
        )

    return None


def _classify_pixels(img_bgr: np.ndarray, path: str) -> ClassificationResult:
    """Run pixel-level classification on a BGR image array."""
    signals = []
    raw_score = 0
    image_type = "unknown"
    is_full_disk = False

    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32)
    v_ch = hsv[:, :, 2].astype(np.float32)

    # --- Step 2: Background analysis ---
    border_margin = max(5, int(min(H, W) * 0.05))
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:border_margin, :] = True
    border_mask[-border_margin:, :] = True
    border_mask[:, :border_margin] = True
    border_mask[:, -border_margin:] = True

    border_v = v_ch[border_mask]
    border_white_ratio = (border_v > 240).mean()
    border_dark_ratio = (border_v < 25).mean()

    if border_white_ratio > 0.5:
        raw_score += SCORE_WHITE_BACKGROUND
        signals.append("white_background")
    elif border_dark_ratio > 0.2:
        raw_score += SCORE_DARK_BACKGROUND
        signals.append("dark_background")

    # --- Step 3: Global dark pixel ratio ---
    dark_pixel_ratio = (v_ch < 20).mean()
    if dark_pixel_ratio > 0.15:
        raw_score += SCORE_DARK_REGION_STRONG
        signals.append(f"dark_region_{dark_pixel_ratio:.0%}")
    elif dark_pixel_ratio > 0.05:
        raw_score += SCORE_DARK_REGION_MODERATE
        signals.append("moderate_dark_region")

    # --- Step 4: Hough Circle Transform ---
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    short_dim = min(H, W)
    min_r = int(short_dim * HOUGH_CIRCLE_MIN_FRACTION)
    max_r = int(short_dim * HOUGH_CIRCLE_MAX_FRACTION)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=short_dim * 0.4,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        cx, cy, r = circles[0][0]
        center_x_ok = (W * 0.25) < cx < (W * 0.75)
        center_y_ok = (H * 0.25) < cy < (H * 0.75)
        if center_x_ok and center_y_ok:
            raw_score += SCORE_FULL_DISK_CIRCLE
            signals.append(f"full_disk_circle_r{int(r)}")
            is_full_disk = True
        else:
            raw_score += SCORE_OFF_CENTER_CIRCLE
            signals.append("off_center_circle")
    else:
        is_full_disk = False

    # --- Step 5: AIA false-color detection ---
    sat_mask = s_ch > AIA_MIN_SATURATION

    orange_mask = (
        (h_ch >= AIA_ORANGE_HUE[0]) & (h_ch <= AIA_ORANGE_HUE[1]) & sat_mask
    )
    blue_mask = (
        (h_ch >= AIA_BLUE_HUE[0]) & (h_ch <= AIA_BLUE_HUE[1]) & sat_mask
    )
    red_mask = (
        ((h_ch <= AIA_RED_HUE_LOW[1]) | (h_ch >= AIA_RED_HUE_HIGH[0])) & sat_mask
    )

    aia_ratio = orange_mask.mean() + blue_mask.mean() + red_mask.mean()

    if aia_ratio > 0.15:
        raw_score += SCORE_AIA_STRONG
        signals.append(f"aia_false_color_{aia_ratio:.0%}")
        image_type = "aia_false_color"
    elif aia_ratio > 0.05:
        raw_score += SCORE_AIA_WEAK
        signals.append(f"aia_weak_color_{aia_ratio:.0%}")
        image_type = "aia_false_color"

    # --- Step 6: HMI grayscale detection ---
    # Near-grayscale images are HMI (continuum intensity, magnetic field, or dopplergram).
    # No prior score is required: a clean HMI cropped region may have no other signals.
    mean_saturation = s_ch.mean()
    if mean_saturation < 15:
        if is_full_disk:
            raw_score += SCORE_HMI_FULL_DISK
            signals.append(f"hmi_grayscale_sat{mean_saturation:.1f}")
            if image_type == "unknown":
                image_type = "hmi_grayscale"
        else:
            # Cropped HMI region: give a base positive score unconditionally.
            raw_score += 3
            signals.append(f"hmi_region_sat{mean_saturation:.1f}")
            if image_type == "unknown":
                image_type = "hmi_grayscale"

    # --- Step 7: Texture analysis (cropped active regions) ---
    # Solar surfaces have characteristic granulation / sunspot texture.
    # No prior score required: high texture alone is a weak positive signal.
    gray_std = float(gray.std())
    if gray_std > 40 and not is_full_disk:
        raw_score += SCORE_HIGH_TEXTURE
        signals.append(f"high_texture_std{gray_std:.0f}")

    # --- Step 8: Scientific plot penalty ---
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    interior = gray[H // 8: 7 * H // 8, W // 8: 7 * W // 8]
    interior_white_ratio = (interior > 240).mean()

    if edge_density > 0.15 and interior_white_ratio > 0.30:
        raw_score += SCORE_PLOT_STRONG
        signals.append(
            f"scientific_plot_edges{edge_density:.2f}_white{interior_white_ratio:.0%}"
        )
    elif edge_density > 0.10 and interior_white_ratio > 0.20:
        raw_score += SCORE_PLOT_MODERATE
        signals.append("possible_diagram")

    # --- Final decision ---
    is_solar = raw_score >= SOLAR_SCORE_THRESHOLD
    normalized = max(0.0, min(1.0, raw_score / SCORE_NORMALIZATION))

    if is_solar and image_type == "unknown":
        image_type = "hmi_grayscale"

    logger.debug(
        "%s: raw_score=%d, is_solar=%s, signals=%s",
        path, raw_score, is_solar, signals,
    )

    return ClassificationResult(
        is_solar=is_solar,
        score=normalized,
        signals=signals,
        image_type=image_type,
    )
