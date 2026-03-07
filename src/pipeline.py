import argparse
import sys
from pathlib import Path

# Fix path to load local packages
sys.path.append(str(Path(__file__).parent))

from skimage import io
import numpy as np

from data_fetcher import SunpyFetcher
from preprocessor import Preprocessor
from matcher import LightGlueMatcher
from visualizer import Visualizer


def run_pipeline(
    paper_image_path: str,
    target_date: str,
    output_dir: str = "./outputs",
    download_dir: str = "./data/sunpy_images/",
    conf_thresh: float = 0.1,
    show: bool = False,
    downsample_paper: float = 1.0,
    downsample_original: float = 1.0,
):
    print("=== SDO Paper to Observation Pipeline ===")
    
    # 1. Fetch Data
    fetcher = SunpyFetcher(download_dir=download_dir)
    amap = fetcher.fetch_hmi_continuum(target_date)
    
    # 2. Preprocess Data
    print("\n--- Preprocessing Images ---")
    raw_amap_data = amap.data
    original_raw = Preprocessor.preprocess_raw_continuum(raw_amap_data)
    
    paper_img_raw = io.imread(paper_image_path, as_gray=True)
    paper_gray = Preprocessor.preprocess_paper_image(paper_img_raw)
    
    import cv2
    # compute and apply downsampling factors separately for original and paper
    if downsample_original != 1.0 or downsample_paper != 1.0:
        print(f"Downsampling images (orig={downsample_original}, paper={downsample_paper}) for faster testing...")
    if len(original_raw.shape) >= 2 and downsample_original != 1.0:
        new_h, new_w = int(original_raw.shape[0] * downsample_original), int(original_raw.shape[1] * downsample_original)
        original_raw = cv2.resize(original_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if len(paper_gray.shape) >= 2 and downsample_paper != 1.0:
        new_h, new_w = int(paper_gray.shape[0] * downsample_paper), int(paper_gray.shape[1] * downsample_paper)
        if min(new_h, new_w) < 128:
            print(f"Skipping downsample for paper image. It would become too small ({new_w}x{new_h}).")
        else:
            paper_gray = cv2.resize(paper_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad images to multiple of 16 for DISK/LightGlue
    paper_padded = Preprocessor.pad_to_multiple16(paper_gray)
    original_padded = Preprocessor.pad_to_multiple16(original_raw)
    
    # 3. Match Features
    print("\n--- Feature Matching (LightGlue) ---")
    matcher = LightGlueMatcher(conf_thresh=conf_thresh)
    src_pts, dst_pts, _ = matcher.match(paper_padded, original_padded)
    
    m_sim, inlier_mask = matcher.estimate_transform(src_pts, dst_pts)
    
    # 4. Synthesize & Visualize
    print("\n--- Visualizing & Reporting ---")
    viz = Visualizer(output_dir=output_dir)
    
    match_img, ar_crop, artifacts = None, None, {}
    if m_sim is not None:
        match_img, ar_crop, artifacts = viz.generate_match_visualizations(
            paper_padded, original_padded, src_pts, dst_pts, inlier_mask, m_sim
        )
        print("Success! Created aligned overlay and crop.")
        print("Saved match artifacts:")
        for name, path in artifacts.items():
            print(f"  - {name}: {path}")
    else:
        print("Could not find a reliable transform. Check inputs or adjust thresholds.")
        
    summary_path = viz.plot_summary(paper_gray, original_raw, match_img, ar_crop, show=show)
    print(f"Summary visualization: {summary_path}")
    if show:
        print("Displayed summary figure in a window.")
    print("Pipeline Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDO Match Pipeline")
    parser.add_argument("--paper", type=str, required=True, help="Path to input paper image.")
    parser.add_argument("--date", type=str, required=True, help="Target UTC date (e.g. '2012-07-04T09:54:53Z').")
    parser.add_argument("--outdir", type=str, default="./outputs", help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.1, help="LightGlue confidence threshold.")
    parser.add_argument("--show", action="store_true", help="Display summary plot interactively.")
    parser.add_argument("--downsample", type=float, default=None,
                        help="(deprecated) Uniform downsample factor for both images")
    parser.add_argument("--downsample-paper", type=float, default=None,
                        help="Downsample factor for the paper image only")
    parser.add_argument("--downsample-original", type=float, default=None,
                        help="Downsample factor for the HMI observation only")

    args = parser.parse_args()

    # determine effective scaling; maintain backwards compatibility
    if args.downsample is not None:
        ds_paper = ds_orig = args.downsample
    else:
        ds_paper = args.downsample_paper if args.downsample_paper is not None else 1.0
        ds_orig = args.downsample_original if args.downsample_original is not None else 1.0

    run_pipeline(
        paper_image_path=args.paper,
        target_date=args.date,
        output_dir=args.outdir,
        conf_thresh=args.conf,
        show=args.show,
        downsample_paper=ds_paper,
        downsample_original=ds_orig,
    )
