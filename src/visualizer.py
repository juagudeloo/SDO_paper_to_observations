import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class Visualizer:
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_match_visualizations(self, paper_img, original_img, src_pts, dst_pts, inlier_mask, m_sim):
        """ Draw overlays, rectangles and match lines between the pictures. """
        ho, wo = original_img.shape[:2]
        hp, wp = paper_img.shape[:2]
        
        # Color conversion for visualization
        paper_c = cv2.cvtColor(paper_img, cv2.COLOR_GRAY2BGR)
        orig_c = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        overlay_path = self.output_dir / 'overlay_lightglue.jpg'
        rect_path = self.output_dir / 'rect_lightglue.jpg'
        matches_path = self.output_dir / 'matches_lightglue.jpg'
        crop_path = self.output_dir / 'ar_crop_lightglue.jpg'
        
        # 1. Overlay image
        warped = cv2.warpAffine(paper_c, m_sim, (wo, ho))
        overlay = cv2.addWeighted(orig_c, 0.7, warped, 0.3, 0)
        cv2.imwrite(str(overlay_path), overlay)
        
        # 2. Draw rectangle on original
        corners = np.float32([[0, 0], [wp, 0], [wp, hp], [0, hp]]).reshape(-1, 1, 2)
        tcorners = cv2.transform(corners, m_sim).astype(int)
        
        rect_img = orig_c.copy()
        cv2.polylines(rect_img, [tcorners], True, (0, 255, 0), 3)
        cv2.imwrite(str(rect_path), rect_img)
        
        # 3. Match Visualization (Draw top 100 inliers)
        # Filter purely by inliers
        inlier_indices = np.nonzero(inlier_mask.ravel() == 1)[0]
        top_inliers = inlier_indices[:100]
        
        k0 = cv2.KeyPoint_convert(src_pts[top_inliers])
        k1 = cv2.KeyPoint_convert(dst_pts[top_inliers])
        matches = [cv2.DMatch(i, i, 0) for i in range(len(top_inliers))]
        
        match_img = cv2.drawMatches(paper_c, k0, orig_c, k1, matches, None, flags=0)
        cv2.imwrite(str(matches_path), match_img)
        
        # 4. Extract aligned/matched bounding box crop
        xmin, xmax = tcorners[:, 0, 0].min(), tcorners[:, 0, 0].max()
        ymin, ymax = tcorners[:, 0, 1].min(), tcorners[:, 0, 1].max()
        xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
        xmax, ymax = min(wo, int(xmax)), min(ho, int(ymax))
        
        ar_crop = original_img[ymin:ymax, xmin:xmax]
        cv2.imwrite(str(crop_path), ar_crop)
        
        return match_img, ar_crop, {
            "overlay": overlay_path,
            "rectangle": rect_path,
            "matches": matches_path,
            "crop": crop_path,
        }

    def plot_summary(self, paper_gray, original_raw, match_img, ar_crop, show: bool = False):
        """ Plot a beautiful 4 panel summary figure that was requested in the nb. """
        plt.figure(figsize=(20, 5))
        plt.subplot(141)
        plt.imshow(paper_gray, cmap='gray')
        plt.title('Paper Image')
        
        plt.subplot(142)
        plt.imshow(original_raw, cmap='gray')
        plt.title('Raw Continuum')
        
        plt.subplot(143)
        if match_img is not None:
            plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
            plt.title('LightGlue Matches (Inliers)')
        else:
            plt.text(0.5, 0.5, 'No Matches', ha='center', fontsize=16)
            plt.title('Matches')
            
        plt.subplot(144)
        if ar_crop is not None and ar_crop.shape[0] > 0 and ar_crop.shape[1] > 0:
            plt.imshow(ar_crop, cmap='gray')
            plt.title('AR Crop Region')
        else:
            plt.text(0.5, 0.5, 'No valid region', ha='center', fontsize=16)
            plt.title('AR Crop')
            
        plt.tight_layout()
        summary_path = self.output_dir / 'lightglue_summary_final.png'
        plt.savefig(str(summary_path), dpi=150)
        print(f"Summary plot saved to {summary_path}")
        if show:
            plt.show()
        plt.close()
        return summary_path
