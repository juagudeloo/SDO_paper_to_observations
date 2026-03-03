import cv2
import numpy as np
import torch
import kornia as K
import kornia.feature as KF


class LightGlueMatcher:
    def __init__(self, device=None, conf_thresh=0.1, max_matches=1000):
        if device is None:
            self.device = K.utils.get_cuda_device_if_available('cuda:0') or torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Initializing DISK + LightGlueMatcher on {self.device}")
        
        # Load models
        self.disk = KF.DISK.from_pretrained("depth").eval().to(self.device)
        self.matcher = KF.LightGlueMatcher('disk').eval().to(self.device)
        self.conf_thresh = conf_thresh
        self.max_matches = max_matches

    def image_to_tensor(self, img_gray: np.ndarray) -> torch.Tensor:
        """ Convert grayscale numpy to float32 tensor (BxCxHxW) """
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        return K.image_to_tensor(img_rgb, keepdim=False).to(self.device)

    def match(self, img0_gray: np.ndarray, img1_gray: np.ndarray):
        """
        Takes 2 padded grayscale images, extracts DISK features, and matches with LightGlue.
        Returns the transform matrix M_sim, keypoints, masks, etc.
        """
        img0_t = self.image_to_tensor(img0_gray)
        img1_t = self.image_to_tensor(img1_gray)

        print(f"Extracting features for shapes: img0={img0_gray.shape}, img1={img1_gray.shape}")
        
        with torch.no_grad():
            feats0 = self.disk(img0_t)
            feats1 = self.disk(img1_t)
            corr = self.matcher({'image0': feats0, 'image1': feats1})

        # Coordinates assume padded sizes (if input images are already padded)
        # Note: In the original notebook, there was un-padding scale logic. If we run RANSAC on the
        # full padded matrices, we can just transform and keep within limits, or un-pad the coords.
        # Here we return coordinates relative to the input arrays img0_gray / img1_gray
        
        # Original coords are normalized by tensor shape for unpadding in the notebook, but if 
        # img0_t and img0_gray match size, we just need to retrieve kpts.
        # Kornia returns keypoints in [x, y] coordinates.
        kpts0 = corr['keypoints0'][0].cpu().numpy()
        kpts1 = corr['keypoints1'][0].cpu().numpy()
        scores = corr['scores'][0].cpu().numpy()

        valid_mask = scores > self.conf_thresh
        
        # Apply mask and cap max matches for RANSAC speed
        src_pts = kpts0[valid_mask][:self.max_matches].reshape(-1, 1, 2)
        dst_pts = kpts1[valid_mask][:self.max_matches].reshape(-1, 1, 2)
        
        valid_scores = scores[valid_mask][:self.max_matches]

        print(f"LightGlue returned {len(src_pts)} valid matches (conf > {self.conf_thresh}).")
        
        return src_pts, dst_pts, valid_scores

    @staticmethod
    def estimate_transform(src_pts, dst_pts, ransac_thresh=3.0, confidence=0.995):
        if len(src_pts) < 10:
            print("Too few points for RANSAC (< 10).")
            return None, None

        M_sim, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=ransac_thresh, 
            confidence=confidence
        )
        
        num_inliers = np.sum(inlier_mask)
        print(f"RANSAC found {num_inliers}/{len(src_pts)} inliers ({(num_inliers/len(src_pts))*100:.1f}%).")
        
        if M_sim is not None:
             scale = np.sqrt(M_sim[0,0]**2 + M_sim[1,0]**2)
             angle_deg = np.degrees(np.arctan2(M_sim[1,0], M_sim[0,0]))
             tx, ty = M_sim[0,2], M_sim[1,2]
             print(f"Transform params - Scale: {scale:.3f}, Rot: {angle_deg:.1f}°, Tx: {tx:.1f}, Ty: {ty:.1f}")
            
        return M_sim, inlier_mask
