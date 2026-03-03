import cv2
import numpy as np


class Preprocessor:
    @staticmethod
    def preprocess_raw_continuum(amap_data: np.ndarray) -> np.ndarray:
        """
        Raw HMI continuum: abs(map_data) -> normalized uint8, NO histeq.
        This handles the signed magnetic data or raw intensities.
        """
        data = np.abs(amap_data).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        return cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def preprocess_paper_image(paper_img: np.ndarray) -> np.ndarray:
        """ Ensures the paper image is a proper uint8 grayscale """
        if paper_img.dtype != np.uint8:
            return (paper_img * 255).astype(np.uint8)
        return paper_img

    @staticmethod
    def pad_to_multiple16(img: np.ndarray) -> np.ndarray:
        """
        Pad image to a multiple of 16 which is a requirement for DISK features model.
        """
        h, w = img.shape[:2]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        
        # If color image
        if len(img.shape) == 3:
            return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        # If grayscale
        return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')
