"""
Preprocessing Transforms for Javanese OCR.
"""

from typing import Tuple

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageOps


class CLAHE:
    """
    Contrast Limited Adaptive Histogram Equalization for PIL Images.

    Enhances local contrast, useful for low-quality scans or photos.
    Works on grayscale or RGB (applied to L channel in LAB space).
    """

    def __init__(
        self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        """
        Args:
            clip_limit: Threshold for contrast limiting (higher = more contrast)
            tile_grid_size: Size of grid for histogram equalization
        """

        # Validate parameters
        if clip_limit <= 0:
            raise ValueError(f"clip_limit must be positive, got {clip_limit}")
        if any(x <= 0 for x in tile_grid_size):
            raise ValueError(f"tile_grid_size must be positive, got {tile_grid_size}")

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = None  # Lazy initialization

    def __call__(self, img: Image.Image) -> Image.Image:
        # Initialize CLAHE object on first call
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
            )

        img_np = np.array(img)

        # Grayscale: apply directly
        if img_np.ndim == 2:
            enhanced = self._clahe.apply(img_np)
            return Image.fromarray(enhanced)

        # RGB: apply to L channel in LAB color space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge((l, a, b))
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


class ResizeByHeight:
    """
    Resize image to target height while preserving aspect ratio.
    """

    def __init__(self, height: int):
        """
        Args:
            height: Target height in pixels
        """
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")
        self.height = height

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if h == 0:
            raise ValueError("Image height is 0")

        # Calculate new width maintaining aspect ratio
        new_w = int(w * self.height / h)
        return img.resize((new_w, self.height), Image.BILINEAR)


class Invert:
    """
    Invert image colors (black ↔ white).

    Useful for handling black-on-white vs white-on-black text.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.invert(img)


def get_preprocessing_pipeline(img_height: int, enhance: bool = False) -> T.Compose:
    """
    Create preprocessing pipeline for inference/training.

    Pipeline:
        1. Convert to grayscale (OCR works on single channel)
        2. (Optional) CLAHE contrast enhancement
        3. Resize to target height (preserve aspect ratio)
        4. Convert to tensor [0, 1] normalized

    Args:
        img_height: Target image height (e.g., 48)
        enhance: Apply CLAHE for low-quality images

    Returns:
        Composed torchvision transforms
    """
    if img_height <= 0:
        raise ValueError(f"img_height must be positive, got {img_height}")

    transforms = [
        T.Grayscale(num_output_channels=1),  # RGB/RGBA → Grayscale
    ]

    if enhance:
        transforms.append(CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)))

    transforms.extend(
        [
            ResizeByHeight(img_height),  # Resize to target height
            T.ToTensor(),  # Convert to [C, H, W] tensor, normalize to [0, 1]
        ]
    )

    return T.Compose(transforms)
