"""
Data Augmentation for Javanese Script OCR.
"""

from typing import Optional

import albumentations as A
import cv2
import numpy as np


def random_padding(
    img: np.ndarray,
    max_horizontal: int = 20,
    max_vertical: int = 10,
    fill_value: int = 255,
    **kwargs,
) -> np.ndarray:
    """
    Apply random padding to image borders.

    Args:
        img: Input image (uint8 or float32)
        max_horizontal: Maximum horizontal padding (left/right)
        max_vertical: Maximum vertical padding (top/bottom)
        fill_value: Padding color (255 for white)

    Returns:
        Padded image
    """

    # Ensure uint8 format
    if img.dtype != np.uint8:
        if img.max() <= 1.1:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Determine fill color (grayscale vs RGB)
    fill = fill_value if img.ndim == 2 else (fill_value,) * 3

    # Random padding amounts
    pad_left = np.random.randint(0, max_horizontal + 1)
    pad_right = np.random.randint(0, max_horizontal + 1)
    pad_top = np.random.randint(0, max_vertical + 1)
    pad_bottom = np.random.randint(0, max_vertical + 1)

    return cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=fill
    )


def get_augmentation_pipeline(
    prob: float = 1.0,
    seed: Optional[int] = None,
) -> A.Compose:
    """
    Create augmentation pipeline for training.

    Augmentations:
        - Random padding (50%): Add white borders
        - Morphological ops (50%): Dilation or erosion (ink variation)
        - Affine transforms (50%): Scale, rotation, shear
        - Gaussian noise (50%): Sensor/scanning noise
        - Gaussian blur (50%): Focus/compression artifacts
        - Brightness/Contrast (50%): Lighting variations

    Args:
        prob: Overall probability of applying augmentations
        seed: Random seed for reproducibility

    Returns:
        Albumentations composition
    """

    return A.Compose(
        [
            # Add white borders (simulates varied margins)
            A.Lambda(image=random_padding, p=0.5),
            # Ink thickness variation
            A.OneOf(
                [
                    A.Morphological(scale=(3, 5), operation="dilation", p=1.0),
                    A.Morphological(scale=(3, 5), operation="erosion", p=1.0),
                ],
                p=0.5,
            ),
            # Geometric transforms (writing angle, scale)
            A.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 80-120% size
                rotate=(-2, 2),  # ±2 degrees
                shear=(-3, 3),  # ±3 degrees shear
                fit_output=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(255, 255, 255),  # White fill
                p=0.5,
            ),
            # Scanning/sensor noise
            A.GaussNoise(std_range=(0.02, 0.1), mean_range=(-0.2, 0.2), p=0.5),
            # Focus/compression blur
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 2.5), p=0.5),
            # Lighting conditions
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                ensure_safe_range=True,
                p=0.5,
            ),
        ],
        p=prob,
        seed=seed,
    )
