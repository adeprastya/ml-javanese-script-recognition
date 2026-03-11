"""
Javanese OCR Dataset for PyTorch DataLoader.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.vocabulary import CHAR2IDX


class JavaneseOCRDataset(Dataset):
    """
    Dataset for Javanese script recognition with CTC.

    Args:
        csv_path: Path to CSV with columns: 'image', 'transcription'
        img_dir: Directory containing images
        preprocessing: PIL transform
        augmentation: Albumentations transform (applied before preprocessing)
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        preprocessing: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
    ):
        # Validate inputs
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)

        # Validate required columns
        required_cols = ["image", "transcription"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing columns: {missing_cols}")

        self.preprocessing = preprocessing
        self.augmentation = augmentation

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to character indices, filtering unknown characters."""
        indices = [CHAR2IDX[c] for c in text if c in CHAR2IDX]
        if not indices:
            raise ValueError(f"No valid characters in text: {text}")
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """
        Returns:
            image: [C, H, W] tensor
            label: [L] encoded character indices
            label_len: length of label
            filename: image filename
        """
        row = self.df.iloc[idx]
        filename = row["image"]
        img_path = self.img_dir / filename

        # Load image
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert(
            "RGB"
        )  # RGB for Albumentations compatibility

        # Augmentation (Albumentations expects numpy array)
        if self.augmentation is not None:
            img_np = np.array(img)
            augmented = self.augmentation(image=img_np)
            img = Image.fromarray(augmented["image"])

        # Preprocessing (Torchvision transforms expect PIL Image)
        if self.preprocessing is not None:
            img = self.preprocessing(img)

        # Encode label
        label = self.encode(row["transcription"])

        return img, label, len(label), filename
