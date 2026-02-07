import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

from data.vocabulary import CHAR2IDX


class JavaneseOCRDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        preprocessing=None,
        augmentation=None,
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

        self.preprocessing = preprocessing
        self.augmentation = augmentation

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(
            [CHAR2IDX[c] for c in text if c in CHAR2IDX],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["image"])
        img = Image.open(img_path).convert("RGB")

        # Augmentation
        if self.augmentation is not None:
            img_np = np.array(img)
            augmented = self.augmentation(image=img_np)
            img = Image.fromarray(augmented["image"])

        # Preprocessing
        if self.preprocessing is not None:
            img = self.preprocessing(img)

        # Label encoding
        label = self.encode(row["transcription"])

        return img, label, len(label), row["image"]
