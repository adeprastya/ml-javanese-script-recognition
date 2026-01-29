import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.aksara import CHAR2IDX
from src.transforms import ResizeByHeight, CLAHE


class JavaneseOCRDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        img_height: int,
        augment=None,
        clahe: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.augment = augment

        base_transforms = [T.Grayscale(1)]

        if clahe:
            base_transforms.append(CLAHE())

        base_transforms += [
            ResizeByHeight(img_height),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]

        self.transform = T.Compose(base_transforms)

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

        if self.augment is not None:
            img = self.augment(img)

        img = self.transform(img)
        label = self.encode(row["transcription"])

        return img, label, len(label)


def ctc_collate_fn(batch):
    images, labels, label_lens = zip(*batch)

    widths = [img.shape[-1] for img in images]
    max_width = ((max(widths) + 3) // 4) * 4  # align to stride 4

    padded_images = []
    input_lens = []

    for img, w in zip(images, widths):
        padded_images.append(torch.nn.functional.pad(img, (0, max_width - w)))
        input_lens.append(w // 4)

    return (
        torch.stack(padded_images),  # [B, 1, H, W_max]
        torch.cat(labels),  # [sum(label_len)]
        torch.tensor(label_lens),  # [B]
        torch.tensor(input_lens),  # [B]
    )
