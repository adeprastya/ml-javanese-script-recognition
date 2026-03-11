from typing import Dict
import torch
from torch.utils.data import ConcatDataset, DataLoader
from data.dataset import JavaneseOCRDataset
from data.collate import ctc_collate


def create_dataloaders(
    data_sources: Dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple:
    """Create train/val/test DataLoaders."""

    train_ds = ConcatDataset(
        [
            JavaneseOCRDataset(
                src["csv"],
                src["img_dir"],
                preprocessing=src["prep"],
                augmentation=src["aug"],
            )
            for src in data_sources["train"]
        ]
    )
    val_ds = ConcatDataset(
        [
            JavaneseOCRDataset(
                src["csv"],
                src["img_dir"],
                preprocessing=src["prep"],
                augmentation=src["aug"],
            )
            for src in data_sources["val"]
        ]
    )
    test_ds = ConcatDataset(
        [
            JavaneseOCRDataset(
                src["csv"],
                src["img_dir"],
                preprocessing=src["prep"],
                augmentation=src["aug"],
            )
            for src in data_sources["test"]
        ]
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ctc_collate,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ctc_collate,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ctc_collate,
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
