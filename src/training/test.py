"""
Test/Inference Loop for CTC-based OCR.
"""

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vocabulary import IDX2CHAR
from decoding.ctc_decoder import ctc_greedy_decode, decode_targets


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Run inference on test set (no loss computation).

    Args:
        model: CNN-BiLSTM model
        loader: Test DataLoader
        device: Device to run on (cpu/cuda)

    Returns:
        Tuple of (predictions, references, filenames)
    """
    model.eval()

    if len(loader) == 0:
        raise ValueError("DataLoader is empty")

    all_preds = []
    all_refs = []
    all_filenames = []

    for (
        images,
        labels,
        label_lens,
        _,
        filenames,
    ) in tqdm(loader, desc="Testing", leave=False):
        # Move to device with async transfer
        images = images.to(device, non_blocking=True)

        # Forward pass: [B, T, C] logits
        logits = model(images)

        # Decode predictions (move to CPU for decoding)
        pred_indices = logits.argmax(dim=2).cpu()  # [B, T]
        batch_preds = [ctc_greedy_decode(seq, IDX2CHAR) for seq in pred_indices]

        # Decode ground truth (already on CPU)
        batch_refs = decode_targets(labels, label_lens, IDX2CHAR)

        all_preds.extend(batch_preds)
        all_refs.extend(batch_refs)
        all_filenames.extend(filenames)

    return all_preds, all_refs, all_filenames
