"""
Validation Loop for CTC-based OCR.
"""

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vocabulary import IDX2CHAR
from decoding.ctc_decoder import best_path_decode, decode_targets


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.CTCLoss,
    device: torch.device,
) -> Tuple[float, List[str], List[str]]:
    """
    Validate model for one epoch.

    Args:
        model: CNN-BiLSTM model
        loader: Validation DataLoader
        criterion: CTC loss function
        device: Device to run on (cpu/cuda)

    Returns:
        Tuple of (average_loss, predictions, references)
    """

    model.eval()

    if len(loader) == 0:
        raise ValueError("DataLoader is empty")

    total_loss = 0.0
    num_batches = 0
    all_preds, all_refs = [], []

    for images, labels, label_lens, input_lens, _ in tqdm(
        loader, desc="Val", leave=False
    ):
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        label_lens = label_lens.to(device)
        input_lens = input_lens.to(device)

        # Forward pass: [B, T, C] logits
        logits = model(images)

        # CTC expects log probabilities: [T, B, C]
        log_probs = logits.log_softmax(2).permute(1, 0, 2)

        # Compute CTC loss
        loss = criterion(
            log_probs,
            labels,
            input_lens,
            label_lens,
        )

        # Decode predictions (move to CPU for decoding)
        pred_indices = logits.argmax(dim=2).cpu()  # [B, T]
        preds = [best_path_decode(seq, IDX2CHAR, blank=0) for seq in pred_indices]

        # Decode ground truth
        refs = decode_targets(labels.cpu(), label_lens.cpu(), IDX2CHAR)

        all_preds.extend(preds)
        all_refs.extend(refs)
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss, all_preds, all_refs
