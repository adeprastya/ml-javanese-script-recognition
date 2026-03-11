"""
Training Loop for CTC-based OCR.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
) -> Tuple[float, int]:
    """
    Train model for one epoch.

    Args:
        model: CNN-BiLSTM model
        loader: Training DataLoader
        criterion: CTC loss function
        optimizer: Optimizer (e.g., Adam, AdamW)
        device: Device to run on (cpu/cuda)
        grad_clip: Gradient clipping threshold (prevents exploding gradients)

    Returns:
        Tuple of (average_loss, num_steps)
    """

    model.train()

    if len(loader) == 0:
        raise ValueError("DataLoader is empty")

    total_loss = 0.0
    num_batches = 0

    for images, labels, label_lens, input_lens, _ in tqdm(
        loader, desc="Train", leave=False
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

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / len(loader)
    return avg_loss, num_batches
