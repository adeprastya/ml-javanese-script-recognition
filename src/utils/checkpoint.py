from pathlib import Path
import logging
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    val_loss: float,
    cer: float,
    save_path: Path,
    logger: logging.Logger,
) -> None:
    """Save model checkpoint."""

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "cer": cer,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path.name}")
