"""
CTC Collate Function for DataLoader.

Handles variable-width images by padding to GPU-optimal dimensions
and preparing tensors for CTC loss computation.
"""

from typing import List, Tuple
import torch


def ctc_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, str]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[str, ...]]:
    """
    Collate function for CTC loss. Pads images to max width (multiple of 4 for GPU efficiency).

    Args:
        batch: List of (image[C,H,W], label[L], label_len, filename)

    Returns:
        images[B,C,H,W], labels[sum(L)], label_lens[B], input_lens[B], filenames
    """

    images, labels, label_lens, filenames = zip(*batch)

    # Round max width to multiple of 4 for GPU efficiency
    widths = [img.shape[-1] for img in images]
    max_width = ((max(widths) + 3) // 4) * 4

    padded_images = []
    input_lens = []

    for img, width in zip(images, widths):
        # Pad to max_width (value=1.0 for white background in 0-1 range)
        padded_img = torch.nn.functional.pad(img, (0, max_width - width), value=1.0)
        padded_images.append(padded_img)

        # CNN downsamples width by 4x
        input_lens.append(width // 4)

    return (
        torch.stack(padded_images),  # [B,C,H,W]
        torch.cat(labels),  # [sum(L)]
        torch.tensor(label_lens, dtype=torch.long),  # [B]
        torch.tensor(input_lens, dtype=torch.long),  # [B]
        filenames,
    )
