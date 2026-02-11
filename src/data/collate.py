import torch


def ctc_collate(batch):
    images, labels, label_lens, filenames = zip(*batch)

    widths = [img.shape[-1] for img in images]
    max_width = (
        (max(widths) + 3) // 4
    ) * 4  # round to nearest multiple of 4 (GPU optimization)

    padded_images = []
    input_lens = []

    for img, w in zip(images, widths):
        padded_images.append(
            torch.nn.functional.pad(img, (0, max_width - w), value=1.0)
        )
        input_lens.append(w // 4)

    return (
        torch.stack(padded_images),  # [B, C(1), H, W_max]
        torch.cat(labels),  # [sum(label_len)]
        torch.tensor(label_lens),  # [B]
        torch.tensor(input_lens),  # [B]
        filenames,
    )
