import torch
from tqdm import tqdm

from decoding.ctc_decoder import ctc_greedy_decode, decode_targets
from data.vocabulary import IDX2CHAR


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
):
    model.eval()
    all_refs = []
    all_preds = []
    all_imgs = []

    for (
        images,
        labels,
        label_lens,
        input_lens,
        filenames,
    ) in tqdm(loader, desc="Inference", leave=False):
        images = images.to(device, non_blocking=True)

        # forward
        logits = model(images)  # [B, T, C]
        preds = logits.argmax(dim=2)  # [B, T]

        # decode
        batch_refs = decode_targets(labels, label_lens, IDX2CHAR)
        batch_preds = [ctc_greedy_decode(p, IDX2CHAR) for p in preds]

        all_preds.extend(batch_preds)
        all_refs.extend(batch_refs)
        all_imgs.extend(filenames)

    return all_preds, all_refs, all_imgs
