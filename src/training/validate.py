import torch
from tqdm import tqdm

from data.vocabulary import IDX2CHAR
from decoding.ctc_decoder import ctc_greedy_decode, decode_targets


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.CTCLoss,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    all_preds, all_refs = [], []

    for images, labels, label_lens, input_lens, filenames in tqdm(
        loader, desc="Val", leave=False
    ):
        images = images.to(device)
        labels = labels.to(device)
        label_lens = label_lens.to(device)
        input_lens = input_lens.to(device)

        logits = model(images)
        log_probs = logits.log_softmax(2)

        loss = criterion(
            log_probs.permute(1, 0, 2),
            labels,
            input_lens,
            label_lens,
        )

        preds = [ctc_greedy_decode(p, IDX2CHAR) for p in logits.argmax(2)]
        refs = decode_targets(labels, label_lens, IDX2CHAR)

        all_preds.extend(preds)
        all_refs.extend(refs)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_refs
