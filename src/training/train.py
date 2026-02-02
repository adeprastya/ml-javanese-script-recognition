import torch
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = 0.0
    global_step = 0

    for images, labels, label_lens, input_lens, filenames in tqdm(
        loader, desc="Train", leave=False
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

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

    avg_loss = total_loss / len(loader)
    return avg_loss, global_step
