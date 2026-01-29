def main():
    print("Importing modules...")

    import os
    import time
    import random
    import json
    from datetime import datetime

    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, ConcatDataset

    from src.aksara import CHAR_LIST, IDX2CHAR, BLANK_IDX
    from src.dataset import JavaneseOCRDataset, ctc_collate_fn
    from src.cnn_bilstm import CNNBiLSTM
    from src.decoding import ctc_greedy_decode, decode_targets
    from src.ocr_metrics import character_error_rate, exact_match_accuracy

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # =======================
    # REPRODUCIBILITY
    # =======================
    print("Setting reproducibility...")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(SEED)

    # =======================
    # CONFIG
    # =======================
    print("Configuring...")

    IMG_HEIGHT = 64
    BATCH_SIZE = 16
    NUM_WORKERS = 3
    EPOCHS = 5
    EARLY_STOP_PATIENCE = 10
    MODEL_NAME = "test"

    BASE_DATA_DIR = "./data/word_nglegena_synthetic_20260124_160250"
    MODEL_DIR = f"./models/{MODEL_NAME}"

    TRAIN_CSV = f"{BASE_DATA_DIR}/label_train.csv"
    TRAIN_AUG_CSV = f"{BASE_DATA_DIR}/1_2x_label_train_aug.csv"
    VAL_CSV = f"{BASE_DATA_DIR}/label_val.csv"

    TRAIN_IMG = f"{BASE_DATA_DIR}/image_train"
    TRAIN_AUG_IMG = f"{BASE_DATA_DIR}/1_2x_image_train_aug"
    VAL_IMG = f"{BASE_DATA_DIR}/image_val"

    # =======================
    # MODEL
    # =======================
    print("Building model...")

    num_classes = len(CHAR_LIST) + 1  # + blank
    model = CNNBiLSTM(num_classes).to(DEVICE)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # =======================
    # DATA
    # =======================
    print("Loading data...")

    syn_train_ds = JavaneseOCRDataset(TRAIN_CSV, TRAIN_IMG, IMG_HEIGHT)
    aug_train_ds = JavaneseOCRDataset(TRAIN_AUG_CSV, TRAIN_AUG_IMG, IMG_HEIGHT)
    # real_train_ds =

    syn_val_ds = JavaneseOCRDataset(VAL_CSV, VAL_IMG, IMG_HEIGHT)
    # real_val_ds =

    train_ds = ConcatDataset(
        [
            syn_train_ds,
            aug_train_ds,
        ]
    )
    val_ds = ConcatDataset([syn_val_ds])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=ctc_collate_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=ctc_collate_fn,
        generator=generator,
    )

    # =======================
    # TRAINING
    # =======================
    print("Training model...")

    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.listdir(MODEL_DIR):
        raise RuntimeError("Model directory is not empty")

    epoch_logs = []
    start_time = time.time()
    train_start_wall = datetime.now()
    train_start_ts = time.time()

    best_val_loss = float("inf")
    best_cer = float("inf")
    trigger_times = 0
    early_stopped = False
    global_step = 0

    for epoch in range(EPOCHS):
        # ---------- TRAIN PHASE ----------
        model.train()
        train_loss = 0.0

        for images, labels, label_lens, input_lens in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            label_lens = label_lens.to(DEVICE)
            input_lens = input_lens.to(DEVICE)

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

            global_step += 1
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------- VALIDATION PHASE ----------
        model.eval()
        val_loss = 0.0
        all_preds, all_refs = [], []

        with torch.no_grad():
            for images, labels, label_lens, input_lens in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False
            ):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                label_lens = label_lens.to(DEVICE)
                input_lens = input_lens.to(DEVICE)

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
                val_loss += loss.item()

        val_loss /= len(val_loader)
        cer = character_error_rate(all_preds, all_refs)
        em = exact_match_accuracy(all_preds, all_refs)

        # ---------- SCHEDULER ----------
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ------------- LOG -------------
        log_line = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "cer": float(cer),
            "em": float(em),
            "global_step": global_step,
        }
        print(
            f"Epoch {log_line['epoch']:02d} | "
            f"Train Loss: {log_line['train_loss']:.4f} | "
            f"Val Loss: {log_line['val_loss']:.4f} | "
            f"CER: {log_line['cer']:.4f} | "
            f"EM: {log_line['em']:.4f}"
        )
        epoch_logs.append(log_line)

        # -------- CHECKPOINT SAVE ---------
        if cer < best_cer:
            best_cer = cer
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "best_cer": best_cer,
                    "val_loss": val_loss,
                    "epoch": epoch + 1,
                    "step": global_step,
                },
                f"{MODEL_DIR}/best_model.pth",
            )
            print(">> Saved best model")

        # ---------- EARLY STOPPING ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= EARLY_STOP_PATIENCE:
                print("## Early stopping")
                early_stopped = True
                break

    # ---------- FINAL SAVE ----------
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch + 1,
            "step": global_step,
        },
        f"{MODEL_DIR}/last_model.pth",
    )
    print(">> Saved last model")

    # =======================
    # LOGGING
    # =======================
    print("Creating logs...")

    train_end_wall = datetime.now()
    train_end_ts = time.time()
    train_duration_sec = int(train_end_ts - train_start_ts)
    duration = int(time.time() - start_time)
    print(f"Training finished in {duration // 60}m {duration % 60}s")

    def format_duration(sec):
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ---------- JSON LOG ----------
    log_data = {
        "meta": {
            "model_name": MODEL_NAME,
            "device": str(DEVICE),
        },
        "dataset": {
            "train": {
                "total_samples": len(train_ds),
                "components": [
                    {
                        "name": "synthetic_train",
                        "type": "synthetic",
                        "num_samples": len(syn_train_ds),
                        "csv": os.path.basename(TRAIN_CSV),
                        "img_dir": os.path.basename(TRAIN_IMG),
                    },
                    {
                        "name": "augmented_train",
                        "type": "augmented",
                        "num_samples": len(aug_train_ds),
                        "csv": os.path.basename(TRAIN_AUG_CSV),
                        "img_dir": os.path.basename(TRAIN_AUG_IMG),
                    },
                ],
            },
            "val": {
                "total_samples": len(val_ds),
                "components": [
                    {
                        "name": "synthetic_val",
                        "type": "synthetic",
                        "num_samples": len(syn_val_ds),
                        "csv": os.path.basename(VAL_CSV),
                        "img_dir": os.path.basename(VAL_IMG),
                    },
                ],
            },
        },
        "charset": {
            "size": len(IDX2CHAR),
            "blank_index": BLANK_IDX,
            "mapping": IDX2CHAR,
        },
        "config": {
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__ if scheduler else None,
            "criterion": criterion.__class__.__name__,
            "epochs": EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "batch_size": BATCH_SIZE,
        },
        "epochs": epoch_logs,
        "summary": {
            "start_time": train_start_wall.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": train_end_wall.strftime("%Y-%m-%d %H:%M:%S"),
            "training_duration": format_duration(train_duration_sec),
            "training_duration_sec": train_duration_sec,
            "early_stopped": early_stopped,
            "best_val_cer": float(best_cer),
            "best_val_loss": float(best_val_loss),
        },
    }
    with open(f"{MODEL_DIR}/logs.json", "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # ---------- PLOT LOG ----------
    x = [e["epoch"] for e in epoch_logs]
    train_loss = [e["train_loss"] for e in epoch_logs]
    val_loss = [e["val_loss"] for e in epoch_logs]
    cer = [e["cer"] for e in epoch_logs]
    em = [e["em"] for e in epoch_logs]
    # Train & Val Loss
    plt.figure(figsize=(8, 5))
    plt.plot(x, train_loss, label="Train Loss")
    plt.plot(x, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CTC Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/loss_curve.png")
    plt.close()
    # CER
    plt.figure(figsize=(8, 5))
    plt.plot(x, cer, label="CER", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Character Error Rate")
    plt.title("Validation CER")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/cer_curve.png")
    plt.close()
    # EM
    plt.figure(figsize=(8, 5))
    plt.plot(x, em, label="EM", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Exact Match Accuracy")
    plt.title("Validation EM")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/em_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
