def main():
    print("Importing modules...")

    import os
    import time
    import random
    from datetime import datetime

    import numpy as np

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, ConcatDataset

    from utils.path import PROJECT_ROOT
    from data.vocabulary import NUM_CLASSES, BLANK_IDX
    from data.dataset import JavaneseOCRDataset
    from data.collate import ctc_collate
    from model.cnn_bilstm import CNNBiLSTM
    from metric.ocr_metrics import batch_cer, batch_em
    from transform.augmentation import augmentation_transform
    from transform.preprocessing import preprocessing_transform
    from training.train import train_one_epoch
    from training.validate import validate_one_epoch
    from training.test import test_one_epoch
    from training.logging import (
        save_training_logs,
        save_training_plot,
        save_test_result,
    )

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

    print("Loading Config...")

    MODEL_NAME = "null"

    CNN_LAYER = 5
    BILSTM_LAYER = 2

    LEARNING_RATE = 1e-4

    EPOCHS = 50
    EARLY_STOP_PATIENCE = 7
    CER_EPS = 1e-3
    LOSS_EPS = 1e-4

    IMG_HEIGHT = 64
    BATCH_SIZE = 8
    NUM_WORKERS = 3

    MODEL_DIR = f"{PROJECT_ROOT}/builds/{MODEL_NAME}"
    BASE_SYNT_DIR = f"{PROJECT_ROOT}/dataset/word_nglegena_synthetic_20260130_155231"
    BASE_REAL_DIR = f"{PROJECT_ROOT}/dataset/word_nglegena_handwritten_20260130_155805"

    DATA_SOURCES = {
        "train": [
            {
                "csv": f"{BASE_SYNT_DIR}/label_train.csv",
                "img_dir": f"{BASE_SYNT_DIR}/image_train",
                "aug": augmentation_transform(prob=1.0, seed=SEED),
                "prep": preprocessing_transform(img_height=IMG_HEIGHT, enhance=False),
            },
        ],
        "val": [
            {
                "csv": f"{BASE_SYNT_DIR}/label_val.csv",
                "img_dir": f"{BASE_SYNT_DIR}/image_val",
                "aug": None,
                "prep": preprocessing_transform(img_height=IMG_HEIGHT, enhance=False),
            },
        ],
        "test": [
            {
                "csv": f"{BASE_REAL_DIR}/label_1.csv",
                "img_dir": f"{BASE_REAL_DIR}/image_1",
                "aug": None,
                "prep": preprocessing_transform(img_height=IMG_HEIGHT, enhance=False),
            },
            {
                "csv": f"{BASE_REAL_DIR}/label_2.csv",
                "img_dir": f"{BASE_REAL_DIR}/image_2",
                "aug": None,
                "prep": preprocessing_transform(img_height=IMG_HEIGHT, enhance=False),
            },
            {
                "csv": f"{PROJECT_ROOT}/dataset/test-sample/label.csv",
                "img_dir": f"{PROJECT_ROOT}/dataset/test-sample/image",
                "aug": None,
                "prep": preprocessing_transform(img_height=IMG_HEIGHT, enhance=True),
            },
        ],
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.listdir(MODEL_DIR):
        raise RuntimeError("Model directory is not empty")

    # =======================
    # DATA
    # =======================

    print("Loading data...")

    train_ds = ConcatDataset(
        [
            JavaneseOCRDataset(
                str(src["csv"]),
                str(src["img_dir"]),
                preprocessing=src["prep"],
                augmentation=src["aug"],
            )
            for src in DATA_SOURCES["train"]
        ]
    )
    val_ds = ConcatDataset(
        [
            JavaneseOCRDataset(
                str(src["csv"]),
                str(src["img_dir"]),
                preprocessing=src["prep"],
                augmentation=src["aug"],
            )
            for src in DATA_SOURCES["val"]
        ]
    )
    test_ds = ConcatDataset(
        [
            JavaneseOCRDataset(
                str(src["csv"]),
                str(src["img_dir"]),
                preprocessing=src["prep"],
                augmentation=src["aug"],
            )
            for src in DATA_SOURCES["test"]
        ]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=ctc_collate,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=ctc_collate,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=ctc_collate,
        generator=generator,
    )

    # =======================
    # MODEL
    # =======================

    print("Building model...")

    model = CNNBiLSTM(NUM_CLASSES, cnn_layers=CNN_LAYER, rnn_layers=BILSTM_LAYER).to(
        DEVICE
    )
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # =======================
    # TRAINING
    # =======================

    print("Training model...")

    epoch_logs = []
    start_time = time.time()
    train_start_wall = datetime.now()
    train_start_ts = time.time()

    global_step = 0
    best_cer = float("inf")
    best_val_loss_cp = float("inf")

    best_val_loss_es = float("inf")
    trigger_times = 0
    early_stopped = False
    early_stopped_at = None

    for epoch in range(EPOCHS):
        train_loss, step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
        )
        global_step += step

        val_loss, all_preds, all_refs = validate_one_epoch(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        cer = batch_cer(all_preds, all_refs)
        em = batch_em(all_preds, all_refs)

        # ---------- SCHEDULER ----------
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ---------- LOG ----------
        log_line = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "cer": float(cer),
            "em": float(em),
            "global_step": global_step,
        }
        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"CER: {cer:.4f} | "
            f"EM: {em:.4f}"
        )
        epoch_logs.append(log_line)

        # -------- CHECKPOINT SAVE ---------
        if (cer < best_cer - CER_EPS) or (
            abs(cer - best_cer) <= CER_EPS and val_loss < best_val_loss_cp - LOSS_EPS
        ):
            best_cer = cer
            best_val_loss_cp = val_loss
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
        if val_loss < best_val_loss_es - LOSS_EPS:
            best_val_loss_es = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if (trigger_times >= EARLY_STOP_PATIENCE) and (not early_stopped):
                early_stopped = True
                early_stopped_at = epoch
                print(f"## Early stopping triggered at epoch {epoch+1}")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "epoch": epoch + 1,
                        "step": global_step,
                    },
                    f"{MODEL_DIR}/stopped_model.pth",
                )
                print(">> Saved early stopped model")

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
    # TRAINING LOGGING
    # =======================

    print("Creating logs...")

    train_end_wall = datetime.now()
    train_end_ts = time.time()
    train_duration_sec = int(train_end_ts - train_start_ts)
    duration = int(time.time() - start_time)

    print(
        f"Training finished in {duration // 3600}h {duration % 3600 // 60}m {duration % 60}s"
    )
    save_training_logs(
        model_dir=MODEL_DIR,
        model_name=MODEL_NAME,
        device=DEVICE,
        cnn_layers=CNN_LAYER,
        bilstm_layers=BILSTM_LAYER,
        total_params=total_params,
        trainable_params=trainable_params,
        train_ds=train_ds,
        val_ds=val_ds,
        datasets=DATA_SOURCES,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        batch_size=BATCH_SIZE,
        epoch_logs=epoch_logs,
        train_start_wall=train_start_wall,
        train_end_wall=train_end_wall,
        train_duration_sec=train_duration_sec,
        early_stopped=early_stopped,
        early_stopped_at=early_stopped_at,
        best_val_cer=best_cer,
        best_val_loss=best_val_loss_cp,
    )
    save_training_plot(epoch_logs, MODEL_DIR)

    # =======================
    # TESTING
    # =======================

    print("Testing model...")

    model_paths = [
        f"{MODEL_DIR}/best_model.pth",
        f"{MODEL_DIR}/stopped_model.pth",
        f"{MODEL_DIR}/last_model.pth",
    ]

    all_results = {}

    for path in model_paths:
        model_name = os.path.basename(path).replace(".pth", "")

        if not os.path.exists(path):
            print(f"## {model_name} not found, skipping")
            continue

        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        all_preds, all_refs, all_imgs = test_one_epoch(model, test_loader, DEVICE)
        all_results[model_name] = {
            "imgs": all_imgs,
            "preds": all_preds,
            "refs": all_refs,
        }

        cer_val = batch_cer(all_preds, all_refs)
        em_val = batch_em(all_preds, all_refs)
        print(f"{model_name}: CER={cer_val:.4f}, EM={em_val:.4f}")

    save_test_result(MODEL_NAME, all_results, MODEL_DIR, "test_results.json")


if __name__ == "__main__":
    main()
