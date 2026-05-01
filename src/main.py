import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from data.vocabulary import BLANK_IDX, NUM_CLASSES
from transform.augmentation import get_augmentation_pipeline
from transform.preprocessing import get_preprocessing_pipeline
from model.cnn_bilstm import CNNBiLSTM
from training.test import test_one_epoch, DecodeMethod
from training.train import train_one_epoch
from training.validate import validate_one_epoch
from metric.ocr_metrics import batch_cer, batch_em
from training.logging import (
    save_training_result,
    save_testing_result,
)
from utils.seeder import set_seed
from utils.dataloader import create_dataloaders
from utils.logger import setup_logging
from utils.checkpoint import save_checkpoint
from utils.path import PROJECT_ROOT

# ============================================================================
# CONFIGURATION & DATA SOURCES
# ============================================================================

CONFIG = {
    # Experiment
    "seed": 1,
    "model_name": "8-5C3B",
    # Model Architecture
    "cnn_layers": 5,
    "rnn_layers": 3,
    # Training
    "epochs": 70,
    "batch_size": 16,
    "learning_rate": 2e-4,
    "grad_clip": 1.0,
    # Data
    "img_height": 48,
    "num_workers": 3,
}

BASE_SYNT_DIR = "dataset/word_nglegena_synthetic_20260130_155231"
BASE_REAL_DIR = "dataset/word_nglegena_handwritten_20260130_155805"
DATA_SOURCES = {
    "train": [
        {
            "csv": f"{BASE_SYNT_DIR}/label_train.csv",
            "img_dir": f"{BASE_SYNT_DIR}/image_train",
            "aug": get_augmentation_pipeline(prob=1.0, seed=CONFIG["seed"]),
            "prep": get_preprocessing_pipeline(img_height=CONFIG["img_height"]),
        },
    ],
    "val": [
        {
            "csv": f"{BASE_SYNT_DIR}/label_val.csv",
            "img_dir": f"{BASE_SYNT_DIR}/image_val",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=CONFIG["img_height"]),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_5.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_5",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=CONFIG["img_height"]),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_2.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_2",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=CONFIG["img_height"]),
        },
    ],
    "test": [
        {
            "csv": f"{BASE_REAL_DIR}/label_1.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_1",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=CONFIG["img_height"]),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_3.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_3",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=CONFIG["img_height"]),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_4.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_4",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=CONFIG["img_height"]),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_6.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_6",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=CONFIG["img_height"]),
        },
    ],
}


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def main():
    # ===== INITIAL SETUP ========================================
    # Setup directories
    model_dir = Path(PROJECT_ROOT) / "builds" / CONFIG["model_name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    if list(model_dir.iterdir()):
        raise RuntimeError(f"Model directory is not empty: {model_dir}")

    # Setup logging
    logger = setup_logging(model_dir / "training.log")
    logger.info(f"Starting experiment: {CONFIG['model_name']}")
    logger.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Reproducibility
    logger.info(f"Setting seed: {CONFIG['seed']}")
    generator = set_seed(CONFIG["seed"])

    # Build dataset
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = (
        create_dataloaders(
            DATA_SOURCES,
            CONFIG["batch_size"],
            CONFIG["num_workers"],
            device,
            generator,
        )
    )
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    # # ===== DATASET | OUTPUT VISUALIZATION EXAMPLE ===============
    # print(
    #     f"\nTotal Train: {len(train_ds)} | Total Val: {len(val_ds)} | Total Test: {len(test_ds)}"
    # )
    # batch = next(iter(train_loader))
    # print("\nBatch structure:")
    # for i, v in enumerate(batch):
    #     if hasattr(v, "shape"):
    #         print(f"Element {i}: shape = {v.shape}")
    #     else:
    #         print(f"Element {i}: {v}")
    # images, labels, label_lengths, input_lengths, filenames = batch
    # print("\nSample data:")
    # print("Filename:", filenames[0])
    # print("Label length:", label_lengths[0])
    # print("Input length:", input_lengths[0])
    # print("Images:", images[0])
    # return
    # # ===== DATASET | OUTPUT VISUALIZATION EXAMPLE ===============

    # Build model
    logger.info("Building model...")
    model = CNNBiLSTM(
        num_classes=NUM_CLASSES,
        cnn_layers=CONFIG["cnn_layers"],
        rnn_layers=CONFIG["rnn_layers"],
    ).to(device)
    model_info = model.get_model_info()
    logger.info(f"Model: {model_info}")
    # # ===== MODEL | OUTPUT VISUALIZATION EXAMPLE ===============
    # print("\n")
    # for k, v in model_info.items():
    #     print(f"{k:20}: {v}")
    # print(model)
    # return
    # # ===== MODEL | OUTPUT VISUALIZATION EXAMPLE ===============

    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-6,
    )

    # ===== MAIN TRAINING LOOP ========================================
    logger.info("Starting training...")
    epoch_logs = []
    train_start = datetime.now()
    train_start_ts = time.time()

    global_step = 0
    best_cer = float("inf")
    best_em = 0.0
    best_val_loss = float("inf")
    last_lr = CONFIG["learning_rate"]

    try:
        for epoch in range(CONFIG["epochs"]):
            # ===== Training ===============
            train_loss, num_steps = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                CONFIG["grad_clip"],
            )
            global_step += num_steps

            # ===== Validating ===============
            val_loss, all_preds, all_refs = validate_one_epoch(
                model, val_loader, criterion, device
            )

            # Metrics calc
            cer = batch_cer(all_preds, all_refs)
            em = batch_em(all_preds, all_refs)

            # Lr scheduler
            scheduler.step(cer)
            current_lr = scheduler.get_last_lr()[0]
            if current_lr != last_lr:
                logger.info(f"Learning rate adjusted: {current_lr}")
                last_lr = current_lr

            # Find best scores
            if cer < best_cer:
                best_cer = cer
            if em > best_em:
                best_em = em
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Log
            log_line = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "cer": float(cer),
                "em": float(em),
                "global_step": global_step,
            }
            epoch_logs.append(log_line)
            logger.info(
                f"Epoch {epoch+1:02d}/{CONFIG['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"CER: {cer:.4f} | "
                f"EM: {em:.4f}"
            )

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        CONFIG["epochs"],
        global_step,
        val_loss,
        cer,
        model_dir / "last_model.pth",
        logger,
    )

    # Training summary
    train_end = datetime.now()
    train_duration_sec = int(time.time() - train_start_ts)
    hours, remainder = divmod(train_duration_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Training finished in {hours}h {minutes}m {seconds}s")

    # Save training logs
    save_training_result(
        model_dir=str(model_dir),
        model_name=CONFIG["model_name"],
        model=model,
        config=CONFIG,
        device=device,
        cnn_layers=CONFIG["cnn_layers"],
        bilstm_layers=CONFIG["rnn_layers"],
        total_params=model_info["total_params"],
        trainable_params=model_info["trainable_params"],
        train_ds=train_ds,
        val_ds=val_ds,
        datasets=DATA_SOURCES,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        epoch_logs=epoch_logs,
        train_start_wall=train_start,
        train_end_wall=train_end,
        train_duration_sec=train_duration_sec,
        best_val_cer=best_cer,
        best_val_em=best_em,
        best_val_loss=best_val_loss,
    )

    # ===== Testing ===============
    logger.info("Running test evaluation...")
    all_results = {}

    model_path = model_dir / "last_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    (
        all_preds,
        all_refs,
        all_filenames,
        total_sample,
        final_cer_score,
        final_em_score,
        avg_fwd_ms,
        avg_dec_ms,
        peak_vram_res,
        peak_cpu_mb,
    ) = test_one_epoch(
        model,
        test_loader,
        device,
        decode_method=DecodeMethod.BEST_PATH,
        beam_width=0,
        verbose=False,
    )
    all_results = {
        "imgs": all_filenames,
        "preds": all_preds,
        "refs": all_refs,
    }

    # Save testing logs
    save_testing_result(
        model_name=CONFIG["model_name"],
        datasets=DATA_SOURCES,
        test_ds=test_ds,
        results=all_results,
        save_dir=str(model_dir),
        avg_fwd_ms=avg_fwd_ms,
        avg_dec_ms=avg_dec_ms,
        peak_vram_res=peak_vram_res,
        peak_cpu_mb=peak_cpu_mb,
    )

    # ===== FINAL DETAILED REPORT ========================================
    logger.info(f"===== PERFORMANCE REPORT =====")
    logger.info(f"Total Samples    : {total_sample}")
    logger.info(f"Final CER        : {final_cer_score:.4f}")
    logger.info(f"Final EM         : {final_em_score:.4f}")
    logger.info(f"===== LATENCY / TIME =====")
    logger.info(f"GPU Forward Pass : {avg_fwd_ms:.2f} ms/sample")
    logger.info(f"CPU Decoding     : {avg_dec_ms:.2f} ms/sample")
    logger.info(f"Total Latency    : {avg_fwd_ms + avg_dec_ms:.2f} ms/sample")
    logger.info(f"===== MEMORY ALLOCATION =====")
    logger.info(f"Peak GPU VRAM    : {peak_vram_res:.4f} MB (Reserved)")
    logger.info(f"Peak CPU RAM     : {peak_cpu_mb:.4f} MB")


if __name__ == "__main__":
    main()
