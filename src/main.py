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
from training.test import test_one_epoch
from training.train import train_one_epoch
from training.validate import validate_one_epoch
from metric.ocr_metrics import batch_cer, batch_em
from training.logging import (
    save_training_logs,
    save_training_plots,
    save_test_result,
    save_test_plots,
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
    "model_name": "model_name",
    "seed": 42,
    # Model Architecture
    "cnn_layers": 5,
    "rnn_layers": 2,
    # Training
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "grad_clip": 5.0,
    # Data
    "img_height": 48,
    "num_workers": 3,
    # Checkpointing
    "cer_eps": 1e-3,
    "loss_eps": 1e-4,
}


BASE_SYNT_DIR = "dataset/word_nglegena_synthetic_20260130_155231"
BASE_REAL_DIR = "dataset/word_nglegena_handwritten_20260130_155805"
DATA_SOURCES = {
    "train": [
        {
            "csv": f"{BASE_SYNT_DIR}/label_train.csv",
            "img_dir": f"{BASE_SYNT_DIR}/image_train",
            "aug": get_augmentation_pipeline(prob=1.0, seed=CONFIG["seed"]),
            "prep": get_preprocessing_pipeline(
                img_height=CONFIG["img_height"], enhance=False
            ),
        }
    ],
    "val": [
        {
            "csv": f"{BASE_SYNT_DIR}/label_val.csv",
            "img_dir": f"{BASE_SYNT_DIR}/image_val",
            "aug": None,
            "prep": get_preprocessing_pipeline(
                img_height=CONFIG["img_height"], enhance=False
            ),
        },
    ],
    "test": [
        {
            "csv": f"{BASE_REAL_DIR}/label_1.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_1",
            "aug": None,
            "prep": get_preprocessing_pipeline(
                img_height=CONFIG["img_height"], enhance=False
            ),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_2.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_2",
            "aug": None,
            "prep": get_preprocessing_pipeline(
                img_height=CONFIG["img_height"], enhance=False
            ),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_3.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_3",
            "aug": None,
            "prep": get_preprocessing_pipeline(
                img_height=CONFIG["img_height"], enhance=False
            ),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_4.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_4",
            "aug": None,
            "prep": get_preprocessing_pipeline(
                img_height=CONFIG["img_height"], enhance=False
            ),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_5.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_5",
            "aug": None,
            "prep": get_preprocessing_pipeline(
                img_height=CONFIG["img_height"], enhance=False
            ),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_6.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_6",
            "aug": None,
            "prep": get_preprocessing_pipeline(
                img_height=CONFIG["img_height"], enhance=False
            ),
        },
    ],
}


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def main():
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

    # Create dataloaders
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

    # Build model
    logger.info("Building model...")
    model = CNNBiLSTM(
        num_classes=NUM_CLASSES,
        cnn_layers=CONFIG["cnn_layers"],
        rnn_layers=CONFIG["rnn_layers"],
    ).to(device)

    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    model_info = model.get_model_info()
    logger.info(f"Model: {model_info}")

    # Training loop
    logger.info("Starting training...")
    epoch_logs = []
    train_start = datetime.now()
    train_start_ts = time.time()

    global_step = 0
    best_cer = float("inf")
    best_val_loss = float("inf")

    try:
        for epoch in range(CONFIG["epochs"]):
            # Train
            train_loss, num_steps = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                CONFIG["grad_clip"],
            )
            global_step += num_steps

            # Validate
            val_loss, all_preds, all_refs = validate_one_epoch(
                model, val_loader, criterion, device
            )

            # Metrics
            cer = batch_cer(all_preds, all_refs)
            em = batch_em(all_preds, all_refs)

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

            # Save best model
            is_best = False
            if cer < best_cer - CONFIG["cer_eps"]:
                is_best = True
            elif (
                abs(cer - best_cer) <= CONFIG["cer_eps"]
                and val_loss < best_val_loss - CONFIG["loss_eps"]
            ):
                is_best = True

            if is_best:
                best_cer = cer
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    val_loss,
                    cer,
                    model_dir / "best_model.pth",
                    logger,
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
    logger.info("Saving training logs...")
    save_training_logs(
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
        criterion=criterion,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        epoch_logs=epoch_logs,
        train_start_wall=train_start,
        train_end_wall=train_end,
        train_duration_sec=train_duration_sec,
        best_val_cer=best_cer,
        best_val_loss=best_val_loss,
    )
    save_training_plots(epoch_logs, str(model_dir), CONFIG)

    # Testing
    logger.info("Running test evaluation...")
    all_results = {}

    for model_name in ["best_model", "last_model"]:
        model_path = model_dir / f"{model_name}.pth"

        if not model_path.exists():
            logger.warning(f"{model_name} not found, skipping")
            continue

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        all_preds, all_refs, all_imgs = test_one_epoch(model, test_loader, device)
        all_results[model_name] = {
            "imgs": all_imgs,
            "preds": all_preds,
            "refs": all_refs,
        }

        test_cer = batch_cer(all_preds, all_refs)
        test_em = batch_em(all_preds, all_refs)
        logger.info(f"{model_name}: CER={test_cer:.4f}, EM={test_em:.4f}")

    save_test_result(
        CONFIG["model_name"],
        DATA_SOURCES,
        test_ds,
        all_results,
        str(model_dir),
        "test_results.json",
    )
    save_test_plots(all_results, str(model_dir))

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
