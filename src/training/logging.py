import os
import json
import matplotlib.pyplot as plt

from metric.ocr_metrics import batch_cer, batch_em, cer, em


def format_duration(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def save_training_logs(
    *,
    model_dir,
    model_name,
    device,
    datasets,
    train_ds,
    val_ds,
    optimizer,
    scheduler,
    criterion,
    epochs,
    early_stop_patience,
    batch_size,
    epoch_logs,
    train_start_wall,
    train_end_wall,
    train_duration_sec,
    early_stopped,
    early_stopped_at,
    best_val_cer,
    best_val_loss,
):
    def build_components(split_sources, split_dataset):
        components = []
        for src, ds in zip(split_sources, split_dataset.datasets):
            components.append(
                {
                    "csv": os.path.basename(src["csv"]),
                    "img_dir": os.path.basename(src["img_dir"]),
                    "num_samples": len(ds),
                    "augmentation": src.get("aug") is not None,
                }
            )
        return components

    log_data = {
        "meta": {
            "model_name": model_name,
            "device": str(device),
        },
        "dataset": {
            "train": {
                "total_samples": len(train_ds),
                "components": build_components(datasets["train"], train_ds),
            },
            "val": {
                "total_samples": len(val_ds),
                "components": build_components(datasets["val"], val_ds),
            },
        },
        "config": {
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__ if scheduler else None,
            "criterion": criterion.__class__.__name__,
            "epochs": epochs,
            "early_stop_patience": early_stop_patience,
            "batch_size": batch_size,
        },
        "summary": {
            "start_time": train_start_wall.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": train_end_wall.strftime("%Y-%m-%d %H:%M:%S"),
            "training_duration_sec": train_duration_sec,
            "early_stopped": early_stopped,
            "early_stopped_at": early_stopped_at,
            "best_val_cer": float(best_val_cer),
            "best_val_loss": float(best_val_loss),
        },
        "epochs": epoch_logs,
    }

    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(model_dir, "train_log.json")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    return log_path


def save_training_plot(epoch_logs, model_dir):
    x = [e["epoch"] for e in epoch_logs]
    train_loss = [e["train_loss"] for e in epoch_logs]
    val_loss = [e["val_loss"] for e in epoch_logs]
    cer = [e["cer"] for e in epoch_logs]
    em = [e["em"] for e in epoch_logs]

    # --- Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(x, train_loss, label="Train Loss")
    plt.plot(x, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CTC Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "train_plot_loss.png"))
    plt.close()

    # --- CER ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Character Error Rate (CER)", color="red")
    ln1 = ax1.plot(x, cer, label="CER", color="red", linewidth=1.5, alpha=0.7)
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Exact Match Accuracy (EM)", color="green")
    ln2 = ax2.plot(x, em, label="EM", color="green", linewidth=1.5, alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(-0.05, 1.05)
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right", frameon=True, shadow=True)
    plt.title("Character Error Rate & Exact Match Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "train_plot_metrics.png"))
    plt.close()


def save_test_result(model_name, results_dict, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, filename)

    summary = {}
    details = {}

    for model_type, data in results_dict.items():
        preds = data["preds"]
        refs = data["refs"]
        imgs = data.get("imgs", [f"img_{i}" for i in range(len(preds))])

        batch_cer_val = batch_cer(preds, refs)
        batch_em_val = batch_em(preds, refs)
        summary[model_type] = {
            "cer": float(batch_cer_val),
            "em": float(batch_em_val),
            "num_samples": len(preds),
        }

        model_details = []
        for i, p, r in zip(imgs, preds, refs):
            sample_cer = cer(p, r)
            sample_em = em(p, r)
            model_details.append(
                {
                    "img": i,
                    "gt": r,
                    "pred": p,
                    "cer": float(sample_cer),
                    "em": float(sample_em),
                }
            )
        details[model_type] = model_details

    final_log = {"model_name": model_name, "summary": summary, "details": details}

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(final_log, f, indent=2, ensure_ascii=False)

    return log_path
