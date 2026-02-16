import os
import json
import matplotlib.pyplot as plt

from metric.ocr_metrics import batch_cer, batch_em, cer, em


def format_duration(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


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


def save_training_plot(epoch_logs, model_dir):
    x = [e["epoch"] for e in epoch_logs]
    train_loss = [e["train_loss"] for e in epoch_logs]
    val_loss = [e["val_loss"] for e in epoch_logs]
    cer = [e["cer"] for e in epoch_logs]
    em = [e["em"] for e in epoch_logs]

    # -------- Loss --------
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

    # -------- CER --------
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


def save_training_logs(
    *,
    model_dir,
    model_name,
    device,
    total_params,
    trainable_params,
    cnn_layers,
    bilstm_layers,
    datasets,
    train_ds,
    val_ds,
    optimizer,
    criterion,
    epochs,
    batch_size,
    epoch_logs,
    train_start_wall,
    train_end_wall,
    train_duration_sec,
    best_val_cer,
    best_val_loss,
):

    LINE = "=" * 80
    SUBLINE = "-" * 80

    log_data = {
        "meta": {
            "model_name": model_name,
            "cnn_layers": cnn_layers,
            "bilstm_layers": bilstm_layers,
            "total_params": total_params,
            "trainable_params": trainable_params,
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
            "criterion": criterion.__class__.__name__,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "summary": {
            "start_time": train_start_wall.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": train_end_wall.strftime("%Y-%m-%d %H:%M:%S"),
            "training_duration_sec": train_duration_sec,
            "best_val_cer": float(best_val_cer),
            "best_val_loss": float(best_val_loss),
        },
        "epochs": epoch_logs,
    }

    os.makedirs(model_dir, exist_ok=True)
    json_path = os.path.join(model_dir, "train_log.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(model_dir, "train_log.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        # -------- Meta --------
        f.write(LINE + "\nTRAINING SUMMARY\n" + LINE + "\n\n")

        f.write("MODEL INFO\n" + SUBLINE + "\n")
        f.write(f"Model Name       : {model_name}\n")
        f.write(f"CNN Layers       : {cnn_layers}\n")
        f.write(f"BiLSTM Layers    : {bilstm_layers}\n")
        f.write(f"Total Params     : {total_params:,}\n")
        f.write(f"Trainable Params : {trainable_params:,}\n")
        f.write(f"Device           : {device}\n\n")

        # -------- Dataset --------
        f.write("DATASET\n" + SUBLINE + "\n")

        def write_split(name, ds, config):
            f.write(f"{name.upper()}\n")
            f.write(f"  Total Samples : {len(ds)}\n")
            f.write(f"  Components    :\n")

            for i, src in enumerate(config, 1):
                f.write(f"    [{i}]\n")
                f.write(f"      CSV         : {os.path.basename(src['csv'])}\n")
                f.write(f"      IMG DIR     : {os.path.basename(src['img_dir'])}\n")
                f.write(f"      Augmentation: {'Yes' if src.get('aug') else 'No'}\n")
            f.write("\n")

        write_split("train", train_ds, datasets["train"])
        write_split("val", val_ds, datasets["val"])

        # -------- Config --------
        f.write("TRAINING CONFIG\n" + SUBLINE + "\n")
        f.write(f"Optimizer         : {optimizer.__class__.__name__}\n")
        f.write(f"Criterion         : {criterion.__class__.__name__}\n")
        f.write(f"Epochs            : {epochs}\n")
        f.write(f"Batch Size        : {batch_size}\n\n")

        # -------- Summary --------
        f.write("RESULT SUMMARY\n" + SUBLINE + "\n")
        f.write(f"Start Time     : {train_start_wall.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time       : {train_end_wall.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration (sec) : {train_duration_sec:.2f}\n")
        f.write(f"Best Val CER   : {best_val_cer:.6f}\n")
        f.write(f"Best Val Loss  : {best_val_loss:.6f}\n\n")

        # -------- Epoch --------
        f.write("EPOCH LOGS\n" + SUBLINE + "\n")

        C1, C2, C3, C4, C5 = 8, 14, 14, 12, 12

        f.write(
            f"{'EPOCH':<{C1}}"
            f"{'TRAIN_LOSS':<{C2}}"
            f"{'VAL_LOSS':<{C3}}"
            f"{'CER':<{C4}}"
            f"{'EM':<{C5}}\n"
            f"{SUBLINE}\n"
        )

        for e in epoch_logs:
            f.write(
                f"{e['epoch']:<{C1}}"
                f"{e['train_loss']:<{C2}.6f}"
                f"{e['val_loss']:<{C3}.6f}"
                f"{e['cer']:<{C4}.6f}"
                f"{e['em']:<{C5}.6f}\n"
            )
        f.write(SUBLINE + "\n")

    return json_path, txt_path


def save_test_result(model_name, datasets, test_ds, results_dict, save_dir, filename):

    LINE = "=" * 80
    SUBLINE = "-" * 80

    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, filename)
    txt_path = os.path.join(save_dir, os.path.splitext(filename)[0] + ".txt")

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
            model_details.append(
                {
                    "img": i,
                    "gt": r,
                    "pred": p,
                    "cer": float(cer(p, r)),
                    "em": float(em(p, r)),
                }
            )
        details[model_type] = model_details

    final_log = {
        "model_name": model_name,
        "dataset": {
            "test": {
                "total_samples": len(test_ds),
                "components": build_components(datasets["test"], test_ds),
            },
        },
        "summary": summary,
        "details": details,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_log, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:

        f.write(LINE + "\nTEST RESULT SUMMARY\n" + LINE + "\n\n")
        f.write(f"Model Name : {model_name}\n\n")

        # -------- Dataset --------
        f.write("DATASET\n" + SUBLINE + "\n")

        def write_split(name, ds, config):
            f.write(f"{name.upper()}\n")
            f.write(f"  Total Samples : {len(ds)}\n")
            f.write(f"  Components    :\n")

            for i, src in enumerate(config, 1):
                f.write(f"    [{i}]\n")
                f.write(f"      CSV         : {os.path.basename(src['csv'])}\n")
                f.write(f"      IMG DIR     : {os.path.basename(src['img_dir'])}\n")
                f.write(f"      Augmentation: {'Yes' if src.get('aug') else 'No'}\n")
            f.write("\n")

        write_split("Test", test_ds, datasets["test"])

        # -------- Summary --------
        f.write("OVERALL METRICS\n" + SUBLINE + "\n")

        for model_type, s in summary.items():
            f.write(f"Model Variant : {model_type}\n")
            f.write(f"  Num Samples : {s['num_samples']}\n")
            f.write(f"  CER         : {s['cer']:.6f}\n")
            f.write(f"  EM          : {s['em']:.6f}\n")

        # -------- Detail --------
        C1, C2, C3 = 19, 8, 7

        f.write("\nDETAILED PER-SAMPLE RESULTS\n" + SUBLINE + "\n")

        for model_type, model_details in details.items():
            f.write(f"\nModel Variant: {model_type}\n")
            f.write(
                f"{'IMG':<{C1}}"
                f"{'CER':<{C2}}"
                f"{'EM':<{C3}}"
                f"GT => PRED\n"
                f"{SUBLINE}\n"
            )

            for d in model_details:
                f.write(
                    f"{str(d['img'])[:C1]:<{C1}}"
                    f"{d['cer']:<{C2}.2f}"
                    f"{d['em']:<{C3}.1f}"
                    f"{d['gt']} => {d['pred'] or '-'}\n"
                )
            f.write(SUBLINE + "\n")

    return json_path, txt_path
