"""
Training Logging and Visualization.
"""

import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from itertools import zip_longest
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset

from metric.ocr_metrics import batch_cer, batch_em, cer, em, levenshtein


# ============================================================================
# SYSTEM & ENVIRONMENT INFO
# ============================================================================


def get_environment_info() -> Dict[str, str]:
    """Capture complete environment information for reproducibility."""

    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A"
        ),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # GPU info
    if torch.cuda.is_available():
        env_info["num_gpus"] = torch.cuda.device_count()
        env_info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        env_info["gpu_memory"] = [
            f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
            for i in range(torch.cuda.device_count())
        ]
    else:
        env_info["num_gpus"] = 0
        env_info["gpu_names"] = []
        env_info["gpu_memory"] = []

    return env_info


def get_optimizer_config(optimizer) -> Dict:
    """Extract complete optimizer configuration."""

    config = {
        "class": optimizer.__class__.__name__,
    }

    # Get all hyperparameters from param_groups
    if optimizer.param_groups:
        param_group = optimizer.param_groups[0]
        for key, value in param_group.items():
            if key != "params":  # Skip the actual parameters
                config[key] = value

    return config


def get_criterion_config(criterion) -> Dict:
    """Extract complete criterion configuration."""

    config = {
        "class": criterion.__class__.__name__,
    }

    # For CTCLoss
    if hasattr(criterion, "blank"):
        config["blank"] = criterion.blank
    if hasattr(criterion, "zero_infinity"):
        config["zero_infinity"] = criterion.zero_infinity
    if hasattr(criterion, "reduction"):
        config["reduction"] = criterion.reduction

    return config


def get_model_architecture(model) -> Dict:
    """Extract complete model architecture details."""

    arch_info = {}

    # Get model info if available
    if hasattr(model, "get_model_info"):
        arch_info.update(model.get_model_info())

    # Layer-by-layer details
    if hasattr(model, "cnn"):
        arch_info["cnn_structure"] = str(model.cnn)

    if hasattr(model, "rnn"):
        rnn = model.rnn
        arch_info["rnn_details"] = {
            "input_size": rnn.input_size,
            "hidden_size": rnn.hidden_size,
            "num_layers": rnn.num_layers,
            "bidirectional": rnn.bidirectional,
            "batch_first": rnn.batch_first,
            "dropout": rnn.dropout if rnn.num_layers > 1 else 0.0,
        }

    if hasattr(model, "fc"):
        fc = model.fc
        arch_info["fc_details"] = {
            "in_features": fc.in_features,
            "out_features": fc.out_features,
        }

    return arch_info


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def format_duration(sec: int) -> str:
    """Format seconds to HH:MM:SS."""

    h, remainder = divmod(sec, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistical summary of metric values."""

    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def analyze_errors(
    preds: List[str], refs: List[str], top_k: int = 10
) -> Dict[str, Any]:
    """Analyze common error patterns."""

    substitutions = []
    insertions = []
    deletions = []

    for pred, ref in zip(preds, refs):
        # Menggunakan zip_longest agar karakter yang hilang/lebih tetap terhitung
        for p, r in zip_longest(pred, ref, fillvalue=None):
            if p is None:  # Karakter ada di ref tapi tidak di pred
                deletions.append(r)
            elif r is None:  # Karakter ada di pred tapi tidak di ref
                insertions.append(p)
            elif p != r:
                substitutions.append(f"{r}→{p}")

    sub_counter = Counter(substitutions)
    ins_counter = Counter(insertions)
    del_counter = Counter(deletions)

    return {
        "substitutions": dict(sub_counter.most_common(top_k)),
        "insertions": dict(ins_counter.most_common(top_k)),
        "deletions": dict(del_counter.most_common(top_k)),
        "total_errors": len(substitutions) + len(insertions) + len(deletions),
    }


def build_components(
    split_sources: List[Dict], split_dataset: ConcatDataset
) -> List[Dict]:
    """Build dataset component information."""

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


def write_stats(f, name, stats):
    f.write(f"{name}\n")
    f.write(f"  Mean   : {stats['mean']:.6f}\n")
    f.write(f"  Std    : {stats['std']:.6f}\n")
    f.write(f"  Min    : {stats['min']:.6f}\n")
    f.write(f"  Max    : {stats['max']:.6f}\n")
    f.write(f"  Median : {stats['median']:.6f}\n")
    f.write(f"  Q25    : {stats['q25']:.6f}\n")
    f.write(f"  Q75    : {stats['q75']:.6f}\n\n")


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def save_training_plots(epoch_logs: List[Dict], model_dir: str) -> None:
    """Generate training visualization plots."""

    model_dir = Path(model_dir)

    epochs = [e["epoch"] for e in epoch_logs]
    train_loss = [e["train_loss"] for e in epoch_logs]
    val_losses = [e["val_loss"] for e in epoch_logs]
    cer_vals = [e["cer"] for e in epoch_logs]
    em_vals = [e["em"] for e in epoch_logs]

    # 1. ===== Loss Plot ===============
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2, alpha=0.8)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=2, alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("CTC Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_dir / "train_plot_loss.png", dpi=300)
    plt.close()

    # 2. ===== Metrics Plot ===============
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_cer = "tab:red"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Character Error Rate (CER)", color=color_cer, fontsize=12)
    ln1 = ax1.plot(
        epochs, cer_vals, color=color_cer, linewidth=2, alpha=0.8, label="CER"
    )
    ax1.tick_params(axis="y", labelcolor=color_cer)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_em = "tab:green"
    ax2.set_ylabel("Exact Match Accuracy (EM)", color=color_em, fontsize=12)
    ln2 = ax2.plot(epochs, em_vals, color=color_em, linewidth=2, alpha=0.8, label="EM")
    ax2.tick_params(axis="y", labelcolor=color_em)
    ax2.set_ylim(-0.05, 1.05)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right", fontsize=10, frameon=True, shadow=True)

    plt.title(
        "Character Error Rate & Exact Match Accuracy", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(model_dir / "train_plot_metrics.png", dpi=300)
    plt.close()


def save_testing_plots(results: Dict, model_dir: str) -> None:
    """Generate test evaluation plots."""

    model_dir = Path(model_dir)

    preds = results["preds"]
    refs = results["refs"]

    cer_vals = [cer(p, r) for p, r in zip(preds, refs)]
    ref_lens = [len(r) for r in refs]

    # 1. ===== CER Distribution ===============
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cer_vals, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(
        np.mean(cer_vals),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(cer_vals):.4f}",
    )
    ax.axvline(
        np.median(cer_vals),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(cer_vals):.4f}",
    )
    ax.set_xlabel("Character Error Rate", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"CER Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_dir / f"test_plot_cer_dist.png", dpi=300)
    plt.close()

    # 2. ===== Length vs CER ===============
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ref_lens, cer_vals, alpha=0.5, s=30)
    ax.set_xlabel("Reference Sequence Length", fontsize=12)
    ax.set_ylabel("Character Error Rate", fontsize=12)
    ax.set_title(f"Sequence Length vs CER", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_dir / f"test_plot_length_vs_cer.png", dpi=300)
    plt.close()


# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================


def save_training_result(
    *,
    model_dir: str,
    model_name: str,
    model,
    config: Dict,
    device,
    total_params: int,
    trainable_params: int,
    cnn_layers: int,
    bilstm_layers: int,
    datasets: Dict,
    train_ds: ConcatDataset,
    val_ds: ConcatDataset,
    optimizer,
    criterion,
    epochs: int,
    batch_size: int,
    epoch_logs: List[Dict],
    train_start_wall,
    train_end_wall,
    train_duration_sec: int,
    best_val_cer: float,
    best_val_em: float,
    best_val_loss: float,
) -> Tuple[str, str]:
    """
    Save training logs with COMPLETE reproducibility info.

    Returns:
        Tuple of (json_path, txt_path)
    """

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    LINE = "=" * 80
    SUBLINE = "-" * 80

    # Metrics statistics
    train_losses = [e["train_loss"] for e in epoch_logs]
    val_losses = [e["val_loss"] for e in epoch_logs]
    cer_vals = [e["cer"] for e in epoch_logs]
    em_vals = [e["em"] for e in epoch_logs]

    train_loss_stats = compute_statistics(train_losses)
    val_loss_stats = compute_statistics(val_losses)
    cer_stats = compute_statistics(cer_vals)
    em_stats = compute_statistics(em_vals)

    # Get environment info
    env_info = get_environment_info()

    # Get complete configs
    optimizer_config = get_optimizer_config(optimizer)
    criterion_config = get_criterion_config(criterion)
    model_arch = get_model_architecture(model)

    # Build log data
    log_data = {
        "environment": env_info,
        "hyperparameters": config,
        "model": {
            "name": model_name,
            "cnn_layers": cnn_layers,
            "bilstm_layers": bilstm_layers,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "architecture": model_arch,
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
        "training": {
            "optimizer": optimizer_config,
            "criterion": criterion_config,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": str(device),
        },
        "summary": {
            "start_time": train_start_wall.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": train_end_wall.strftime("%Y-%m-%d %H:%M:%S"),
            "training_duration_sec": train_duration_sec,
            "training_duration_formatted": format_duration(train_duration_sec),
            "best_val_cer": best_val_cer,
            "best_val_em": best_val_em,
            "best_val_loss": best_val_loss,
        },
        "statistics": {
            "train_loss": train_loss_stats,
            "val_loss": val_loss_stats,
            "cer": cer_stats,
            "em": em_stats,
        },
        "epochs": epoch_logs,
    }

    # 1. ===== Save JSON ===============
    json_path = model_dir / "train_log.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # 2. ===== Save TXT ===============
    txt_path = model_dir / "train_log.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(LINE + "\n")
        f.write("TRAINING LOG\n")
        f.write(LINE + "\n\n")

        # Environment
        f.write("ENVIRONMENT INFORMATION\n" + SUBLINE + "\n")
        f.write(f"Python Version     : {env_info['python_version']}\n")
        f.write(f"PyTorch Version    : {env_info['pytorch_version']}\n")
        f.write(f"CUDA Available     : {env_info['cuda_available']}\n")
        f.write(f"CUDA Version       : {env_info['cuda_version']}\n")
        f.write(f"cuDNN Version      : {env_info['cudnn_version']}\n")
        f.write(f"Platform           : {env_info['platform']}\n")
        f.write(f"System             : {env_info['system']}\n")
        f.write(f"Machine            : {env_info['machine']}\n")
        f.write(f"Num GPUs           : {env_info['num_gpus']}\n")
        if env_info["gpu_names"]:
            for i, (name, mem) in enumerate(
                zip(env_info["gpu_names"], env_info["gpu_memory"])
            ):
                f.write(f"GPU {i}             : {name} ({mem})\n")
        f.write("\n")

        # Hyperparameters
        f.write("HYPERPARAMETERS\n" + SUBLINE + "\n")
        for key, value in config.items():
            f.write(f"{key:<20} : {value}\n")
        f.write("\n")

        # Model Info
        f.write("MODEL ARCHITECTURE\n" + SUBLINE + "\n")
        f.write(f"Model Name         : {model_name}\n")
        f.write(f"CNN Layers         : {cnn_layers}\n")
        f.write(f"BiLSTM Layers      : {bilstm_layers}\n")
        f.write(f"Total Parameters   : {total_params:,}\n")
        f.write(f"Trainable Params   : {trainable_params:,}\n")

        if "cnn_structure" in model_arch:
            cnn = model_arch["cnn_structure"]
            f.write(f"\nCNN Configuration  :\n")
            f.write(f"  {cnn}\n")

        if "rnn_details" in model_arch:
            rnn = model_arch["rnn_details"]
            f.write(f"\nRNN Configuration  :\n")
            for key, value in rnn.items():
                f.write(f"  {key:<18} : {value}\n")

        if "fc_details" in model_arch:
            fc = model_arch["fc_details"]
            f.write(f"\nFC Layer           :\n")
            for key, value in fc.items():
                f.write(f"  {key:<18} : {value}\n")
        f.write("\n")

        # Dataset
        f.write("DATASET INFORMATION\n" + SUBLINE + "\n")
        for split_name, split_key in [("TRAIN", "train"), ("VALIDATION", "val")]:
            ds = train_ds if split_key == "train" else val_ds
            f.write(f"{split_name}\n")
            f.write(f"  Total Samples    : {len(ds):,}\n")
            f.write(f"  Components       :\n")
            for i, src in enumerate(datasets[split_key], 1):
                f.write(f"    [{i}]\n")
                f.write(f"      CSV          : {os.path.basename(src['csv'])}\n")
                f.write(f"      IMG DIR      : {os.path.basename(src['img_dir'])}\n")
                f.write(f"      Augmentation : {'Yes' if src.get('aug') else 'No'}\n")
            f.write("\n")

        # Training Config
        f.write("TRAINING CONFIGURATION\n" + SUBLINE + "\n")
        f.write(f"Device             : {device}\n")
        f.write(f"Epochs             : {epochs}\n")
        f.write(f"Batch Size         : {batch_size}\n\n")

        f.write(f"Optimizer          : {optimizer_config['class']}\n")
        for key, value in optimizer_config.items():
            if key != "class":
                f.write(f"  {key:<18} : {value}\n")
        f.write("\n")

        f.write(f"Criterion          : {criterion_config['class']}\n")
        for key, value in criterion_config.items():
            if key != "class":
                f.write(f"  {key:<18} : {value}\n")
        f.write("\n")

        # Summary
        f.write("TRAINING SUMMARY\n" + SUBLINE + "\n")
        f.write(
            f"Start Time         : {train_start_wall.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(
            f"End Time           : {train_end_wall.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Duration           : {format_duration(train_duration_sec)}\n")
        f.write(f"Best Val CER       : {best_val_cer:.6f}\n")
        f.write(f"Best Val EM        : {best_val_em:.6f}\n")
        f.write(f"Best Val Loss      : {best_val_loss:.6f}\n\n")

        # Statistics
        f.write("METRIC STATISTICS\n" + SUBLINE + "\n")

        write_stats(f, "Train Loss", train_loss_stats)
        write_stats(f, "Val Loss", val_loss_stats)
        write_stats(f, "CER", cer_stats)
        write_stats(f, "EM", em_stats)

        # Epoch Logs
        f.write("EPOCH-BY-EPOCH LOGS\n" + SUBLINE + "\n")
        f.write(
            f"{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<14}{'CER':<12}{'EM':<12}{'Best':<6}\n"
        )
        f.write(SUBLINE + "\n")

        for e in epoch_logs:
            f.write(
                f"{e['epoch']:<8}"
                f"{e['train_loss']:<14.6f}"
                f"{e['val_loss']:<14.6f}"
                f"{e['cer']:<12.6f}"
                f"{e['em']:<12.6f}\n"
            )
        f.write(SUBLINE + "\n")

    # 3. ===== Generate plots ===============
    save_training_plots(epoch_logs, str(model_dir))

    return str(json_path), str(txt_path)


def save_testing_result(
    model_name: str,
    datasets: Dict,
    test_ds: ConcatDataset,
    results: Dict,
    save_dir: str,
    avg_fwd_ms: float = 0.0,
    avg_dec_ms: float = 0.0,
    peak_vram_res: float = 0.0,
    peak_cpu_mb: float = 0.0,
) -> Tuple[str, str]:
    """
    Save test results with error analysis.

    Returns:
        Tuple of (json_path, txt_path)
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    LINE = "=" * 80
    SUBLINE = "-" * 80

    preds = results["preds"]
    refs = results["refs"]
    imgs = results.get("imgs", [f"img_{i}" for i in range(len(preds))])

    # Compute metrics
    batch_cer_val = batch_cer(preds=preds, refs=refs)
    batch_em_val = batch_em(preds=preds, refs=refs)

    # Per-sample metrics
    per_sample_cer = [cer(p, r) for p, r in zip(preds, refs)]
    per_sample_em = [em(p, r) for p, r in zip(preds, refs)]

    # Statistics
    cer_stats = compute_statistics(per_sample_cer)
    em_stats = compute_statistics(per_sample_em)

    # Error analysis
    errors = analyze_errors(preds, refs, top_k=10)

    summary = {
        "cer": float(batch_cer_val),
        "em": float(batch_em_val),
        "num_samples": len(preds),
        "num_correct": sum(per_sample_em),
        "num_errors": len(preds) - sum(per_sample_em),
        "cer_statistics": cer_stats,
        "em_statistics": em_stats,
        "avg_fwd_ms": avg_fwd_ms,
        "avg_dec_ms": avg_dec_ms,
        "peak_vram_res": peak_vram_res,
        "peak_cpu_mb": peak_cpu_mb,
    }

    # Detailed results
    details = []
    for img, p, r, c, e in zip(imgs, preds, refs, per_sample_cer, per_sample_em):
        detail = {
            "img": img,
            "gt": r,
            "pred": p,
            "cer": float(c),
            "em": float(e),
            "edit_distance": levenshtein(p, r),
            "gt_length": len(r),
            "pred_length": len(p),
        }
        details.append(detail)

    # Build final log
    final_log = {
        "model_name": model_name,
        "dataset": {
            "test": {
                "total_samples": len(test_ds),
                "components": build_components(datasets["test"], test_ds),
            },
        },
        "summary": summary,
        "error_analysis": errors,
        "details": details,
    }

    # 1. ===== Save JSON ===============
    json_path = save_dir / "test_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_log, f, indent=2, ensure_ascii=False)

    # 2. ===== Save TXT ===============
    txt_path = save_dir / "test_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(LINE + "\n")
        f.write("TEST EVALUATION REPORT\n")
        f.write(LINE + "\n\n")

        f.write(f"Model Name : {model_name}\n\n")

        # Dataset Info
        f.write("TEST DATASET\n" + SUBLINE + "\n")
        f.write(f"Total Samples      : {len(test_ds):,}\n")
        f.write(f"Components         :\n")
        for i, src in enumerate(datasets["test"], 1):
            f.write(f"  [{i}]\n")
            f.write(f"    CSV            : {os.path.basename(src['csv'])}\n")
            f.write(f"    IMG DIR        : {os.path.basename(src['img_dir'])}\n")
            f.write(f"    Augmentation   : {'Yes' if src.get('aug') else 'No'}\n")
        f.write("\n")

        # Overall Summary
        f.write("OVERALL METRICS\n" + SUBLINE + "\n")
        f.write(f"  Num Samples      : {summary['num_samples']:,}\n")
        f.write(f"  Correct          : {summary['num_correct']}\n")
        f.write(f"  Errors           : {summary['num_errors']}\n")
        f.write(f"  CER (micro-avg)  : {summary['cer']:.6f}\n")
        f.write(f"  EM (accuracy)    : {summary['em']:.6f}\n")
        f.write(f"  Avg Forward Time : {summary['avg_fwd_ms']:.2f} ms\n")
        f.write(f"  Avg Decode Time  : {summary['avg_dec_ms']:.2f} ms\n")
        f.write(f"  Peak VRAM Usage  : {summary['peak_vram_res']:.2f} MB\n")
        f.write(f"  Peak CPU Usage   : {summary['peak_cpu_mb']:.2f} MB\n\n")

        # Overall Statistics
        write_stats(f, "CER Statistics", summary["cer_statistics"])
        write_stats(f, "EM Statistics", summary["em_statistics"])

        # Error Analysis
        f.write("ERROR ANALYSIS\n" + SUBLINE + "\n")
        f.write(f"  Total Errors     : {errors['total_errors']}\n\n")
        if errors["substitutions"]:
            f.write(f"  Top Substitutions:\n")
            for sub, count in errors["substitutions"].items():
                f.write(f"    {sub:<15} : {count:>5}\n")
            f.write("\n")

        if errors["insertions"]:
            f.write(f"  Top Insertions   :\n")
            for ins, count in errors["insertions"].items():
                f.write(f"    '{ins}'           : {count:>5}\n")
            f.write("\n")

        if errors["deletions"]:
            f.write(f"  Top Deletions    :\n")
            for dele, count in errors["deletions"].items():
                f.write(f"    '{dele}'           : {count:>5}\n")
            f.write("\n")

        # Detailed Results
        f.write("DETAILED PER-SAMPLE RESULTS\n" + SUBLINE + "\n")
        f.write(
            f"{'Image':<20}{'CER':<10}{'EM':<6}{'GT Len':<8}{'Pred Len':<10}GT => PRED\n"
        )
        f.write(SUBLINE + "\n")

        for d in details:
            img_str = str(d["img"])[:19]
            f.write(
                f"{img_str:<20}"
                f"{d['cer']:<10.4f}"
                f"{d['em']:<6.0f}"
                f"{d['gt_length']:<8}"
                f"{d['pred_length']:<10}"
                f"{d['gt']} => {d['pred'] or '-'}\n"
            )
        f.write(SUBLINE + "\n")

    # 3. ===== Generate plots ===============
    save_testing_plots(results, str(save_dir))

    return str(json_path), str(txt_path)
