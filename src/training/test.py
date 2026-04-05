"""
Test/Inference Loop for CTC-based OCR.
"""

from typing import List, Tuple
from enum import Enum
import gc
import time
import tracemalloc

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vocabulary import IDX2CHAR
from decoding.ctc_decoder import (
    best_path_decode,
    beam_search_decode,
    decode_targets,
)
from metric.ocr_metrics import cer, em, batch_cer, batch_em


class DecodeMethod(Enum):
    BEST_PATH = "best_path"
    BEAM_SEARCH = "beam_search"


@torch.inference_mode()
def test_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    decode_method: DecodeMethod = DecodeMethod.BEST_PATH,
    beam_width: int = 5,
    verbose: bool = False,
) -> Tuple[List[str], List[str], List[str], float, float, float, float, float, float]:
    """
    Menjalankan inferensi dengan pemisahan metrik performa CPU dan GPU secara eksplisit.
    """
    model.eval()

    # ===== PRE-FLUSH & WARMUP ========================================
    # Memory cleaning & CUDA kernel warmup
    tracemalloc.start()
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        try:
            warmup_batch = next(iter(loader))[0].to(device)
            for _ in range(5):
                _ = model(warmup_batch)
            torch.cuda.synchronize()
            del warmup_batch
        except StopIteration:
            pass
        torch.cuda.reset_peak_memory_stats(device)

    total_forward_time = 0.0
    total_decode_time = 0.0
    all_preds, all_refs, all_filenames = [], [], []

    # ===== INFERENCE LOOP ========================================
    for batch in tqdm(loader, desc=f"Testing", leave=False):
        images, labels, label_lens, _, filenames = batch
        images = images.to(device, non_blocking=True)

        # ===== FORWARD PASS (GPU & VRAM) ===============
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_fwd = time.perf_counter()

        logits = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_forward_time += time.perf_counter() - start_fwd

        # ===== DECODING (CPU & RAM) ===============
        start_dec = time.perf_counter()

        if decode_method is DecodeMethod.BEST_PATH:
            pred_indices = logits.argmax(dim=2).cpu()
            batch_preds = [best_path_decode(seq, IDX2CHAR) for seq in pred_indices]
        else:
            probs = torch.softmax(logits, dim=2).cpu()
            batch_preds = [
                beam_search_decode(seq, IDX2CHAR, beam_width=beam_width)[0][0]
                for seq in probs
            ]

        total_decode_time += time.perf_counter() - start_dec

        # Ground truth & Collecting results
        batch_refs = decode_targets(labels, label_lens, IDX2CHAR)
        all_preds.extend(batch_preds)
        all_refs.extend(batch_refs)
        all_filenames.extend(filenames)

        if verbose:
            for fname, pred_text, true_text in zip(filenames, batch_preds, batch_refs):
                print(
                    f"[{fname}] | CER: {cer(pred=pred_text, ref=true_text):.2f} | EM: {em(pred=pred_text, ref=true_text):.0f} ====="
                )
                print(f"GT: {true_text}")
                print(f"PRED: {pred_text}")

    # ===== STATISTIC CALCULATION ========================================
    total_sample = max(len(all_preds), 1)

    # Latency (ms/sample)
    avg_fwd_ms = (total_forward_time / total_sample) * 1000
    avg_dec_ms = (total_decode_time / total_sample) * 1000

    # GPU memory (VRAM) (Physical Allocation)
    if device.type == "cuda":
        peak_vram_res = torch.cuda.max_memory_reserved(device) / (1024**2)
    else:
        peak_vram_res = 0.0

    # CPU memory (RAM) Get peak memory usage
    _, peak_cpu_ram = tracemalloc.get_traced_memory()
    peak_cpu_mb = peak_cpu_ram / (1024**2)
    tracemalloc.stop()

    final_cer_score = batch_cer(preds=all_preds, refs=all_refs)
    final_em_score = batch_em(preds=all_preds, refs=all_refs)

    return (
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
    )
