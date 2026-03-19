"""
Test/Inference Loop for CTC-based OCR.
"""

from typing import List, Tuple
from enum import Enum

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vocabulary import IDX2CHAR
from decoding.ctc_decoder import (
    best_path_decode,
    beam_search_decode,
    decode_targets,
)


class DecodeMethod(Enum):
    BEST_PATH = "best_path"
    BEAM_SEARCH = "beam_search"


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    decode_method: DecodeMethod = DecodeMethod.BEST_PATH,
    beam_width: int = 5,
    verbose: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Run inference on test set (no loss computation).

    Args:
        model: CNN-BiLSTM model
        loader: Test DataLoader
        device: Device to run on (cpu/cuda)
        decode_method: Best path or beam search, Classes: DecodeMethod
        verbose: When True, prints per-sample predictions, a wrong-prediction summary, and the final CER / EM metrics.

    Returns:
        Tuple of (predictions, references, filenames)
    """

    model.eval()

    if len(loader) == 0:
        raise ValueError("DataLoader is empty")

    all_preds: List[str] = []
    all_refs: List[str] = []
    all_filenames: List[str] = []
    results = []

    for (
        images,
        labels,
        label_lens,
        _,
        filenames,
    ) in tqdm(loader, desc="Testing", leave=False):
        # Move to device with async transfer
        images = images.to(device, non_blocking=True)

        # Forward pass: [B, T, C] logits
        logits = model(images)

        if decode_method is DecodeMethod.BEST_PATH:
            # Best path / Greedy : argmax over class dim, then decode on CPU
            pred_indices = logits.argmax(dim=2).cpu()  # [B, T]
            batch_preds = [best_path_decode(seq, IDX2CHAR) for seq in pred_indices]

        elif decode_method is DecodeMethod.BEAM_SEARCH:
            # Beam search needs per-frame probability distributions
            probs = torch.softmax(logits, dim=2).cpu()  # [B, T, C]
            batch_preds = [
                beam_search_decode(seq, IDX2CHAR, beam_width=beam_width)[0][0]
                for seq in probs
            ]

        else:
            raise ValueError(f"Unsupported decode method: {decode_method!r}")

        # Decode ground-truth targets (already on CPU)
        batch_refs = decode_targets(labels, label_lens, IDX2CHAR)

        all_preds.extend(batch_preds)
        all_refs.extend(batch_refs)
        all_filenames.extend(filenames)

        if verbose:
            from metric.ocr_metrics import cer, em

            for fname, pred_text, true_text in zip(filenames, batch_preds, batch_refs):
                c = cer(true_text, pred_text)
                e = em(true_text, pred_text)
                results.append(
                    {
                        "filename": fname,
                        "gt": true_text,
                        "pred": pred_text,
                        "cer": c,
                        "em": e,
                    }
                )
                print(fname)
                print(f"GT   : {true_text}")
                print(f"PRED : {pred_text}")
                print(f"CER  : {c:.2f} | EM : {'True' if e == 1.0 else 'False'}")
                print("-" * 40)

    if verbose:
        from metric.ocr_metrics import batch_cer, batch_em

        print("\n\n====== WRONG PREDICTIONS ======")
        for r in results:
            if r["gt"] != r["pred"]:
                print(r["filename"])
                print(f"GT   : {r['gt']}")
                print(f"PRED : {r['pred']}")
                print(
                    f"CER  : {r['cer']:.2f} | EM : {'True' if r['em'] == 1.0 else 'False'}"
                )
                print("-" * 40)

        print("\n\n===== FINAL TEST SUMMARY =====")
        print(f"Decode Method : {decode_method.value}")
        if decode_method is DecodeMethod.BEAM_SEARCH:
            print(f"Beam Width    : {beam_width}")
        print(f"Total Samples : {len(results)}")
        print(f"Final CER     : {batch_cer(all_refs, all_preds)}")
        print(f"Final EM      : {batch_em(all_refs, all_preds)}")
        print("==============================")

    return all_preds, all_refs, all_filenames
