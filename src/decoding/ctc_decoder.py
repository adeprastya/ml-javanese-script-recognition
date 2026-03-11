"""
CTC Decoding Utilities.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import torch


def ctc_greedy_decode(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    blank: int = 0,
) -> str:
    """
    Greedy CTC decoding: collapse repeats and remove blanks.

    Args:
        preds: [T] predicted class indices
        idx2char: mapping from index to character
        blank: CTC blank label index

    Returns:
        Decoded string
    """
    if len(preds) == 0:
        return ""

    decoded = []
    prev = blank

    for p in preds:
        p = int(p)
        # Keep if: different from previous AND not blank
        if p != prev and p != blank:
            if p not in idx2char:
                raise ValueError(f"Unknown index {p} not in idx2char")
            decoded.append(idx2char[p])
        prev = p

    return "".join(decoded)


def ctc_beam_search_decode(
    probs: torch.Tensor, idx2char: Dict[int, str], beam_width: int = 10, blank: int = 0
) -> List[Tuple[str, float]]:
    """
    CTC Beam Search Decoding.

    Args:
        probs: [T, C] probability matrix (after Softmax)
        idx2char: mapping from index to character
        beam_width: number of hypotheses to keep
        blank: index for blank label

    Returns:
        List of (decoded_text, probability) tuples sorted best-first.
    """
    T, C = probs.shape
    beam = {(): (1.0, 0.0)}

    for t in range(T):
        new_beam = defaultdict(lambda: (0.0, 0.0))

        for c in range(C):
            p = probs[t, c].item()

            if c == blank:
                for prefix, (p_b, p_nb) in beam.items():
                    n_p_b, n_p_nb = new_beam[prefix]
                    new_beam[prefix] = (n_p_b + p * (p_b + p_nb), n_p_nb)
                continue

            # Bug 1 fix: validate key exists, raise ValueError like greedy decode
            if c not in idx2char:
                raise ValueError(f"Unknown index {c} not in idx2char")
            char = idx2char[c]

            for prefix, (p_b, p_nb) in beam.items():
                last_char = prefix[-1] if len(prefix) > 0 else None

                if char == last_char:
                    o_p_b, o_p_nb = new_beam[prefix]
                    new_beam[prefix] = (o_p_b, o_p_nb + p * p_nb)

                    new_prefix = prefix + (char,)
                    n_p_b, n_p_nb = new_beam[new_prefix]
                    new_beam[new_prefix] = (n_p_b, n_p_nb + p * p_b)
                else:
                    new_prefix = prefix + (char,)
                    n_p_b, n_p_nb = new_beam[new_prefix]
                    new_beam[new_prefix] = (n_p_b, n_p_nb + p * (p_b + p_nb))

        beam = dict(
            sorted(new_beam.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)[
                :beam_width
            ]
        )

    # Bug 2 fix: explicitly sort results before returning
    results = sorted(
        [("".join(prefix), p_b + p_nb) for prefix, (p_b, p_nb) in beam.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    return results


def decode_targets(
    labels: torch.Tensor,
    label_lens: torch.Tensor,
    idx2char: Dict[int, str],
) -> List[str]:
    """
    Decode flattened CTC targets to strings.

    Args:
        labels: [sum(label_lens)] concatenated target indices
        label_lens: [B] length of each target sequence
        idx2char: mapping from index to character

    Returns:
        List of decoded strings
    """
    if len(labels) == 0:
        return []

    # Validate total length matches
    total_len = sum(label_lens)
    if len(labels) != total_len:
        raise ValueError(
            f"Length mismatch: labels={len(labels)}, sum(label_lens)={total_len}"
        )

    texts = []
    offset = 0

    for length in label_lens:
        length = int(length)
        seq = labels[offset : offset + length].tolist()

        # Decode sequence
        chars = []
        for idx in seq:
            if idx not in idx2char:
                raise ValueError(f"Unknown index {idx} not in idx2char")
            chars.append(idx2char[idx])

        texts.append("".join(chars))
        offset += length

    return texts
