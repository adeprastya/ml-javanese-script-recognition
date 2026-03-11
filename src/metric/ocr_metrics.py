"""
Evaluation Metrics for OCR/HTR.
"""

from typing import List


def levenshtein(a: str, b: str) -> int:
    """
    Compute Levenshtein edit distance (insertions, deletions, substitutions).

    Args:
        a: Source string
        b: Target string

    Returns:
        Minimum edit distance
    """
    n, m = len(a), len(b)

    # Edge cases
    if n == 0:
        return m
    if m == 0:
        return n

    # Space-optimized DP (O(m) space)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,  # deletion
                dp[j - 1] + 1,  # insertion
                prev + (a[i - 1] != b[j - 1]),  # substitution
            )
            prev = cur

    return dp[m]


def cer(pred: str, ref: str) -> float:
    """
    Character Error Rate: edit_distance / reference_length.

    Args:
        pred: Predicted string
        ref: Reference (ground truth) string

    Returns:
        CER in range [0, inf), lower is better
    """
    if not isinstance(pred, str) or not isinstance(ref, str):
        raise TypeError(f"Expected str, got pred={type(pred)}, ref={type(ref)}")

    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else float("inf")

    return levenshtein(pred, ref) / len(ref)


def em(pred: str, ref: str) -> float:
    """
    Exact Match: 1.0 if strings are identical, else 0.0.

    Args:
        pred: Predicted string
        ref: Reference string

    Returns:
        1.0 or 0.0
    """
    if not isinstance(pred, str) or not isinstance(ref, str):
        raise TypeError(f"Expected str, got pred={type(pred)}, ref={type(ref)}")

    return 1.0 if pred == ref else 0.0


def batch_cer(preds: List[str], refs: List[str]) -> float:
    """
    Batch CER: total_edits / total_chars.

    Args:
        preds: List of predicted strings
        refs: List of reference strings

    Returns:
        Micro-averaged CER
    """
    if len(preds) != len(refs):
        raise ValueError(f"Length mismatch: preds={len(preds)}, refs={len(refs)}")

    if len(refs) == 0:
        return 0.0

    total_edits = sum(levenshtein(p, r) for p, r in zip(preds, refs))
    total_chars = sum(len(r) for r in refs)

    return total_edits / max(total_chars, 1)


def batch_em(preds: List[str], refs: List[str]) -> float:
    """
    Batch Exact Match Accuracy: correct_count / total_count.

    Args:
        preds: List of predicted strings
        refs: List of reference strings

    Returns:
        Accuracy in range [0, 1]
    """
    if len(preds) != len(refs):
        raise ValueError(f"Length mismatch: preds={len(preds)}, refs={len(refs)}")

    if len(refs) == 0:
        return 0.0

    correct = sum(p == r for p, r in zip(preds, refs))
    return correct / len(refs)
