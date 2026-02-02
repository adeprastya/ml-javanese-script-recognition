def levenshtein(a: str, b: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    """
    n, m = len(a), len(b)

    if n == 0:
        return m
    if m == 0:
        return n

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


def cer(pred, ref):
    """
    Compute Character Error Rate for a single prediction vs reference
    """
    if isinstance(pred, str) and isinstance(ref, str):
        total_edits = levenshtein(pred, ref)
        total_chars = len(ref)
        return total_edits / max(total_chars, 1)
    else:
        raise ValueError("pred and ref must be strings")


def em(pred, ref):
    """
    Exact Match for a single prediction vs reference
    """
    if isinstance(pred, str) and isinstance(ref, str):
        return 1.0 if pred == ref else 0.0
    else:
        raise ValueError("pred and ref must be strings")


def batch_cer(preds, refs):
    """
    Compute CER for a batch of predictions and references
    preds: list of str
    refs: list of str
    """
    assert len(preds) == len(refs), "preds and refs must have same length"
    total_edits = sum(levenshtein(p, r) for p, r in zip(preds, refs))
    total_chars = sum(len(r) for r in refs)
    return total_edits / max(total_chars, 1)


def batch_em(preds, refs):
    """
    Compute Exact Match Accuracy for a batch of predictions and references
    """
    assert len(preds) == len(refs), "preds and refs must have same length"
    correct = sum(p == r for p, r in zip(preds, refs))
    return correct / max(len(refs), 1)
