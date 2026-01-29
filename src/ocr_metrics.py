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


def character_error_rate(preds, refs):
    """
    CER = total edit distance / total characters in references
    """
    assert len(preds) == len(refs), "preds and refs must have same length"

    total_edits = sum(levenshtein(p, r) for p, r in zip(preds, refs))
    total_chars = sum(len(r) for r in refs)

    return total_edits / max(total_chars, 1)


def exact_match_accuracy(preds, refs):
    """
    Exact Match Accuracy (sentence-level)
    """
    assert len(preds) == len(refs), "preds and refs must have same length"

    correct = sum(p == r for p, r in zip(preds, refs))
    return correct / max(len(refs), 1)
