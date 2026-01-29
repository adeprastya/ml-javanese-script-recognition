def ctc_greedy_decode(preds, idx2char, blank=0):
    """
    Greedy CTC decoding for a single time-step sequence.
    preds: Tensor[T]
    """
    decoded = []
    prev = blank

    for p in preds:
        p = int(p)
        if p != prev and p != blank:
            decoded.append(idx2char[p])
        prev = p

    return "".join(decoded)


def decode_targets(labels, label_lens, idx2char):
    """
    Decode flattened CTC targets into list of strings.
    """
    texts = []
    offset = 0

    for length in label_lens:
        seq = labels[offset : offset + length].tolist()
        texts.append("".join(idx2char[i] for i in seq))
        offset += length

    return texts
