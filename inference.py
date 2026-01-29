def main():
    print("Importing modules...")

    import torch
    from torch.utils.data import DataLoader, ConcatDataset
    from tqdm import tqdm

    from src.cnn_bilstm import CNNBiLSTM
    from src.aksara import CHAR_LIST, IDX2CHAR
    from src.dataset import JavaneseOCRDataset, ctc_collate_fn
    from src.decoding import decode_targets, ctc_greedy_decode
    from src.ocr_metrics import character_error_rate, exact_match_accuracy

    # =======================
    # CONFIG
    # =======================
    print("Configuring...")

    BASE_DATA_DIR = "data/word_nglegena_handwritten_20260129_191226"

    MODEL_PATH = "models/5layerCNN_50epoch/last_model.pth"
    BATCH_SIZE = 16
    NUM_WORKERS = 3
    IMG_HEIGHT = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CHAR_LIST) + 1  # + blank

    # =======================
    # MODEL
    # =======================
    print("Loading model...")

    model = CNNBiLSTM(num_classes=num_classes).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # =======================
    # DATA
    # =======================
    print("Loading data...")

    test_ds_0 = JavaneseOCRDataset(
        csv_path="data/test-sample/label.csv",
        img_dir="data/test-sample/image",
        img_height=IMG_HEIGHT,
        clahe=False,
    )
    test_ds_1 = JavaneseOCRDataset(
        csv_path=f"{BASE_DATA_DIR}/label_1.csv",
        img_dir=f"{BASE_DATA_DIR}/image_1",
        img_height=IMG_HEIGHT,
        clahe=False,
    )

    test_ds = ConcatDataset([test_ds_1])

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=ctc_collate_fn,
    )

    # =======================
    # INFERENCE
    # =======================
    print("Inferencing...")

    all_preds, all_refs = [], []
    sample_idx = 1

    pbar = tqdm(
        test_loader,
        desc="Inference",
        unit="batch",
        leave=False,
    )

    for images, labels, label_lens, input_lens in pbar:
        images = images.to(device, non_blocking=True)

        with torch.inference_mode():
            logits = model(images)  # [B, T, C]
            preds = logits.argmax(dim=2)  # [B, T]

        batch_output = []

        gt_texts = decode_targets(labels, label_lens, IDX2CHAR)

        for b in range(images.size(0)):
            pred_text = ctc_greedy_decode(preds[b], IDX2CHAR)
            gt_text = gt_texts[b]

            all_preds.append(pred_text)
            all_refs.append(gt_text)

            mark = "✓" if pred_text == gt_text else ""
            batch_output.append(
                f"{sample_idx:03d} | GT: {gt_text:<12} | "
                f"Pred: {pred_text:<12} | {mark}"
            )
            sample_idx += 1

        tqdm.write("\n".join(batch_output))

    # ----------- METRICS -----------
    cer = character_error_rate(all_preds, all_refs)
    em = exact_match_accuracy(all_preds, all_refs)

    print("\n" + "=" * 36)
    print(f"Character Error Rate : {cer:.4f}")
    print(f"Exact Match Accuracy : {em:.4f}")
    print("=" * 36)


if __name__ == "__main__":
    main()
