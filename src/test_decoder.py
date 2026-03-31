import torch
from torch.utils.data import DataLoader, ConcatDataset

from data.collate import ctc_collate
from data.dataset import JavaneseOCRDataset
from model.cnn_bilstm import CNNBiLSTM
from transform.preprocessing import get_preprocessing_pipeline
from training.test import test_one_epoch, DecodeMethod

CONFIG = [
    # Warmup
    {
        "cnn": 3,
        "decoder": DecodeMethod.BEST_PATH,
        "beam": 99,
    },
    # ====================
    {
        "cnn": 3,
        "decoder": DecodeMethod.BEST_PATH,
        "beam": 0,
    },
    # {
    #     "cnn": 3,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 5,
    # },
    # {
    #     "cnn": 3,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 10,
    # },
    # {
    #     "cnn": 3,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 20,
    # },
    # {
    #     "cnn": 3,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 50,
    # },
    # # ====================
    # {
    #     "cnn": 4,
    #     "decoder": DecodeMethod.BEST_PATH,
    #     "beam": 0,
    # },
    # {
    #     "cnn": 4,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 5,
    # },
    # {
    #     "cnn": 4,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 10,
    # },
    # {
    #     "cnn": 4,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 20,
    # },
    # {
    #     "cnn": 4,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 50,
    # },
    # # ====================
    # {
    #     "cnn": 5,
    #     "decoder": DecodeMethod.BEST_PATH,
    #     "beam": 0,
    # },
    # {
    #     "cnn": 5,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 5,
    # },
    # {
    #     "cnn": 5,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 10,
    # },
    # {
    #     "cnn": 5,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 20,
    # },
    # {
    #     "cnn": 5,
    #     "decoder": DecodeMethod.BEAM_SEARCH,
    #     "beam": 50,
    # },
]

BASE_REAL_DIR = "dataset/word_nglegena_handwritten_20260130_155805"
DATA_SOURCES = {
    "test": [
        {
            "csv": f"{BASE_REAL_DIR}/label_1.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_1",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=48, enhance=False),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_2.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_2",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=48, enhance=False),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_3.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_3",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=48, enhance=False),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_4.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_4",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=48, enhance=False),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_5.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_5",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=48, enhance=False),
        },
        {
            "csv": f"{BASE_REAL_DIR}/label_6.csv",
            "img_dir": f"{BASE_REAL_DIR}/image_6",
            "aug": None,
            "prep": get_preprocessing_pipeline(img_height=48, enhance=False),
        },
    ],
}


def main(cnn, decoder, beam_width):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = ConcatDataset(
        [
            JavaneseOCRDataset(
                src["csv"],
                src["img_dir"],
                preprocessing=src["prep"],
                augmentation=src["aug"],
            )
            for src in DATA_SOURCES["test"]
        ]
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        collate_fn=ctc_collate,
    )

    model = CNNBiLSTM(num_classes=21, cnn_layers=cnn, rnn_layers=2).to(device)
    model_path = ""
    if cnn == 3:
        model_path = "builds/scenario_300test/1_scenario-3cnn_2bilstm/last_model.pth"
    elif cnn == 4:
        model_path = "builds/scenario_300test/2_scenario-4cnn_2bilstm/last_model.pth"
    elif cnn == 5:
        model_path = "builds/scenario_300test/3_scenario-5cnn_2bilstm/last_model.pth"
    else:
        raise TypeError("Invalid CNN layer count {cnn}. Must be 3, 4, or 5.")

    checkpoint = torch.load(
        model_path,
        weights_only=True,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_preds, all_refs, all_filenames, _, _, _, _, _, _ = test_one_epoch(
        model,
        test_loader,
        torch.device(device),
        decoder,
        beam_width,
        verbose=False,
    )

    return all_preds, all_refs, all_filenames


if __name__ == "__main__":
    for c in CONFIG:
        cnn = c["cnn"]
        decoder = c["decoder"]
        beam = c["beam"]

        print(f"===== {cnn}CNN-2BiLSTM | {decoder} | {beam} =========================")
        for i in range(3):
            main(cnn, decoder, beam)
            print("==============================\n")
