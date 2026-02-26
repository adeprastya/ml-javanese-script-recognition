import copy
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes: int, cnn_layers: int, rnn_layers: int):
        super().__init__()

        if cnn_layers not in [3, 4, 5, 6, 7]:
            raise ValueError(
                f"Invalid cnn_layers: {cnn_layers}, must be 3, 4, 5, 6, or 7."
            )
        if rnn_layers not in [1, 2, 3]:
            raise ValueError(f"Invalid rnn_layers: {rnn_layers}, must be 1, 2, or 3.")

        # ========================================
        # Feature Extraction & Downsampling (CNN)
        # ========================================
        cnn_blocks = {
            # Stage 1 Transition (1 -> 64 channels) & Downsampling (48 -> 24 height)
            "stage1_transition": [
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
            ],
            # Stage 1 Unit (64 -> 64 channels)
            "stage1_unit": [
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ],
            # Stage 2 Transition (64 -> 128 channels) & Downsampling (24 -> 12 height)
            "stage2_transition": [
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
            ],
            # Stage 2 Unit (128 -> 128 channels)
            "stage2_unit": [
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ],
            # Stage 3 Transition (128 -> 256 channels) & Downsampling (12 -> 6 height)
            "stage3_transition": [
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
            ],
            # Global Vertical Pooling (6 -> 1 height)
            "global_pool": [nn.AdaptiveAvgPool2d((1, None))],
        }

        cnn_structure_map = {
            3: [
                "stage1_transition",
                "stage2_transition",
                "stage3_transition",
                "global_pool",
            ],
            4: [
                "stage1_transition",
                "stage2_transition",
                "stage2_unit",
                "stage3_transition",
                "global_pool",
            ],
            5: [
                "stage1_transition",
                "stage1_unit",
                "stage2_transition",
                "stage2_unit",
                "stage3_transition",
                "global_pool",
            ],
            6: [
                "stage1_transition",
                "stage1_unit",
                "stage2_transition",
                "stage2_unit",
                "stage2_unit",
                "stage3_transition",
                "global_pool",
            ],
            7: [
                "stage1_transition",
                "stage1_unit",
                "stage1_unit",
                "stage2_transition",
                "stage2_unit",
                "stage2_unit",
                "stage3_transition",
                "global_pool",
            ],
        }

        layers = []
        for block_name in cnn_structure_map[cnn_layers]:
            block_layers = copy.deepcopy(cnn_blocks[block_name])
            layers.extend(block_layers)
        self.cnn = nn.Sequential(*layers)

        # ========================================
        # Contextual Sequence Modeling (BiLSTM)
        # ========================================
        self.rnn = nn.LSTM(
            input_size=256,  # H' * C
            hidden_size=256,  # H' * C
            num_layers=rnn_layers,  # 1 / 2 / 3
            bidirectional=True,
            batch_first=True,
        )

        # ========================================
        # Class Distribution Probabilities (Linear)
        # ========================================
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        # Feature Extraction & Downsampling
        x = self.cnn(x)  # [B, C(256), H(1), W]

        # Reshape
        x = x.squeeze(2)  # [B, C/F(256), W/T]
        x = x.permute(0, 2, 1)  # [B, W/T, C/F(256)]

        # Contextual Sequence Modeling
        x, _ = self.rnn(x)  # [B, T, F(512)]

        # Class Distribution Probabilities
        x = self.fc(x)  # [B, W, Class(21)]

        return x


# # -------- Feature Extraction & Downsampling (CNN) --------
# if cnn_layers == 3:
#     self.cnn = nn.Sequential(
#         # Block 1
#         nn.Conv2d(1, 64, kernel_size=3, padding=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 2)),  # 48 -> 24
#         # Block 2
#         nn.Conv2d(64, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 2)),  # 24 -> 12
#         # Block 3
#         nn.Conv2d(128, 256, kernel_size=3, padding=1),
#         nn.BatchNorm2d(256),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 1)),  # 12 -> 6
#         # Global Pool
#         nn.AdaptiveAvgPool2d((1, None)),  # 6 -> 1
#     )
# elif cnn_layers == 4:
#     self.cnn = nn.Sequential(
#         # Block 1
#         nn.Conv2d(1, 64, kernel_size=3, padding=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 2)),  # 48 -> 24
#         # Block 2
#         nn.Conv2d(64, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 2)),  # 24 -> 12
#         # Block 3
#         nn.Conv2d(128, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         # Block 4
#         nn.Conv2d(128, 256, kernel_size=3, padding=1),
#         nn.BatchNorm2d(256),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 1)),  # 12 -> 6
#         # Global Pool
#         nn.AdaptiveAvgPool2d((1, None)),  # 6 -> 1
#     )
# elif cnn_layers == 5:
#     self.cnn = nn.Sequential(
#         # Block 1
#         nn.Conv2d(1, 64, kernel_size=3, padding=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 2)),  # 48 -> 24
#         # Block 2
#         nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         # Block 3
#         nn.Conv2d(64, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 2)),  # 24 -> 12
#         # Block 4
#         nn.Conv2d(128, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         # Block 5
#         nn.Conv2d(128, 256, kernel_size=3, padding=1),
#         nn.BatchNorm2d(256),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 1)),  # 12 -> 6
#         # Global Pool
#         nn.AdaptiveAvgPool2d((1, None)),  # 6 -> 1
#     )
# elif cnn_layers == 6:
#     self.cnn = nn.Sequential(
#         # Block 1
#         nn.Conv2d(1, 64, kernel_size=3, padding=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 2)),  # 48 -> 24
#         # Block 2
#         nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         # Block 3
#         nn.Conv2d(64, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 2)),  # 24 -> 12
#         # Block 4
#         nn.Conv2d(128, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         # Block 5
#         nn.Conv2d(128, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         # Block 6
#         nn.Conv2d(128, 256, kernel_size=3, padding=1),
#         nn.BatchNorm2d(256),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 1)),  # 12 -> 6
#         # Global Pool
#         nn.AdaptiveAvgPool2d((1, None)),  # 6 -> 1
#     )
# elif cnn_layers == 7:
#     self.cnn = nn.Sequential(
#         # Block 1
#         nn.Conv2d(1, 64, kernel_size=3, padding=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 2)),  # 48 -> 24
#         # Block 2
#         nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         # Block 3
#         nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         # Block 4
#         nn.Conv2d(64, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 2)),  # 24 -> 12
#         # Block 5
#         nn.Conv2d(128, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         # Block 6
#         nn.Conv2d(128, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         # Block 7
#         nn.Conv2d(128, 256, kernel_size=3, padding=1),
#         nn.BatchNorm2d(256),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=(2, 1)),  # 12 -> 6
#         # Global Pool
#         nn.AdaptiveAvgPool2d((1, None)),  # 6 -> 1
#     )
