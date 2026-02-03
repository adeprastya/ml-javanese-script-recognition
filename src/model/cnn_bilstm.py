import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes: int, cnn_layers: int, rnn_layers: int):
        super().__init__()

        if cnn_layers == 3:
            self.cnn = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 64 -> 32
                # Block 2
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 32 -> 16
                # Block 3
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),  # 16 -> 8
                # Global
                nn.AdaptiveAvgPool2d((1, None)),
            )
        elif cnn_layers == 5:
            self.cnn = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 64 -> 32
                # Block 2
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 32 -> 16
                # Block 4
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Block 5
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),  # 16 -> 8
                # Global
                nn.AdaptiveAvgPool2d((1, None)),
            )
        elif cnn_layers == 7:
            self.cnn = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 64 -> 32
                # Block 2
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Block 3
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Block 4
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 32 -> 16
                # Block 5
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Block 6
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Block 7
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),  # 16 -> 8
                # Global
                nn.AdaptiveAvgPool2d((1, None)),
            )
        else:
            raise ValueError(f"Invalid value for cnn_layers: {cnn_layers}")

        # ---------- Sequence modeling ----------
        self.rnn = nn.LSTM(
            input_size=256,  # [256, 512]
            hidden_size=256,
            num_layers=rnn_layers,  # [1, 2, 3]
            bidirectional=True,
            batch_first=True,
        )

        # ---------- Classification ----------
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # [B, CNN out ch, 1, W']
        x = x.squeeze(2)  # [B, CNN out ch, W']
        x = x.permute(0, 2, 1)  # [B, W', CNN out ch]

        x, _ = self.rnn(x)  # [B, W', BiLSTM out ch]
        x = self.fc(x)  # [B, W', num_classes]

        return x


# # --------- CNN feature extractor (3 conv layers) --------
# self.cnn = nn.Sequential(
#     # Block 1
#     nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(2, 2),
#     # Block 2
#     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(kernel_size=(2, 2)),
#     # Block 3
#     nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
#     nn.BatchNorm2d(512),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(kernel_size=(2, 1)),
#     # Collapse height dimension to 1 (C, 1, W)
#     nn.AdaptiveAvgPool2d((1, None)),
# )

# # -------- CNN feature extractor (5 conv layers) --------
# self.cnn = nn.Sequential(
#     # Block 1
#     nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(2, 2),
#     # Block 2
#     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(2, 2),
#     # Block 3
#     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#     nn.BatchNorm2d(256),
#     nn.ReLU(inplace=True),
#     # Block 4
#     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#     nn.ReLU(inplace=True),
#     nn.BatchNorm2d(256),  # UPDATED
#     nn.MaxPool2d(kernel_size=(2, 1)),
#     # Block 5
#     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
#     nn.BatchNorm2d(512),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(kernel_size=(2, 1)),
#     # Collapse height dimension to 1 (C, 1, W)
#     nn.AdaptiveAvgPool2d((1, None)),
# )

# # --------- CNN feature extractor (7 conv layers) ----------
# self.cnn = nn.Sequential(
#     # Block 1
#     nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(2, 2),
#     # Block 2
#     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(2, 2),
#     # Block 3
#     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#     nn.BatchNorm2d(256),  # UPDATED
#     nn.ReLU(inplace=True),
#     # Block 4
#     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#     nn.BatchNorm2d(256),  # UPDATED
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(kernel_size=(2, 1)),
#     # Block 5
#     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
#     nn.BatchNorm2d(512),  # UPDATED
#     nn.ReLU(inplace=True),
#     # Block 6
#     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#     nn.BatchNorm2d(512),  # UPDATED
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(kernel_size=(2, 1)),
#     # Block 7
#     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#     nn.BatchNorm2d(512),  # UPDATED
#     nn.ReLU(inplace=True),
#     # Collapse height dimension to 1
#     nn.AdaptiveAvgPool2d((1, None)),
# )
