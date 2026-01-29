import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """
    CNN + BiLSTM model for CTC-based OCR.
    Input  : [B, 1, H, W]
    Output : [B, T, num_classes]
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # # --------- CNN feature extractor (3 conv layers) --------
        # self.cnn = nn.Sequential(
        #     # Block 1
        #     nn.Conv2d(1, 64, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),
        #     # Block 2
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),
        #     # Block 3
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     # Projection to 512
        #     nn.Conv2d(256, 512, 1),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, None)),
        # )

        # -------- CNN feature extractor (5 conv layers) --------
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            # Collapse height dimension to 1
            nn.AdaptiveAvgPool2d((1, None)),
        )

        # # --------- CNN feature extractor (7 conv layers) ----------
        # self.cnn = nn.Sequential(
        #     # Block 1
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),  # H/2, W/2
        #     # Block 2
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2),  # H/4, W/4
        #     # Block 3
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     # Block 4
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 1)),  # H/8, W tetap
        #     # Block 5
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     # Block 6
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 1)),  # H/16, W tetap
        #     # Block 7
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     # Collapse height dimension to 1
        #     nn.AdaptiveAvgPool2d((1, None)),
        # )

        # ---------- Sequence modeling ----------
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        # ---------- Classification ----------
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        x: [B, 1, H, W]
        return: [B, T, num_classes]
        """
        x = self.cnn(x)  # [B, 512, 1, W']
        x = x.squeeze(2)  # [B, 512, W']
        x = x.permute(0, 2, 1)  # [B, W', 512]

        x, _ = self.rnn(x)  # [B, W', 512]
        x = self.fc(x)  # [B, W', num_classes]

        return x
