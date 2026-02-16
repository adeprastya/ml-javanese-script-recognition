import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes: int, cnn_layers: int, rnn_layers: int):
        super().__init__()

        if cnn_layers not in [3, 5, 7]:
            raise ValueError(f"Invalid cnn_layers: {cnn_layers}, must be 3, 5, or 7.")
        if rnn_layers not in [1, 2, 3]:
            raise ValueError(f"Invalid rnn_layers: {rnn_layers}, must be 1, 2, or 3.")

        # -------- Feature extraction --------
        if cnn_layers == 3:
            self.cnn = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 48 -> 24
                # Block 2
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 24 -> 12
                # Block 3
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),  # 12 -> 6
                # Global Pool
                nn.AdaptiveAvgPool2d((1, None)),  # 6 -> 1
            )
        elif cnn_layers == 5:
            self.cnn = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 48 -> 24
                # Block 2
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 24 -> 12
                # Block 4
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Block 5
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),  # 12 -> 6
                # Global Pool
                nn.AdaptiveAvgPool2d((1, None)),  # 6 -> 1
            )
        elif cnn_layers == 7:
            self.cnn = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 48 -> 24
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
                nn.MaxPool2d(kernel_size=(2, 2)),  # 24 -> 12
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
                nn.MaxPool2d(kernel_size=(2, 1)),  # 12 -> 6
                # Global Pool
                nn.AdaptiveAvgPool2d((1, None)),  # 6 -> 1
            )

        # -------- Sequence modeling --------
        self.rnn = nn.LSTM(
            input_size=256,  # H' * C
            hidden_size=256,  # H' * C
            num_layers=rnn_layers,  # 1 / 2 / 3
            bidirectional=True,
            batch_first=True,
        )

        # -------- Classification --------
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.cnn(x)  # [B, C(256), H(1), W]

        # Reshape
        x = x.squeeze(2)  # [B, C/F(256), W/T]
        x = x.permute(0, 2, 1)  # [B, W/T, C/F(256)]

        # Sequence modeling
        x, _ = self.rnn(x)  # [B, T, F(512)]

        # Classification
        x = self.fc(x)  # [B, W, Class(21)]

        return x
