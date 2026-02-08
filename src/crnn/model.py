"""
CRNN Text Recognizer — Model Definition
==========================================
CNN + BiLSTM + CTC for multi-script text recognition.

Architecture:
  Input (32xWxC) -> CNN Encoder -> BiLSTM Sequence Model -> Linear -> CTC Decode

Supports 12 Indian scripts + Latin (700+ unique characters).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .config import CRNNConfig as cfg


class CNNEncoder(nn.Module):
    """7-layer CNN feature extractor: (B,C,32,W) -> (B,W',512)"""

    def __init__(self, in_channels=1):
        super().__init__()
        ch = cfg.CNN_CHANNELS  # [64, 128, 256, 256, 512, 512, 512]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], 3, 1, 1), nn.BatchNorm2d(ch[0]), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(ch[0], ch[1], 3, 1, 1), nn.BatchNorm2d(ch[1]), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(ch[1], ch[2], 3, 1, 1), nn.BatchNorm2d(ch[2]), nn.ReLU(True),
            nn.Conv2d(ch[2], ch[3], 3, 1, 1), nn.BatchNorm2d(ch[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), nn.Dropout2d(0.1),
            nn.Conv2d(ch[3], ch[4], 3, 1, 1), nn.BatchNorm2d(ch[4]), nn.ReLU(True),
            nn.Conv2d(ch[4], ch[5], 3, 1, 1), nn.BatchNorm2d(ch[5]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), nn.Dropout2d(0.1),
            nn.Conv2d(ch[5], ch[6], 2, 1, 0), nn.BatchNorm2d(ch[6]), nn.ReLU(True),
        )

    def forward(self, x):
        conv = self.features(x)     # (B, 512, 1, W')
        conv = conv.squeeze(2)      # (B, 512, W')
        return conv.permute(0, 2, 1)  # (B, W', 512)


class BiLSTMSequenceModel(nn.Module):
    """Bidirectional LSTM for sequential context."""

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.linear(output)


class CRNN(nn.Module):
    """
    CRNN: CNN + BiLSTM + CTC for multi-script text recognition.

    Pipeline: Image(32xW) -> CNN -> BiLSTM -> Linear -> CTC
    """

    def __init__(self, num_classes, img_h=None, num_channels=None):
        super().__init__()
        self.num_classes = num_classes
        self.img_h = img_h or cfg.IMG_HEIGHT
        self.num_channels = num_channels or cfg.NUM_CHANNELS

        self.cnn = CNNEncoder(in_channels=self.num_channels)
        self.rnn = BiLSTMSequenceModel(
            cfg.CNN_CHANNELS[-1], cfg.HIDDEN_SIZE,
            cfg.NUM_LSTM_LAYERS, cfg.DROPOUT)
        self.output = nn.Linear(cfg.HIDDEN_SIZE, num_classes)

    def forward(self, x):
        features = self.cnn(x)            # (B, W', 512)
        sequence = self.rnn(features)     # (B, W', 256)
        logits = self.output(sequence)    # (B, W', num_classes)
        return F.log_softmax(logits, dim=2)

    def decode_greedy(self, log_probs):
        _, preds = log_probs.max(dim=2)
        results = []
        for b in range(preds.size(0)):
            decoded, prev = [], -1
            for p in preds[b].tolist():
                if p != prev and p != cfg.CTC_BLANK:
                    decoded.append(p)
                prev = p
            results.append(decoded)
        return results

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(),
                     "num_classes": self.num_classes,
                     "img_h": self.img_h,
                     "num_channels": self.num_channels}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["num_classes"], ckpt.get("img_h"), ckpt.get("num_channels"))
        model.load_state_dict(ckpt["state_dict"])
        return model


class CharCodec:
    """Maps characters <-> indices for CTC."""

    def __init__(self, charset_path=None):
        if charset_path:
            with open(charset_path, "r", encoding="utf-8") as f:
                chars = [line.strip() for line in f if line.strip()]
        else:
            chars = []
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}
        self.idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}
        self.num_classes = len(chars) + 1

    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, indices):
        return "".join(self.idx_to_char.get(idx, "") for idx in indices)

    def __len__(self):
        return self.num_classes
