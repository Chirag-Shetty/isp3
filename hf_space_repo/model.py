"""
model.py  -  Binary Fall Detection Transformer-CNN-LSTM
--------------------------------------------------------
Architecture matches FallDetectionTransformerCNNLSTM from training exactly.
2 output classes: 0=NO-FALL, 1=FALL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FallDetectionTransformerCNNLSTM(nn.Module):
    def __init__(self, input_size=20, d_model=64, nhead=4,
                 num_transformer_layers=2, cnn_channels=32,
                 lstm_hidden=64, num_classes=2, dropout=0.4):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=dropout,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2))

        self.lstm = nn.LSTM(
            input_size=cnn_channels, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes))

    def forward(self, x):
        x = self.input_proj(x)          # (B, 40, 64)
        x = self.transformer(x)          # (B, 40, 64)
        x = x.transpose(1, 2)           # (B, 64, 40)
        x = self.cnn(x)                 # (B, 32, 20)
        x = x.transpose(1, 2)           # (B, 20, 32)
        _, (h, _) = self.lstm(x)        # h: (1, B, 64)
        x = h.squeeze(0)                # (B, 64)
        return self.classifier(x)       # (B, 2)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=-1)
