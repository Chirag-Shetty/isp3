"""
model.py  —  Multi-Class Transformer-CNN-LSTM architecture
------------------------------------------------------------
Matches the architecture from multi_class_training_colab.py exactly.

Architecture:
  input_proj           : Linear(20, 64)
  transformer_encoder  : TransformerEncoder (2 layers, d_model=64, nhead=4)
  cnn                  : Conv1d(64→32, k=3) → ReLU → BN → Dropout → MaxPool
  lstm                 : LSTM(32 → 64)
  fc                   : Dropout → Linear(64→32) → ReLU → Dropout → Linear(32→6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassTransformerCNNLSTM(nn.Module):
    def __init__(self, num_features=20, num_classes=6, d_model=64, nhead=4,
                 num_layers=2, cnn_channels=32, lstm_hidden=64, dropout=0.4):
        super().__init__()
        # Transformer Stage
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dropout=dropout, dim_feedforward=128
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CNN Stage
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2)
        )

        # LSTM Stage
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden,
                           batch_first=True, num_layers=1)

        # Output Stage
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=-1)
