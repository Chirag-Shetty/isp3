"""
model.py
--------
The exact Transformer-CNN-LSTM architecture from your training notebook.
Load this on the RPi with your best_model.pth weights.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.query(x), dim=1)
        return (attn_weights * x).sum(dim=1)


class TransformerCNNLSTM(nn.Module):
    def __init__(
        self,
        num_features=12,
        num_classes=6,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        cnn_filters=128,
        cnn_kernel=3,
        lstm_hidden=128,
        dropout=0.4,
    ):
        super().__init__()

        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout * 0.5)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout * 0.5,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_norm = nn.LayerNorm(d_model)

        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, cnn_filters, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_filters), nn.GELU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_filters), nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout * 0.5,
        )

        self.attention_pool = AttentionPooling(lstm_hidden * 2)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 128), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.transformer_norm(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.attention_pool(x)
        return self.classifier(x)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=-1)
