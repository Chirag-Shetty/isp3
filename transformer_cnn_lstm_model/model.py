"""
Transformer-CNN-LSTM Hybrid Architecture for Radar-Based Fall Detection
=======================================================================
Architecture Overview:
  1. CNN Block      — Extracts local spatial features from radar feature sequences
  2. Transformer    — Self-attention over temporal windows for global context
  3. LSTM           — Sequential memory for activity state tracking
  4. Classifier     — Fully-connected head → 4-class softmax

Classes: [Standing, Sitting, Walking, Fall]

Author  : IDP Group
Version : 2.1.0
Date    : 2025-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
#  Positional Encoding for Transformer
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# ─────────────────────────────────────────────
#  CNN Feature Extractor Block
# ─────────────────────────────────────────────
class CNNFeatureExtractor(nn.Module):
    """
    1-D Temporal CNN to extract local patterns from the radar feature sequence.
    Input  : (batch, seq_len, input_dim)
    Output : (batch, seq_len, cnn_channels)
    """

    def __init__(self, input_dim: int, cnn_channels: int = 128, kernel_size: int = 3):
        super().__init__()
        self.conv_block = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_dim, cnn_channels // 2, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels // 2),
            nn.GELU(),
            # Layer 2
            nn.Conv1d(cnn_channels // 2, cnn_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            # Layer 3 — residual depth
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )
        # Projection shortcut for residual
        self.shortcut = nn.Conv1d(input_dim, cnn_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) → transpose to (batch, input_dim, seq_len)
        x_t = x.transpose(1, 2)
        out = self.conv_block(x_t) + self.shortcut(x_t)
        return out.transpose(1, 2)  # back to (batch, seq_len, cnn_channels)


# ─────────────────────────────────────────────
#  Transformer Encoder Block
# ─────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    """
    Multi-head self-attention + feedforward sublayers.
    Captures global temporal dependencies across the radar frame window.
    """

    def __init__(self, d_model: int, nhead: int = 8, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # positional encoding expects (seq_len, batch, d_model)
        x_pe = self.pos_enc(x.transpose(0, 1)).transpose(0, 1)
        out = self.encoder(x_pe)
        return self.norm(out)


# ─────────────────────────────────────────────
#  LSTM Temporal Memory Block
# ─────────────────────────────────────────────
class LSTMTemporalBlock(nn.Module):
    """
    Bidirectional LSTM to model sequential state transitions
    (e.g., walking → stumble → fall).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2  # bidirectional

    def forward(self, x: torch.Tensor):
        out, (h_n, _) = self.lstm(x)
        # Concatenate last forward and backward hidden states
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        return out, last_hidden


# ─────────────────────────────────────────────
#  Attention Pooling
# ─────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """Learnable attention-weighted pooling over sequence dimension."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(x), dim=1)   # (batch, seq_len, 1)
        return (x * weights).sum(dim=1)                 # (batch, hidden_dim)


# ─────────────────────────────────────────────
#  Main Transformer-CNN-LSTM Model
# ─────────────────────────────────────────────
class TransformerCNNLSTM(nn.Module):
    """
    Hybrid Deep Learning Architecture for Radar-Based Fall Detection.

    Data flow:
        Input features  →  CNN  →  Transformer  →  LSTM  →  Attention Pool
                        →  Classifier  →  4-class output

    Args:
        input_dim   : number of input features per frame (e.g., 18)
        seq_len     : temporal window length (number of radar frames)
        num_classes : number of activity classes (default: 4)
        cnn_ch      : CNN output channels / Transformer d_model
        lstm_hidden : LSTM hidden state dimension
        dropout     : global dropout rate
    """

    def __init__(
        self,
        input_dim: int = 18,
        seq_len: int = 30,
        num_classes: int = 4,
        cnn_ch: int = 128,
        lstm_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len

        # Stage 1: CNN — local spatial-feature extraction
        self.cnn = CNNFeatureExtractor(input_dim, cnn_channels=cnn_ch)

        # Stage 2: Transformer — global temporal attention
        self.transformer = TransformerEncoderBlock(
            d_model=cnn_ch, nhead=8, dim_ff=cnn_ch * 4, dropout=dropout
        )

        # Stage 3: LSTM — sequential state memory
        self.lstm = LSTMTemporalBlock(cnn_ch, hidden_dim=lstm_hidden, num_layers=2, dropout=dropout)

        # Attention pooling over LSTM outputs
        self.attn_pool = AttentionPooling(self.lstm.output_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm.output_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, input_dim) — sequence of radar feature vectors
        Returns:
            logits : (batch, num_classes)
        """
        # CNN stage
        cnn_out = self.cnn(x)                           # (B, T, cnn_ch)

        # Transformer stage
        trans_out = self.transformer(cnn_out)            # (B, T, cnn_ch)

        # LSTM stage
        lstm_out, lstm_last = self.lstm(trans_out)       # (B, T, lstm*2), (B, lstm*2)

        # Attention pooling over LSTM sequence
        pooled = self.attn_pool(lstm_out)               # (B, lstm*2)

        # Fuse pooled + last hidden state
        fused = torch.cat([pooled, lstm_last], dim=1)   # (B, lstm*4)

        return self.classifier(fused)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
#  Convenience factory
# ─────────────────────────────────────────────
def build_model(cfg: dict) -> TransformerCNNLSTM:
    return TransformerCNNLSTM(
        input_dim=cfg.get("input_dim", 18),
        seq_len=cfg.get("seq_len", 30),
        num_classes=cfg.get("num_classes", 4),
        cnn_ch=cfg.get("cnn_channels", 128),
        lstm_hidden=cfg.get("lstm_hidden", 256),
        dropout=cfg.get("dropout", 0.3),
    )


if __name__ == "__main__":
    model = TransformerCNNLSTM()
    dummy = torch.randn(8, 30, 18)          # batch=8, seq=30, features=18
    out = model(dummy)
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {model.num_parameters:,}")
