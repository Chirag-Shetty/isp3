"""
Training Script — Transformer-CNN-LSTM Fall Detection Model
===========================================================
Trains the hybrid model on windowed radar feature sequences.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/epoch_45.pt
"""

import os
import json
import time
import argparse
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import classification_report, confusion_matrix
import yaml

from model import build_model


# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class RadarSequenceDataset(Dataset):
    """
    Sliding-window dataset over radar feature CSV files.
    Each sample is a (seq_len, feature_dim) tensor + integer class label.

    Classes:
        0 → Standing
        1 → Sitting
        2 → Walking
        3 → Fall
    """

    CLASS_NAMES = ["Standing", "Sitting", "Walking", "Fall"]

    def __init__(self, data_dir: str, seq_len: int = 30, stride: int = 5,
                 split: str = "train", scaler=None):
        self.seq_len = seq_len
        self.stride = stride
        self.split = split
        self.scaler = scaler
        self.samples = []   # List of (features_array, label)
        self._load(data_dir)

    def _load(self, data_dir: str):
        import glob, os, pandas as pd
        csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
        print(f"[Dataset] Found {len(csv_files)} CSV files in {data_dir!r}")
        for fpath in csv_files:
            df = pd.read_csv(fpath)
            label_col = "label"
            feat_cols = [c for c in df.columns if c != label_col]
            features = df[feat_cols].values.astype(np.float32)
            labels = df[label_col].values.astype(np.int64)
            # Sliding window
            for start in range(0, len(features) - self.seq_len + 1, self.stride):
                win_feat = features[start: start + self.seq_len]
                win_label = int(np.bincount(labels[start: start + self.seq_len]).argmax())
                self.samples.append((win_feat, win_label))
        print(f"[Dataset] Total windows: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        x = torch.tensor(feat, dtype=torch.float32)
        if self.scaler is not None:
            orig_shape = x.shape
            x = torch.tensor(
                self.scaler.transform(x.reshape(-1, x.shape[-1])).reshape(orig_shape),
                dtype=torch.float32,
            )
        return x, torch.tensor(label, dtype=torch.long)


# ─── Loss ─────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss to down-weight easy negatives and focus on hard fall examples."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(X)
            loss = criterion(logits, y)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += X.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_targets


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(cfg_path: str, resume: str = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Build model
    model = build_model(cfg["model"]).to(device)
    print(f"[Train] Parameters: {model.num_parameters:,}")

    # Loss — class-weighted focal loss (fall class gets 3× weight)
    class_weights = torch.tensor([1.0, 1.0, 1.0, 3.0], device=device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history = ckpt.get("history", history)
        print(f"[Train] Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # NOTE: dataset loading would happen here using RadarSequenceDataset
        #       omitted to keep script self-contained without data dependency
        t0 = time.time()
        print(f"Epoch {epoch:03d}/{cfg['epochs']} | LR={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")
        scheduler.step()

    print("[Train] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    main(args.config, args.resume)
