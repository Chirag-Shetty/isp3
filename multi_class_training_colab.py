# -*- coding: utf-8 -*-
"""Multi-Class Radar Activity Detection (with Data Augmentation)

This script is designed to be copy-pasted into a Google Colab Cell.
It trains a multi-class Transformer-CNN-LSTM model for human activity recognition using the mmWave Radar.
Since the dataset is small, it applies noise and temporal shift augmentation.

Feature extraction is aligned 1:1 with rpi_pipeline/feature_extract.py so inference matches training.

Remember to upload your dataset to Colab (e.g., zip all your JSONs into `dataset.zip` and unzip it).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import glob
import random
import pickle

# =========================================================
# 1. Configuration & Classes
# =========================================================
dataset_path = "./dataset"  # Change this if your unzipped folder has a different name
NUM_FEATURES = 20
WINDOW_SIZE = 40
FRAME_DT = 0.055   # seconds between frames (~18 fps), must match rpi config
SNR_THRESHOLD = 10.0

# Define your multi-class dictionary based on your folders
CLASSES = {
    "Standing_walk": 0,
    "Sitting_chair": 1,
    "sitting_floor": 2,
    "Stand_Sit_chair_transition": 3,
    "chair_floor_transition": 4,
    "stand_floor_transition": 5,
}
NUM_CLASSES = len(CLASSES)

# =========================================================
# 2. Feature Extraction (Aligned with rpi_pipeline/feature_extract.py)
# =========================================================
# Data format from sensor:
#   pointCloud: list of [x, y, z, doppler, snr, ...]  (arrays, NOT dicts)
#   trackData:  list of [trackId, x, y, z, vx, vy, vz, ax, ay, az, ...]  (arrays)
#   heightData: list of [trackId, personHeight, bottomHeight]  (arrays)
#
# Feature vector (20-dim):
#   0-2:   centroid x, y, z
#   3-5:   velocity vx, vy, vz (decomposed from doppler)
#   6-8:   acceleration ax, ay, az
#   9:     point count
#   10:    spread_xy
#   11:    height_range
#   12-17: track x, y, z, vx, vy, vz
#   18-19: person height, bottom height

def extract_single_feature_vector(frame_dict, prev_velocity=None):
    """Extract 20-dim feature vector from a single radar frame.
    
    Mirrors rpi_pipeline/feature_extract.py exactly.
    """
    pc = frame_dict.get("pointCloud", [])
    td = frame_dict.get("trackData", [])
    hd = frame_dict.get("heightData", [])

    feats = np.zeros(NUM_FEATURES, dtype=np.float32)
    current_velocity = np.zeros(3, dtype=np.float32)
    
    # --- Point Cloud Features (indices 0-11) ---
    if len(pc) > 0:
        pts = np.array(pc, dtype=np.float32)
        
        # SNR filtering (column 4 if available)
        if pts.shape[1] > 4:
            snr_mask = pts[:, 4] >= SNR_THRESHOLD
            if snr_mask.sum() > 0:
                pts = pts[snr_mask]
        
        n_points = len(pts)
        if n_points > 0:
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            doppler = pts[:, 3] if pts.shape[1] > 3 else np.zeros(n_points)
            
            x_mean, y_mean, z_mean = x.mean(), y.mean(), z.mean()
            
            # Decompose doppler into vx, vy, vz (same as RPi pipeline)
            r = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2) + 1e-8
            vx_mean = doppler.mean() * (x_mean / r)
            vy_mean = doppler.mean() * (y_mean / r)
            vz_mean = doppler.mean() * (z_mean / r)
            current_velocity = np.array([vx_mean, vy_mean, vz_mean], dtype=np.float32)
            
            # Acceleration
            if prev_velocity is not None:
                acc = (current_velocity - prev_velocity) / FRAME_DT
            else:
                acc = np.zeros(3, dtype=np.float32)
            
            spread_xy = float(np.sqrt(x.var() + y.var())) if n_points > 1 else 0.0
            height_range = float(z.max() - z.min()) if n_points > 1 else 0.0
            
            feats[0:3] = [x_mean, y_mean, z_mean]          # Centroid
            feats[3:6] = [vx_mean, vy_mean, vz_mean]       # Velocity
            feats[6:9] = acc                                 # Acceleration
            feats[9]   = float(n_points)                     # Point count
            feats[10]  = spread_xy                           # Spread XY
            feats[11]  = height_range                        # Height range
    
    # --- Track Features (indices 12-17) ---
    if len(td) > 0:
        t = td[0]  # First track: [trackId, x, y, z, vx, vy, vz, ...]
        if len(t) >= 7:
            feats[12:18] = np.array(t[1:7], dtype=np.float32)
    
    # --- Height Features (indices 18-19) ---
    if len(hd) > 0:
        h = hd[0]  # [trackId, personHeight, bottomHeight]
        if len(h) >= 3:
            feats[18] = float(h[1])  # person height
            feats[19] = float(h[2])  # bottom height

    return feats, current_velocity


# =========================================================
# 3. Data Augmentation (Solves "Less Dataset" Problem)
# =========================================================
def augment_window(window, noise_level=0.02):
    """
    Since the dataset is small, we artificially multiply our data 
    by slightly altering the feature vectors.
    """
    augmented = window.copy()
    
    # Apply multiple augmentations with some probability
    # Gaussian noise
    if random.random() < 0.7:
        noise = np.random.normal(0, noise_level, augmented.shape)
        augmented += noise
    
    # Velocity jitter
    if random.random() < 0.5:
        shift_amount = random.uniform(-0.05, 0.05)
        augmented[:, 3:6] += shift_amount  # Velocity components
    
    # Spatial scale
    if random.random() < 0.5:
        scale_factor = random.uniform(0.9, 1.1)
        augmented[:, 0:3] *= scale_factor  # Scale centroid X,Y,Z
        
    # Temporal jitter (randomly drop or duplicate a few frames)
    if random.random() < 0.3:
        idx = random.randint(1, len(augmented) - 2)
        augmented[idx] = (augmented[idx-1] + augmented[idx+1]) / 2
        
    return augmented


# =========================================================
# 4. Dataset Loader
# =========================================================
class MultiClassRadarDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        window = self.X[idx].copy()
        if self.augment:
             window = augment_window(window)
        return torch.tensor(window, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


# =========================================================
# 5. Extract & Windowize Data
# =========================================================
print("Loading and preparing data...")
X_raw, y_raw = [], []

for class_name, class_id in CLASSES.items():
    folder = os.path.join(dataset_path, class_name)
    if not os.path.exists(folder): continue
    
    json_files = glob.glob(os.path.join(folder, "*.json"))
    print(f"  {class_name} (class {class_id}): {len(json_files)} files")
    
    for file in json_files:
        with open(file, 'r') as f:
            data_file = json.load(f)
            
        if "data" not in data_file:
            continue
            
        frames = data_file["data"]
        file_feats = []
        prev_vel = None
        
        for row in frames:
            fd = row.get("frameData", {})
            frame_dict = {
                "pointCloud": fd.get("pointCloud", []),
                "trackData": fd.get("trackData", []),
                "heightData": fd.get("heightData", [])
            }
            feat, prev_vel = extract_single_feature_vector(frame_dict, prev_vel)
            file_feats.append(feat)
        
        # Use a smaller stride for more windows from limited data
        stride = 3
        for i in range(0, len(file_feats) - WINDOW_SIZE + 1, stride):
            window = np.array(file_feats[i : i + WINDOW_SIZE])
            X_raw.append(window)
            y_raw.append(class_id)
            
X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

# Print class distribution
print(f"\nExtracted {len(X_raw)} total windows representing {NUM_CLASSES} classes.")
class_counts = Counter(y_raw.tolist())
for class_name, class_id in CLASSES.items():
    print(f"  {class_name}: {class_counts.get(class_id, 0)} windows")

if len(X_raw) == 0:
    raise ValueError("No data extracted! Check your dataset_path and folder names.")

# =========================================================
# 6. Standardization & Data Splitting
# =========================================================
# Flatten for standard scaler, then reshape back
X_flat = X_raw.reshape(-1, NUM_FEATURES)
scaler = StandardScaler()
X_flat_scaled = scaler.fit_transform(X_flat)
X_scaled = X_flat_scaled.reshape(-1, WINDOW_SIZE, NUM_FEATURES)

# Save scaler for RPi pipeline
with open("multi_class_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

print(f"\nTrain: {len(X_train)} windows, Test: {len(X_test)} windows")

# Compute class weights for imbalanced data
train_counts = Counter(y_train.tolist())
total_train = len(y_train)
class_weights = torch.tensor(
    [total_train / (NUM_CLASSES * train_counts.get(i, 1)) for i in range(NUM_CLASSES)],
    dtype=torch.float32
)
print(f"Class weights: {class_weights.tolist()}")

# Create datasets
train_dataset = MultiClassRadarDataset(X_train, y_train, augment=True)
test_dataset = MultiClassRadarDataset(X_test, y_test, augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# =========================================================
# 7. Model Architecture (Transformer -> CNN -> LSTM)
# =========================================================
class MultiClassTransformerCNNLSTM(nn.Module):
    def __init__(self, num_features=20, num_classes=NUM_CLASSES, d_model=64, nhead=4, 
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
                           batch_first=True, dropout=0.0, num_layers=1)
        
        # Output Stage
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Transformer
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        
        # CNN
        x = x.transpose(1, 2)
        x = self.cnn(x)
        
        # LSTM
        x = x.transpose(1, 2)
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # Classifier
        out = self.fc(last_hidden)
        return out


model = MultiClassTransformerCNNLSTM()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\nModel on: {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# =========================================================
# 8. Training Loop with Early Stopping & LR Scheduling
# =========================================================
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)

EPOCHS = 150
best_val_acc = 0.0
patience = 25
patience_counter = 0

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total if total > 0 else 0
    
    # --- Validation ---
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)
            
    val_acc = val_correct / val_total if val_total > 0 else 0
    
    # Step the scheduler
    scheduler.step()
    
    # Early stopping & best model saving
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "multi_class_model_best.pth")
    else:
        patience_counter += 1
    
    if (epoch+1) % 5 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f} "
              f"- Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f} "
              f"- Best Val: {best_val_acc:.4f} - LR: {lr:.6f}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f}")
        break

# Load best model
model.load_state_dict(torch.load("multi_class_model_best.pth"))

# =========================================================
# 9. Final Evaluation (Per-Class)
# =========================================================
model.eval()
class_correct = Counter()
class_total = Counter()
all_preds, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        
        for pred, label in zip(predicted.cpu().numpy(), y_batch.cpu().numpy()):
            all_preds.append(pred)
            all_labels.append(label)
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

print(f"\n{'='*50}")
print(f"Final Results (Best Model)")
print(f"{'='*50}")
overall_correct = sum(class_correct.values())
overall_total = sum(class_total.values())
print(f"Overall Accuracy: {overall_correct/overall_total:.4f} ({overall_correct}/{overall_total})")
print(f"\nPer-Class Accuracy:")
for class_name, class_id in CLASSES.items():
    total = class_total.get(class_id, 0)
    correct = class_correct.get(class_id, 0)
    acc = correct / total if total > 0 else 0
    print(f"  {class_name}: {acc:.4f} ({correct}/{total})")

print(f"\nDownload 'multi_class_model_best.pth' and 'multi_class_scaler.pkl'!")
