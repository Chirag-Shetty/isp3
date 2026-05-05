"""
Binary Fall Detection — Training Script
Equivalent to train_fall_detection.ipynb
Reads CLAUDE_binary.md spec and executes it.
"""
# ── Cell 1: Imports ───────────────────────────────────────────────────────────
import json, os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
print('All imports OK')

# ── Cell 2: Config ────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
MANIFEST_CSV = str(_SCRIPT_DIR / 'aug_dataset_binary' / 'dataset_manifest.csv')

WINDOW_SIZE   = 40
STRIDE        = 3
FRAME_DT      = 0.055
SNR_THRESHOLD = 10.0
BATCH_SIZE    = 16
MAX_EPOCHS    = 150
LR            = 0.001
WEIGHT_DECAY  = 0.001
PATIENCE      = 25
GRAD_CLIP     = 1.0
NUM_CLASSES   = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device  : {device}')
print(f'Manifest: {MANIFEST_CSV}')

# ── Cell 3: Feature extraction ────────────────────────────────────────────────
def extract_features(frame_data: dict, prev_feat=None) -> np.ndarray:
    pts = np.array(frame_data.get('pointCloud', []))
    if len(pts) > 0:
        pts = pts[pts[:, 4] >= SNR_THRESHOLD]

    feat = np.zeros(20)

    if len(pts) > 0:
        feat[0] = np.mean(pts[:, 0])
        feat[1] = np.mean(pts[:, 1])
        feat[2] = np.mean(pts[:, 2])

        angles = np.arctan2(pts[:, 0], pts[:, 1])
        elev   = np.arctan2(pts[:, 2], np.sqrt(pts[:,0]**2 + pts[:,1]**2))
        feat[3] = np.mean(pts[:, 3] * np.sin(angles))
        feat[4] = np.mean(pts[:, 3] * np.cos(angles))
        feat[5] = np.mean(pts[:, 3] * np.sin(elev))

        if prev_feat is not None:
            feat[6] = (feat[3] - prev_feat[3]) / FRAME_DT
            feat[7] = (feat[4] - prev_feat[4]) / FRAME_DT
            feat[8] = (feat[5] - prev_feat[5]) / FRAME_DT

        feat[9]  = len(pts)
        feat[10] = np.sqrt(np.var(pts[:,0]) + np.var(pts[:,1]))
        feat[11] = np.max(pts[:,2]) - np.min(pts[:,2])

    track = frame_data.get('trackData', [])
    if len(track) > 0:
        t = track[0]
        for i, idx in enumerate(range(12, 18)):
            feat[idx] = t[i+1] if len(t) > i+1 else 0.0

    height = frame_data.get('heightData', [])
    if len(height) > 0:
        h = height[0]
        feat[18] = h[1] if len(h) > 1 else 0.0
        feat[19] = h[2] if len(h) > 2 else 0.0

    return feat


def make_windows(features: np.ndarray, label: int):
    windows, labels = [], []
    for start in range(0, len(features) - WINDOW_SIZE + 1, STRIDE):
        windows.append(features[start:start + WINDOW_SIZE])
        labels.append(label)
    return windows, labels

print('Feature functions defined.')

# ── Cell 4: Data loading ──────────────────────────────────────────────────────
def load_frames(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'data' in raw:
        return raw['data']
    elif isinstance(raw, list):
        return raw
    elif isinstance(raw, dict) and 'frameData' in raw:
        return [raw]
    return []


def build_dataset(manifest_csv: str):
    manifest = pd.read_csv(manifest_csv)
    all_windows, all_labels = [], []
    skipped = 0

    for _, row in manifest.iterrows():
        label = int(row['label'])

        fpath = Path(row['file'])
        if not fpath.exists():
            # try relative to manifest directory
            fpath = Path(manifest_csv).parent / fpath.name
        if not fpath.exists():
            # reconstruct from label folder
            sub = 'class_1_fall' if label == 1 else 'class_0_nofall'
            fpath = Path(manifest_csv).parent / sub / Path(row['file']).name
        if not fpath.exists():
            skipped += 1
            continue

        try:
            frames = load_frames(fpath)
        except Exception:
            skipped += 1
            continue

        if len(frames) == 0:
            skipped += 1
            continue

        feats, prev = [], None
        for frame in frames:
            fd   = frame.get('frameData', frame)
            feat = extract_features(fd, prev)
            feats.append(feat)
            prev = feat

        feats = np.array(feats)
        if len(feats) < WINDOW_SIZE:
            skipped += 1
            continue

        wins, labs = make_windows(feats, label)
        all_windows.extend(wins)
        all_labels.extend(labs)

    print(f'Skipped {skipped} files (too short, missing, or empty)')
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels,  dtype=np.int64)
    return X, y

# ── Cell 5: Run build_dataset ─────────────────────────────────────────────────
print('\n--- Loading dataset ---')
X, y = build_dataset(MANIFEST_CSV)
print(f'Dataset shape: X={X.shape}, y={y.shape}')

nofall_count = int((y == 0).sum())
fall_count   = int((y == 1).sum())
ratio = nofall_count / fall_count if fall_count > 0 else float('inf')
print(f'\nNO-FALL windows : {nofall_count}')
print(f'FALL windows    : {fall_count}')
print(f'Balance ratio   : {ratio:.1f}x')

# ── Cell 6: Split + Scaler ────────────────────────────────────────────────────
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.176, stratify=y_tv, random_state=42)

print(f'\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.reshape(-1, 20)).reshape(X_train.shape)
X_val_s   = scaler.transform(X_val.reshape(-1, 20)).reshape(X_val.shape)
X_test_s  = scaler.transform(X_test.reshape(-1, 20)).reshape(X_test.shape)

with open('fall_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print('Scaler saved -> fall_scaler.pkl')

# ── Cell 7: DataLoaders ───────────────────────────────────────────────────────
def make_loader(X_np, y_np, shuffle=False):
    ds = TensorDataset(torch.FloatTensor(X_np), torch.LongTensor(y_np))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train_s, y_train, shuffle=True)
val_loader   = make_loader(X_val_s,   y_val)
test_loader  = make_loader(X_test_s,  y_test)
print(f'Loaders: train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)}')

# ── Cell 8: Model ─────────────────────────────────────────────────────────────
class FallDetectionTransformerCNNLSTM(nn.Module):
    def __init__(self, input_size=20, d_model=64, nhead=4,
                 num_transformer_layers=2, cnn_channels=32,
                 lstm_hidden=64, dropout=0.4):
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
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2))

    def forward(self, x):
        x = self.input_proj(x)          # (B, 40, 64)
        x = self.transformer(x)          # (B, 40, 64)
        x = x.transpose(1, 2)           # (B, 64, 40)
        x = self.cnn(x)                 # (B, 32, 20)
        x = x.transpose(1, 2)           # (B, 20, 32)
        _, (h, _) = self.lstm(x)        # h: (1, B, 64)
        return self.classifier(h.squeeze(0))   # (B, 2)


model = FallDetectionTransformerCNNLSTM().to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nModel parameters: {n_params:,}')

# ── Cell 9: Training loop ─────────────────────────────────────────────────────
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
criterion = nn.CrossEntropyLoss(
    weight=torch.FloatTensor(cw).to(device), label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

best_val_loss    = float('inf')
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

print('\n--- Training ---')
for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            val_loss += criterion(out, yb).item()
            correct  += (out.argmax(1) == yb).sum().item()
            total    += len(yb)

    val_acc = correct / total
    tl = train_loss / len(train_loader)
    vl = val_loss   / len(val_loader)
    history['train_loss'].append(tl)
    history['val_loss'].append(vl)
    history['val_acc'].append(val_acc)
    scheduler.step()

    if vl < best_val_loss:
        best_val_loss    = vl
        patience_counter = 0
        torch.save(model.state_dict(), 'fall_detection_model_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1:3d} | Train: {tl:.4f} '
              f'| Val: {vl:.4f} | Acc: {val_acc:.4f}')

print(f'\nBest val loss: {best_val_loss:.4f}')
print('Best model saved -> fall_detection_model_best.pth')

# ── Cell 10: Training curves ──────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'],   label='Val')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], color='green', label='Val Accuracy')
plt.title('Validation Accuracy'); plt.xlabel('Epoch')
plt.ylim(0, 1); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.close()
print('Saved training_curves.png')

# ── Cell 11: Evaluation ───────────────────────────────────────────────────────
model.load_state_dict(torch.load('fall_detection_model_best.pth', map_location=device))
model.eval()

all_preds, all_true, all_probs = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out   = model(xb)
        probs = torch.softmax(out, dim=1)[:, 1]
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_true.extend(yb.numpy())
        all_probs.extend(probs.cpu().numpy())

print('\n--- Test Set Report ---')
print(classification_report(all_true, all_preds, target_names=['NO-FALL', 'FALL']))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

cm = confusion_matrix(all_true, all_preds)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues',
            xticklabels=['NO-FALL', 'FALL'], yticklabels=['NO-FALL', 'FALL'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

fpr, tpr, _ = roc_curve(all_true, all_probs)
auc = roc_auc_score(all_true, all_probs)
axes[1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
axes[1].plot([0, 1], [0, 1], '--', color='gray')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate (Recall)')
axes[1].set_title('ROC Curve')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation.png', dpi=150)
plt.close()
print('Saved evaluation.png')

# ── Cell 12: Output file sizes ────────────────────────────────────────────────
print('\n=== Output Files ===')
for fname in ['fall_detection_model_best.pth', 'fall_scaler.pkl',
              'evaluation.png', 'training_curves.png']:
    if os.path.exists(fname):
        print(f'  {fname}: {os.path.getsize(fname)/1024:.1f} KB')
    else:
        print(f'  {fname}: NOT FOUND')
print('\nDone! FALL detected when model(x).argmax() == 1')
