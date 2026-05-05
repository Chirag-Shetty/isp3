"""
evaluate.py
===========
Tests the trained fall detection model directly on the raw,
ORIGINAL recordings inside Dataset1/ subfolders.

Ground truth from folder name:
  fall_stand*, fall_walk*           -> FALL   (label 1)
  sitchair_stand_tr*, sitting_chair,
  stand_sitchair_tr*, standing_still -> NO-FALL (label 0)

Outputs: classification report, confusion matrix, ROC, per-folder
         accuracy, probability distribution -> raw_eval_results.png
         and raw_eval_results.csv
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

_HERE        = Path(__file__).resolve().parent
DATASET_ROOT = _HERE / "Dataset1"  # fall_stand/, sitting_chair/ etc.
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH   = _HERE / "fall_detection_model_best.pth"
SCALER_PATH  = _HERE / "fall_scaler.pkl"
OUTPUT_IMG   = _HERE / "raw_eval_results.png"
OUTPUT_CSV   = _HERE / "raw_eval_results.csv"

WINDOW_SIZE   = 40
SNR_THRESHOLD = 10.0
FRAME_DT      = 0.055

FALL_PREFIXES   = ["fall_stand", "fall_walk"]
NOFALL_PREFIXES = ["sitchair_stand_tr", "sitting_chair",
                   "stand_sitchair_tr", "standing_still"]
FALL_LOOSE_FILES = {"1data.json", "2data.json", "4data.json"}


# ─────────────────────────────────────────────
# FOLDER -> LABEL
# ─────────────────────────────────────────────
def get_label(folder_name: str):
    fl = folder_name.lower()
    for p in FALL_PREFIXES:
        if fl.startswith(p.lower()):
            return 1
    for p in NOFALL_PREFIXES:
        if fl.startswith(p.lower()):
            return 0
    return None


# ─────────────────────────────────────────────
# MODEL (must match training exactly)
# ─────────────────────────────────────────────
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
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (h, _) = self.lstm(x)
        return self.classifier(h.squeeze(0))


# ─────────────────────────────────────────────
# FEATURE EXTRACTION (identical to training)
# ─────────────────────────────────────────────
def extract_features(frame_data: dict, prev_feat=None) -> np.ndarray:
    pts = np.array(frame_data.get("pointCloud", []))
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

    track = frame_data.get("trackData", [])
    if len(track) > 0:
        t = track[0]
        for i, idx in enumerate(range(12, 18)):
            feat[idx] = t[i+1] if len(t) > i+1 else 0.0

    height = frame_data.get("heightData", [])
    if len(height) > 0:
        h = height[0]
        feat[18] = h[1] if len(h) > 1 else 0.0
        feat[19] = h[2] if len(h) > 2 else 0.0
    return feat


def load_frames(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    elif isinstance(raw, list):
        return raw
    elif isinstance(raw, dict) and "frameData" in raw:
        return [raw]
    return []


def file_to_windows(path: Path, scaler):
    """Extract features -> scale -> non-overlapping windows."""
    frames = load_frames(path)
    if len(frames) < WINDOW_SIZE:
        return None

    feats, prev = [], None
    for frame in frames:
        fd   = frame.get("frameData", frame)
        feat = extract_features(fd, prev)
        feats.append(feat)
        prev = feat

    feats  = np.array(feats, dtype=np.float32)
    scaled = scaler.transform(feats.reshape(-1, 20)).reshape(feats.shape)

    windows = []
    for start in range(0, len(scaled) - WINDOW_SIZE + 1, WINDOW_SIZE):
        windows.append(scaled[start:start + WINDOW_SIZE])
    return np.array(windows, dtype=np.float32) if windows else None


# ─────────────────────────────────────────────
# PREDICT ONE FILE
# ─────────────────────────────────────────────
def predict_file(path: Path, scaler, model, device):
    windows = file_to_windows(path, scaler)
    if windows is None:
        return None, None, None

    x = torch.FloatTensor(windows).to(device)
    with torch.no_grad():
        out   = model(x)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()

    file_pred = int(np.round(np.mean(preds)))
    file_prob = float(np.mean(probs))
    return file_pred, file_prob, len(windows)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found -> {MODEL_PATH}")
        print("  Run run_fall_detection.py first.")
        return

    model = FallDetectionTransformerCNNLSTM().to(device)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model.eval()
    print(f"Model  : {MODEL_PATH.name}")

    with open(str(SCALER_PATH), "rb") as f:
        scaler = pickle.load(f)
    print(f"Scaler : {SCALER_PATH.name}")
    print(f"Data   : {DATASET_ROOT}\n")

    records = []
    skipped = []

    # --- Subfolders ---
    for folder in sorted(DATASET_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        label = get_label(folder.name)
        if label is None:
            print(f"  [SKIP] {folder.name}")
            continue

        label_name = "FALL" if label == 1 else "NO-FALL"
        json_files = sorted(folder.glob("*.json"))
        if not json_files:
            continue

        print(f"  [{folder.name}] -> {label_name}  ({len(json_files)} files)")

        for jf in json_files:
            pred, prob, nw = predict_file(jf, scaler, model, device)
            if pred is None:
                skipped.append(jf.name)
                print(f"    SKIP {jf.name} (< {WINDOW_SIZE} frames)")
                continue

            correct   = pred == label
            pred_name = "FALL" if pred == 1 else "NO-FALL"
            marker    = "OK" if correct else "WRONG"
            print(f"    {jf.name:30s}  pred={pred_name:7s}  P(FALL)={prob:.3f}  [{marker}]")

            records.append({
                "folder":     folder.name,
                "file":       jf.name,
                "true_label": label,
                "true_name":  label_name,
                "pred_label": pred,
                "pred_name":  pred_name,
                "prob_fall":  round(prob, 4),
                "correct":    correct,
                "n_windows":  nw,
            })

    # --- Loose FALL files in root ---
    for jf in sorted(DATASET_ROOT.glob("*.json")):
        if jf.name not in FALL_LOOSE_FILES:
            continue
        print(f"\n  [loose] {jf.name} -> FALL")
        pred, prob, nw = predict_file(jf, scaler, model, device)
        if pred is None:
            skipped.append(jf.name)
            continue
        pred_name = "FALL" if pred == 1 else "NO-FALL"
        correct   = pred == 1
        print(f"    pred={pred_name}  P(FALL)={prob:.3f}  [{'OK' if correct else 'WRONG'}]")
        records.append({
            "folder": "(root)", "file": jf.name,
            "true_label": 1, "true_name": "FALL",
            "pred_label": pred, "pred_name": pred_name,
            "prob_fall": round(prob, 4),
            "correct": correct, "n_windows": nw,
        })

    if not records:
        print("\nNo files evaluated. Check paths.")
        return

    # --- Results ---
    df        = pd.DataFrame(records)
    all_true  = df["true_label"].values
    all_preds = df["pred_label"].values
    all_probs = df["prob_fall"].values

    print("\n" + "=" * 60)
    print("  OVERALL RESULTS ON RAW ORIGINAL DATA")
    print("=" * 60)
    print(classification_report(all_true, all_preds,
                                 target_names=["NO-FALL", "FALL"]))

    auc         = roc_auc_score(all_true, all_probs)
    overall_acc = np.mean(all_preds == all_true)
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  Overall Accuracy: {overall_acc:.4f}  ({overall_acc*100:.1f}%)")
    print(f"  Files evaluated : {len(df)}")
    if skipped:
        print(f"  Files skipped   : {len(skipped)} (too short)")

    # Per-folder
    print("\n" + "=" * 60)
    print("  PER-FOLDER RESULTS")
    print("=" * 60)
    print(f"  {'Folder':<30} {'Label':>7} {'Acc':>6} {'OK':>4}/{'>Total'}")
    print(f"  {'-'*30} {'-'*7} {'-'*6} {'-'*10}")

    folder_stats = []
    for fname, grp in df.groupby("folder"):
        acc   = grp["correct"].mean()
        ok    = int(grp["correct"].sum())
        tot   = len(grp)
        lname = grp["true_name"].iloc[0]
        flag  = "  <- WEAK" if acc < 0.80 else ""
        folder_stats.append((fname, lname, acc, ok, tot))
        print(f"  {fname:<30} {lname:>7} {acc:>6.2f} {ok:>4}/{tot}{flag}")

    # Save CSV
    df.to_csv(str(OUTPUT_CSV), index=False)
    print(f"\n  Results CSV: {OUTPUT_CSV.name}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Fall Detection - Raw Original Data\n"
        f"Accuracy: {overall_acc*100:.1f}%  |  AUC: {auc:.3f}  |  Files: {len(df)}",
        fontsize=13, fontweight='bold')

    # 1. Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues',
                xticklabels=["NO-FALL", "FALL"],
                yticklabels=["NO-FALL", "FALL"])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(all_true, all_probs)
    axes[0, 1].plot(fpr, tpr, color='crimson', lw=2, label=f'AUC = {auc:.3f}')
    axes[0, 1].plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    axes[0, 1].fill_between(fpr, tpr, alpha=0.1, color='crimson')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Per-folder accuracy
    f_names = [x[0] for x in folder_stats]
    f_labs  = [x[1] for x in folder_stats]
    f_accs  = [x[2] for x in folder_stats]
    colors  = ['#ef4444' if a < 0.80 else '#22c55e' for a in f_accs]
    edge_c  = ['#7f1d1d' if l == 'FALL' else '#1e3a5f' for l in f_labs]
    bars = axes[1, 0].barh(f_names, f_accs, color=colors, edgecolor=edge_c, linewidth=1.2)
    axes[1, 0].axvline(x=0.80, color='orange', linestyle='--', lw=1.2, label='80% threshold')
    axes[1, 0].axvline(x=overall_acc, color='blue', linestyle='--', lw=1.2,
                        label=f'Overall ({overall_acc:.2f})')
    axes[1, 0].set_xlim(0, 1.10)
    axes[1, 0].set_xlabel('Accuracy')
    axes[1, 0].set_title('Per-Folder Accuracy\n(red border=FALL, blue=NO-FALL)')
    axes[1, 0].legend(fontsize=8)
    for bar, acc in zip(bars, f_accs):
        axes[1, 0].text(bar.get_width() + 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f'{acc:.2f}', va='center', fontsize=8)

    # 4. P(FALL) distribution
    fall_p   = all_probs[all_true == 1]
    nofall_p = all_probs[all_true == 0]
    bins = np.linspace(0, 1, 21)
    axes[1, 1].hist(nofall_p, bins=bins, alpha=0.65, color='steelblue',
                     label='NO-FALL', density=True)
    axes[1, 1].hist(fall_p, bins=bins, alpha=0.65, color='crimson',
                     label='FALL', density=True)
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='threshold=0.5')
    axes[1, 1].set_xlabel('P(FALL)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUTPUT_IMG), dpi=150, bbox_inches='tight')
    print(f"  Plot saved   : {OUTPUT_IMG.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()