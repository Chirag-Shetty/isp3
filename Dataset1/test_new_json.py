"""
test_new_json.py
================
Runs every JSON file in new_test_json/ through the trained
fall detection model and prints predictions.

No ground-truth labels needed — just raw predictions.
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
TEST_DIR     = _HERE / "new_test_json"
MODEL_PATH   = _HERE / "fall_detection_model_best.pth"
SCALER_PATH  = _HERE / "fall_scaler.pkl"
OUTPUT_IMG   = _HERE / "new_test_results.png"

WINDOW_SIZE   = 40
SNR_THRESHOLD = 10.0
FRAME_DT      = 0.055
FALL_THRESHOLD = 0.5   # P(FALL) >= this -> predicted FALL


# ── Model (identical to training) ─────────────────────────────────────────────
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


# ── Feature extraction (identical to training) ────────────────────────────────
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


def predict_file(path: Path, scaler, model, device):
    """Returns (pred_label, prob_fall, n_frames, n_windows, per_window_probs)."""
    frames = load_frames(path)
    n_frames = len(frames)

    if n_frames < WINDOW_SIZE:
        return None, None, n_frames, 0, []

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

    if not windows:
        return None, None, n_frames, 0, []

    x = torch.FloatTensor(np.array(windows)).to(device)
    with torch.no_grad():
        out   = model(x)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

    file_prob = float(np.mean(probs))
    file_pred = 1 if file_prob >= FALL_THRESHOLD else 0
    return file_pred, file_prob, n_frames, len(windows), probs.tolist()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model = FallDetectionTransformerCNNLSTM().to(device)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model.eval()
    print(f"Model  : {MODEL_PATH.name}")

    with open(str(SCALER_PATH), "rb") as f:
        scaler = pickle.load(f)
    print(f"Scaler : {SCALER_PATH.name}")
    print(f"Folder : {TEST_DIR}\n")

    json_files = sorted(TEST_DIR.glob("*.json"))
    if not json_files:
        print("ERROR: No JSON files found in new_test_json/")
        return

    results = []
    print("=" * 65)
    print(f"  {'File':<20} {'Frames':>7} {'Windows':>8} {'P(FALL)':>9}  Prediction")
    print(f"  {'-'*20} {'-'*7} {'-'*8} {'-'*9}  ----------")

    for jf in json_files:
        pred, prob, nf, nw, win_probs = predict_file(jf, scaler, model, device)

        if pred is None:
            print(f"  {jf.name:<20} {nf:>7}   SKIP (< {WINDOW_SIZE} frames)")
            continue

        label = "*** FALL ***" if pred == 1 else "NO-FALL"
        bar   = "#" * int(prob * 20)
        print(f"  {jf.name:<20} {nf:>7} {nw:>8} {prob:>9.4f}  {label}")
        print(f"  {'':20}  [{bar:<20}] {prob*100:.1f}%")

        results.append({
            "file":       jf.name,
            "n_frames":   nf,
            "n_windows":  nw,
            "prob_fall":  round(prob, 4),
            "prediction": "FALL" if pred == 1 else "NO-FALL",
            "win_probs":  win_probs,
        })

    if not results:
        print("No files could be evaluated.")
        return

    # Summary
    fall_count   = sum(1 for r in results if r["prediction"] == "FALL")
    nofall_count = len(results) - fall_count
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Total files  : {len(results)}")
    print(f"  FALL         : {fall_count}")
    print(f"  NO-FALL      : {nofall_count}")
    avg_prob = np.mean([r["prob_fall"] for r in results])
    print(f"  Avg P(FALL)  : {avg_prob:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fall Detection — new_test_json Predictions", fontsize=13, fontweight='bold')

    # 1. P(FALL) bar chart per file
    names  = [r["file"].replace(".json", "") for r in results]
    probs  = [r["prob_fall"] for r in results]
    colors = ["#ef4444" if p >= FALL_THRESHOLD else "#22c55e" for p in probs]

    bars = ax1.bar(names, probs, color=colors, edgecolor='white', linewidth=0.5)
    ax1.axhline(y=FALL_THRESHOLD, color='black', linestyle='--', lw=1.5,
                label=f'Threshold = {FALL_THRESHOLD}')
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("P(FALL)")
    ax1.set_title("Per-File Fall Probability\n(red = FALL, green = NO-FALL)")
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    for bar, p in zip(bars, probs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{p:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 2. Window-level probability timeline for each file
    ax2.set_title("Window-Level P(FALL) per File")
    ax2.set_xlabel("Window index")
    ax2.set_ylabel("P(FALL)")
    ax2.set_ylim(-0.05, 1.10)
    ax2.axhline(y=FALL_THRESHOLD, color='black', linestyle='--', lw=1, alpha=0.5)
    colors_line = ["#ef4444", "#f97316", "#3b82f6", "#8b5cf6",
                   "#22c55e", "#ec4899", "#14b8a6", "#f59e0b"]
    for i, r in enumerate(results):
        if r["win_probs"]:
            ax2.plot(r["win_probs"],
                     marker='o', markersize=4, linewidth=1.5,
                     color=colors_line[i % len(colors_line)],
                     label=r["file"].replace(".json", ""))
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUTPUT_IMG), dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {OUTPUT_IMG.name}")
    print("Done.")


if __name__ == "__main__":
    main()
