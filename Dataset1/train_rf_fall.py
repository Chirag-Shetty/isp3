"""
train_rf_fall.py
================
Trains a Random Forest (+ optional SVM/GBM comparison) fall detector
using the EXACT SAME feature extraction formula as rpi_pipeline/feature_extract.py.

No deep learning. No scaler mismatch. Works with small data.

Run from: Dataset1/
    python train_rf_fall.py

Outputs:
    fall_rf_model.pkl   <- drop in hf_space_repo/ and rpi_pipeline/
"""

import sys, json, pickle, warnings
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT     = Path(__file__).resolve().parent / "Dataset1"
WINDOW_SIZE   = 40
STRIDE        = 5
SNR_THRESHOLD = 10.0
FRAME_DT      = 0.055
OUTPUT_MODEL  = Path(__file__).resolve().parent / "fall_rf_model.pkl"

FALL_PREFIXES    = ("fall",)
NOFALL_PREFIXES  = ("sit", "stand", "standing")

# ── IDENTICAL to rpi_pipeline/feature_extract.py ──────────────────────────────
def extract_frame_features(point_cloud, track_data, height_data, prev_velocity=None):
    if not point_cloud or len(point_cloud) == 0:
        pc_feat = np.zeros(12, dtype=np.float32)
        current_velocity = np.zeros(3, dtype=np.float32)
    else:
        pts = np.array(point_cloud, dtype=np.float32)
        if pts.shape[1] > 4:
            snr_mask = pts[:, 4] >= SNR_THRESHOLD
            if snr_mask.sum() > 0:
                pts = pts[snr_mask]

        n_points = len(pts)
        if n_points == 0:
            pc_feat = np.zeros(12, dtype=np.float32)
            current_velocity = np.zeros(3, dtype=np.float32)
        else:
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            doppler = pts[:, 3] if pts.shape[1] > 3 else np.zeros(n_points)
            x_mean, y_mean, z_mean = x.mean(), y.mean(), z.mean()
            r = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2) + 1e-8
            vx_mean = doppler.mean() * (x_mean / r)
            vy_mean = doppler.mean() * (y_mean / r)
            vz_mean = doppler.mean() * (z_mean / r)
            current_velocity = np.array([vx_mean, vy_mean, vz_mean], dtype=np.float32)
            acc = (current_velocity - prev_velocity) / FRAME_DT if prev_velocity is not None else np.zeros(3)
            spread_xy    = float(np.sqrt(x.var() + y.var())) if n_points > 1 else 0.0
            height_range = float(z.max() - z.min())         if n_points > 1 else 0.0
            pc_feat = np.array([
                x_mean, y_mean, z_mean, vx_mean, vy_mean, vz_mean,
                acc[0], acc[1], acc[2], float(n_points), spread_xy, height_range
            ], dtype=np.float32)

    track_feat = np.zeros(6, dtype=np.float32)
    if track_data and len(track_data) > 0:
        td = track_data[0]
        if len(td) >= 7:
            track_feat = np.array(td[1:7], dtype=np.float32)

    height_feat = np.zeros(2, dtype=np.float32)
    if height_data and len(height_data) > 0:
        hd = height_data[0]
        if len(hd) >= 3:
            height_feat = np.array(hd[1:3], dtype=np.float32)

    feat = np.concatenate([pc_feat, track_feat, height_feat])
    return feat, current_velocity


def window_to_features(window: np.ndarray) -> np.ndarray:
    """
    Compress (40, 20) window → 1-D feature vector for classical ML.

    Features per-column (20 cols × 6 stats = 120):
        mean, std, min, max, (max-min), linear slope

    Plus 4 physics-motivated extras:
        z_slope        : slope of centroid height → negative = falling
        height_slope   : slope of height_range
        peak_vz        : max |vz| in window → vertical velocity spike
        peak_az        : max |az| in window → vertical acceleration spike

    Total: 124
    """
    T, F = window.shape
    xs = np.arange(T, dtype=np.float32)
    feats = []

    for col in range(F):
        v = window[:, col]
        feats += [v.mean(), v.std(), v.min(), v.max(), v.max()-v.min()]
        # linear slope via polyfit
        slope = np.polyfit(xs, v, 1)[0] if v.std() > 1e-8 else 0.0
        feats.append(float(slope))

    # Physics extras
    z_col   = window[:, 2]   # z centroid
    hr_col  = window[:, 11]  # height range
    vz_col  = window[:, 5]   # vz
    az_col  = window[:, 8]   # az

    feats.append(float(np.polyfit(xs, z_col,  1)[0]))   # z_slope
    feats.append(float(np.polyfit(xs, hr_col, 1)[0]))   # height_slope
    feats.append(float(np.max(np.abs(vz_col))))          # peak_vz
    feats.append(float(np.max(np.abs(az_col))))          # peak_az

    return np.array(feats, dtype=np.float32)


def load_file(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'data' in raw:
        rows = raw['data']
        return [{'pointCloud': r.get('frameData',{}).get('pointCloud',[]),
                 'trackData':  r.get('frameData',{}).get('trackData',[]),
                 'heightData': r.get('frameData',{}).get('heightData',[])}
                for r in rows]
    elif isinstance(raw, list):
        return [{'pointCloud': r.get('frameData',r).get('pointCloud',[]),
                 'trackData':  r.get('frameData',r).get('trackData',[]),
                 'heightData': r.get('frameData',r).get('heightData',[])}
                for r in raw]
    return []


def get_label(folder_name: str):
    n = folder_name.lower()
    if n.startswith(FALL_PREFIXES):
        return 1
    if n.startswith(NOFALL_PREFIXES):
        return 0
    return None


def extract_windows(folder: Path, label: int):
    windows, labels = [], []
    for jf in sorted(folder.glob("*.json")):
        frames = load_file(jf)
        if len(frames) < WINDOW_SIZE:
            continue
        # extract features for every frame
        buf, prev = [], None
        for fr in frames:
            feat, prev = extract_frame_features(
                fr['pointCloud'], fr['trackData'], fr['heightData'], prev)
            buf.append(feat)

        # slide windows
        for start in range(0, len(buf) - WINDOW_SIZE + 1, STRIDE):
            win = np.array(buf[start:start+WINDOW_SIZE], dtype=np.float32)
            windows.append(window_to_features(win))
            labels.append(label)
    return windows, labels


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Binary Fall Detector — Random Forest Training")
    print("=" * 60)

    X_all, y_all = [], []
    folder_stats = []

    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        label = get_label(folder.name)
        if label is None:
            print(f"  [SKIP] {folder.name}")
            continue
        wins, labs = extract_windows(folder, label)
        X_all.extend(wins)
        y_all.extend(labs)
        folder_stats.append((folder.name, label, len(wins)))
        tag = "FALL" if label == 1 else "NO-FALL"
        print(f"  {folder.name:<25} {tag:<8}  {len(wins)} windows")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    print(f"\n  Total windows : {len(X)}  (FALL={sum(y==1)}, NO-FALL={sum(y==0)})")
    print(f"  Feature dims  : {X.shape[1]}")

    # ── Candidate models ──────────────────────────────────────────────────────
    candidates = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=500, max_depth=None,
                class_weight="balanced", random_state=42, n_jobs=-1))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42))
        ]),
        "SVM-RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=5, gamma="scale",
                          class_weight="balanced", probability=True, random_state=42))
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n  5-fold cross-validation:")
    print(f"  {'Model':<20} {'Accuracy':>10} {'F1-FALL':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    best_name, best_score, best_model = None, -1, None
    for name, model in candidates.items():
        acc   = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
        f1    = cross_val_score(model, X, y, cv=cv, scoring="f1").mean()
        auc   = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
        print(f"  {name:<20} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")
        if auc > best_score:
            best_score, best_name, best_model = auc, name, model

    # ── Train best model on ALL data ──────────────────────────────────────────
    print(f"\n  Best model: {best_name} (ROC-AUC={best_score:.4f})")
    best_model.fit(X, y)

    # Final report on training set (sanity check)
    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1]
    print("\n  Training-set report (sanity check):")
    print(classification_report(y, y_pred, target_names=["NO-FALL", "FALL"]))
    print(f"  ROC-AUC on full train: {roc_auc_score(y, y_prob):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "model":          best_model,
        "model_name":     best_name,
        "window_size":    WINDOW_SIZE,
        "n_features_raw": 20,
        "n_features_ml":  X.shape[1],
        "classes":        ["NO-FALL", "FALL"],
        "cv_roc_auc":     best_score,
    }
    with open(str(OUTPUT_MODEL), "wb") as f:
        pickle.dump(payload, f)

    print(f"\n  Saved: {OUTPUT_MODEL.name}")
    print("  Copy this file to hf_space_repo/ and rpi_pipeline/")
    print("=" * 60)


if __name__ == "__main__":
    main()
