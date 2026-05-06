"""
train_extra_models.py
=====================
Trains multiple classical ML models for binary fall detection
and saves each to Dataset1/extra_models/

Models trained:
  1. Random Forest         (already prod model)
  2. Gradient Boosting
  3. Extra Trees
  4. AdaBoost
  5. SVM (RBF kernel)
  6. SVM (Linear kernel)
  7. K-Nearest Neighbours
  8. Logistic Regression
  9. XGBoost               (if installed)
  10. LightGBM             (if installed)

Run from Dataset1/:
    python train_extra_models.py

Outputs: extra_models/<model_name>.pkl + extra_models/comparison_report.csv
"""

import json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path(__file__).resolve().parent / "Dataset1"
OUT_DIR      = Path(__file__).resolve().parent / "extra_models"
OUT_DIR.mkdir(exist_ok=True)

WINDOW_SIZE   = 40
STRIDE        = 5
SNR_THRESHOLD = 10.0
FRAME_DT      = 0.055

FALL_PREFIXES   = ("fall",)
NOFALL_PREFIXES = ("sit", "stand", "standing")

# ── Feature extraction (IDENTICAL to rpi_pipeline/feature_extract.py) ─────────
def extract_frame_features(point_cloud, track_data, height_data, prev_velocity=None):
    if not point_cloud or len(point_cloud) == 0:
        return np.zeros(20, dtype=np.float32), np.zeros(3, dtype=np.float32)

    pts = np.array(point_cloud, dtype=np.float32)
    if pts.shape[1] > 4:
        snr_mask = pts[:, 4] >= SNR_THRESHOLD
        if snr_mask.sum() > 0:
            pts = pts[snr_mask]

    if len(pts) == 0:
        return np.zeros(20, dtype=np.float32), np.zeros(3, dtype=np.float32)

    x, y, z   = pts[:, 0], pts[:, 1], pts[:, 2]
    doppler    = pts[:, 3] if pts.shape[1] > 3 else np.zeros(len(pts))
    x_mean, y_mean, z_mean = x.mean(), y.mean(), z.mean()
    r          = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2) + 1e-8
    vx, vy, vz = doppler.mean()*(x_mean/r), doppler.mean()*(y_mean/r), doppler.mean()*(z_mean/r)
    cur_vel    = np.array([vx, vy, vz], dtype=np.float32)
    acc        = (cur_vel - prev_velocity)/FRAME_DT if prev_velocity is not None else np.zeros(3)
    spread_xy  = float(np.sqrt(x.var()+y.var())) if len(pts)>1 else 0.0
    h_range    = float(z.max()-z.min())          if len(pts)>1 else 0.0

    pc_feat = np.array([x_mean, y_mean, z_mean, vx, vy, vz,
                        acc[0], acc[1], acc[2], float(len(pts)), spread_xy, h_range],
                       dtype=np.float32)

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

    return np.concatenate([pc_feat, track_feat, height_feat]), cur_vel


def window_to_features(window: np.ndarray) -> np.ndarray:
    """(40,20) -> 124-dim summary vector. MUST match train_rf_fall.py."""
    T, F = window.shape
    xs   = np.arange(T, dtype=np.float32)
    feats = []
    for col in range(F):
        v = window[:, col].astype(np.float32)
        feats += [v.mean(), v.std(), v.min(), v.max(), v.max()-v.min()]
        feats.append(float(np.polyfit(xs, v, 1)[0]) if v.std() > 1e-8 else 0.0)
    feats.append(float(np.polyfit(xs, window[:,  2], 1)[0]))  # z_slope
    feats.append(float(np.polyfit(xs, window[:, 11], 1)[0]))  # height_slope
    feats.append(float(np.max(np.abs(window[:, 5]))))          # peak_vz
    feats.append(float(np.max(np.abs(window[:, 8]))))          # peak_az
    return np.array(feats, dtype=np.float32)


def load_file(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'data' in raw:
        return [{'pointCloud': r.get('frameData',{}).get('pointCloud',[]),
                 'trackData':  r.get('frameData',{}).get('trackData',[]),
                 'heightData': r.get('frameData',{}).get('heightData',[])}
                for r in raw['data']]
    elif isinstance(raw, list):
        return [{'pointCloud': r.get('frameData',r).get('pointCloud',[]),
                 'trackData':  r.get('frameData',r).get('trackData',[]),
                 'heightData': r.get('frameData',r).get('heightData',[])}
                for r in raw]
    return []


def get_label(folder_name):
    n = folder_name.lower()
    if n.startswith(FALL_PREFIXES):   return 1
    if n.startswith(NOFALL_PREFIXES): return 0
    return None


def build_dataset():
    X_all, y_all = [], []
    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir(): continue
        label = get_label(folder.name)
        if label is None: continue
        for jf in sorted(folder.glob("*.json")):
            frames = load_file(jf)
            if len(frames) < WINDOW_SIZE: continue
            buf, prev = [], None
            for fr in frames:
                feat, prev = extract_frame_features(
                    fr['pointCloud'], fr['trackData'], fr['heightData'], prev)
                buf.append(feat)
            for start in range(0, len(buf)-WINDOW_SIZE+1, STRIDE):
                win = np.array(buf[start:start+WINDOW_SIZE], dtype=np.float32)
                X_all.append(window_to_features(win))
                y_all.append(label)
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.int32)


# ── MODEL DEFINITIONS ─────────────────────────────────────────────────────────
def get_candidates():
    cands = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                           random_state=42, n_jobs=-1))]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                                         random_state=42, n_jobs=-1))]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                               learning_rate=0.05, subsample=0.8,
                                               random_state=42))]),
        "AdaBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                                       random_state=42))]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                        probability=True, random_state=42))]),
        "SVM_Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1, class_weight="balanced",
                        probability=True, random_state=42))]),
        "KNN_5": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1))]),
        "KNN_11": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=11, metric="euclidean", n_jobs=-1))]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                       random_state=42, n_jobs=-1))]),
    }

    # Optional: XGBoost
    try:
        from xgboost import XGBClassifier
        cands["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, use_label_encoder=False,
                                  eval_metric="logloss", random_state=42, n_jobs=-1))])
        print("  [+] XGBoost available")
    except ImportError:
        print("  [ ] XGBoost not installed (pip install xgboost to add)")

    # Optional: LightGBM
    try:
        from lightgbm import LGBMClassifier
        cands["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   class_weight="balanced", random_state=42, n_jobs=-1,
                                   verbose=-1))])
        print("  [+] LightGBM available")
    except ImportError:
        print("  [ ] LightGBM not installed (pip install lightgbm to add)")

    return cands


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Extra Models — Binary Fall Detection Comparison")
    print("=" * 65)

    print("\n  Building dataset...")
    X, y = build_dataset()
    print(f"  Windows: {len(X)}  FALL={sum(y==1)}  NO-FALL={sum(y==0)}")
    print(f"  Features per window: {X.shape[1]}")

    candidates = get_candidates()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "roc_auc", "precision", "recall"]

    rows = []
    print(f"\n  {'Model':<20} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for name, model in candidates.items():
        res  = cross_validate(model, X, y, cv=cv, scoring=scoring)
        acc  = res["test_accuracy"].mean()
        f1   = res["test_f1"].mean()
        auc  = res["test_roc_auc"].mean()
        prec = res["test_precision"].mean()
        rec  = res["test_recall"].mean()
        print(f"  {name:<20} {acc:>7.4f} {f1:>7.4f} {auc:>7.4f} {prec:>7.4f} {rec:>7.4f}")
        rows.append({"Model": name, "Accuracy": round(acc,4), "F1": round(f1,4),
                     "ROC_AUC": round(auc,4), "Precision": round(prec,4),
                     "Recall": round(rec,4)})

        # Train on full data and save
        model.fit(X, y)
        payload = {"model": model, "model_name": name,
                   "window_size": WINDOW_SIZE, "n_features_raw": 20,
                   "n_features_ml": X.shape[1], "classes": ["NO-FALL", "FALL"],
                   "cv_accuracy": acc, "cv_roc_auc": auc}
        out_path = OUT_DIR / f"{name}.pkl"
        with open(str(out_path), "wb") as f:
            pickle.dump(payload, f)

    # ── Save comparison CSV ────────────────────────────────────────────────────
    df = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_DIR / "comparison_report.csv", index=False)
    print(f"\n  Saved {len(rows)} models to extra_models/")
    print(f"  Comparison report: extra_models/comparison_report.csv")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fall Detection — Model Comparison (5-fold CV)", fontsize=13, fontweight="bold")

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))

    # AUC bar chart
    ax = axes[0]
    bars = ax.barh(df["Model"], df["ROC_AUC"], color=colors, edgecolor="white")
    ax.set_xlim(0.95, 1.005)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("ROC-AUC (5-fold CV)")
    ax.axvline(x=1.0, color="red", linestyle="--", lw=1, alpha=0.5)
    for bar, val in zip(bars, df["ROC_AUC"]):
        ax.text(bar.get_width()-0.001, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", ha="right", fontsize=8, color="white", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Multi-metric grouped bar
    ax2 = axes[1]
    metrics = ["Accuracy", "F1", "Precision", "Recall"]
    x = np.arange(len(df))
    w = 0.2
    for i, m in enumerate(metrics):
        ax2.bar(x + i*w, df[m], width=w, label=m, alpha=0.85)
    ax2.set_xticks(x + w*1.5)
    ax2.set_xticklabels(df["Model"], rotation=40, ha="right", fontsize=7)
    ax2.set_ylim(0.85, 1.02)
    ax2.set_ylabel("Score")
    ax2.set_title("All Metrics (5-fold CV)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved: extra_models/model_comparison.png")

    print(f"\n  WINNER: {df.iloc[0]['Model']}  (AUC={df.iloc[0]['ROC_AUC']})")
    print("=" * 65)


if __name__ == "__main__":
    main()
