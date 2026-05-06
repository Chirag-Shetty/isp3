"""
build_combined_detector.py  (v2 — One-Class SVM + Rule-Based)
=============================================================
Strategy from CLAUDE.md:
  - TIER 1: Rule-based thresholds on z-trajectory  (no training needed)
  - TIER 2: One-Class SVM trained on NO-FALL only  (falls = anomalies)
  - Ensemble: either tier fires -> FALL

Key insight: use OFFSET-INVARIANT features (z_range, z_slope, hrng changes)
so the model works regardless of radar coordinate system.

Run from: Dataset1/
    python combinedmodels/build_combined_detector.py
Saves:    combinedmodels/combined_detector.pkl
"""

import json, pickle
import numpy as np
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path(__file__).resolve().parent.parent / "Dataset1"
OUT_DIR     = Path(__file__).resolve().parent
WINDOW_SIZE = 40
STRIDE      = 5
SNR_THRESH  = 10.0
FRAME_DT    = 0.055

FALL_PREFIXES   = ("fall",)
NOFALL_PREFIXES = ("sit", "stand", "standing")

# Rule-based thresholds (calibrated from data below)
Z_RANGE_THRESH    = 0.45   # window z_max - z_min > this -> fall candidate
HRNG_DROP_THRESH  = 0.30   # height_range drops > this in window -> fall
NPTS_DROP_THRESH  = 0.40   # n_pts drops to < 40% of window max -> fall
VZ_THRESH         = -0.35  # vz_min < this -> rapid downward motion

# ── Feature extractor (same as rpi_pipeline) ─────────────────────────────────
def extract_frame_features(pc, td, hd, prev=None):
    if not pc or len(pc) == 0:
        return np.zeros(20, np.float32), np.zeros(3, np.float32)
    pts = np.array(pc, dtype=np.float32)
    if pts.shape[1] > 4:
        m = pts[:, 4] >= SNR_THRESH
        if m.sum() > 0: pts = pts[m]
    if len(pts) == 0:
        return np.zeros(20, np.float32), np.zeros(3, np.float32)
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    d = pts[:,3] if pts.shape[1] > 3 else np.zeros(len(pts))
    xm,ym,zm = x.mean(), y.mean(), z.mean()
    r = float(np.sqrt(xm**2+ym**2+zm**2)) + 1e-8
    vx,vy,vz = float(d.mean())*(xm/r), float(d.mean())*(ym/r), float(d.mean())*(zm/r)
    cv  = np.array([vx,vy,vz], np.float32)
    acc = (cv-prev)/FRAME_DT if prev is not None else np.zeros(3, np.float32)
    sp  = float(np.sqrt(x.var()+y.var())) if len(pts)>1 else 0.0
    hr  = float(z.max()-z.min())          if len(pts)>1 else 0.0
    pf  = np.array([xm,ym,zm,vx,vy,vz,acc[0],acc[1],acc[2],
                    float(len(pts)),sp,hr], np.float32)
    tf  = np.zeros(6, np.float32)
    if td and len(td)>0 and len(td[0])>=7: tf=np.array(td[0][1:7],np.float32)
    hf  = np.zeros(2, np.float32)
    if hd and len(hd)>0 and len(hd[0])>=3: hf=np.array(hd[0][1:3],np.float32)
    return np.concatenate([pf,tf,hf]), cv


def load_frames(path):
    raw  = json.load(open(path, 'r', encoding='utf-8'))
    rows = raw['data'] if isinstance(raw,dict) and 'data' in raw else raw
    out  = []
    for r in rows:
        fd = r.get('frameData', r)
        out.append({'pointCloud': fd.get('pointCloud',[]),
                    'trackData':  fd.get('trackData',[]),
                    'heightData': fd.get('heightData',[])})
    return out


def file_to_feature_buf(path):
    frames = load_frames(path)
    buf, prev = [], None
    for fr in frames:
        feat, prev = extract_frame_features(
            fr['pointCloud'], fr['trackData'], fr['heightData'], prev)
        buf.append(feat)
    return np.array(buf, dtype=np.float32)  # (T, 20)


# ── Offset-invariant window features ─────────────────────────────────────────
def window_features(win):
    """
    win: (40, 20) — raw feature array from pipeline
    Returns 20-dim OFFSET-INVARIANT feature vector.
    All features are differences/slopes, not absolute values.
    """
    z    = win[:,  2].astype(np.float64)   # centroid z
    hrng = win[:, 11].astype(np.float64)   # height range
    npts = win[:,  9].astype(np.float64)   # n_points
    vz   = win[:,  5].astype(np.float64)   # vertical vel proxy
    spxy = win[:, 10].astype(np.float64)   # spread XY
    xs   = np.arange(len(z), dtype=np.float64)
    h    = len(z) // 2

    def slp(v):
        return float(np.polyfit(xs, v, 1)[0]) if v.std() > 1e-8 else 0.0

    # Normalise n_points to 0-1 range within window
    npts_norm = npts / (npts.max() + 1e-8)

    return np.array([
        # Z trajectory (offset-invariant — differences only)
        z.max() - z.min(),                        # z_range in window
        slp(z),                                   # z_slope (neg = falling)
        z.std(),                                  # z_variability
        z[:h].mean() - z[h:].mean(),              # z first-half minus second-half
        min(np.diff(z)) if len(z)>1 else 0,      # fastest single-frame z drop

        # Height range (collapses when person horizontal)
        hrng.max() - hrng.min(),                  # hrng_range
        slp(hrng),                                # hrng_slope
        hrng[:h].mean() - hrng[h:].mean(),        # hrng first-half minus second-half
        hrng.std(),                               # hrng_variability
        hrng.min(),                               # lowest height_range seen

        # Point count (drops dramatically after fall)
        1.0 - npts_norm[h:].mean(),               # fraction of pts lost in 2nd half
        npts_norm.std(),                          # npts variability
        slp(npts_norm),                           # npts trend

        # Velocity proxy
        vz.min(),                                 # most negative vz
        abs(vz).max(),                            # peak |vz|
        vz.std(),                                 # vz variability

        # Spread XY (increases when lying flat)
        spxy.max() - spxy.min(),                  # spread_xy_range
        spxy[h:].mean() - spxy[:h].mean(),        # spread increases in 2nd half

        # Cross-feature
        (z.max()-z.min()) * (hrng.max()-hrng.min()),   # joint drop signal
        (z[:h].mean()-z[h:].mean()) * hrng[:h].mean(), # z_drop weighted by hrng
    ], dtype=np.float32)


# ── Rule-Based Detector ───────────────────────────────────────────────────────
class RuleDetector:
    """
    TIER 1: Zero training. Fires on any significant fall signature.
    Uses majority vote of 5 independent indicators.
    """
    def predict_window(self, win):
        z    = win[:,  2]
        hrng = win[:, 11]
        npts = win[:,  9]
        vz   = win[:,  5]
        h    = len(z) // 2

        z_range       = float(z.max() - z.min())
        z_drop_1h2h   = float(z[:h].mean() - z[h:].mean())
        hrng_drop     = float(hrng[:h].mean() - hrng[h:].mean())
        npts_ratio    = float(npts[h:].mean() / (npts[:h].mean() + 1e-8))
        vz_min        = float(vz.min())

        signals = dict(z_range=round(z_range,3), z_drop=round(z_drop_1h2h,3),
                       hrng_drop=round(hrng_drop,3), npts_ratio=round(npts_ratio,3),
                       vz_min=round(vz_min,3))
        votes = (
            (z_range     > Z_RANGE_THRESH)    +
            (z_drop_1h2h > Z_RANGE_THRESH*0.6)+
            (hrng_drop   > HRNG_DROP_THRESH)  +
            (npts_ratio  < NPTS_DROP_THRESH)  +
            (vz_min      < VZ_THRESH)
        )
        return bool(votes >= 2), float(votes/5), signals


# ── Build dataset ─────────────────────────────────────────────────────────────
def build_dataset():
    nofall_X, fall_X = [], []
    fall_info = []   # (folder, file, z_range) for diagnostic

    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir(): continue
        nm = folder.name.lower()
        if nm.startswith(FALL_PREFIXES):      label = 1
        elif nm.startswith(NOFALL_PREFIXES):  label = 0
        else: continue

        for jf in sorted(folder.glob("*.json")):
            buf = file_to_feature_buf(jf)
            if len(buf) < WINDOW_SIZE: continue
            wins = [buf[s:s+WINDOW_SIZE]
                    for s in range(0, len(buf)-WINDOW_SIZE+1, STRIDE)]
            for w in wins:
                fv = window_features(w)
                if label == 0:
                    nofall_X.append(fv)
                else:
                    fall_X.append(fv)
                    fall_info.append((folder.name, jf.name, round(float(w[:,2].max()-w[:,2].min()),3)))

    return np.array(nofall_X, np.float32), np.array(fall_X, np.float32), fall_info


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Combined Fall Detector v2  (One-Class SVM + Rules)")
    print("="*60)

    print("\n  Building dataset ...")
    nofall_X, fall_X, fall_info = build_dataset()
    print(f"  NO-FALL windows : {len(nofall_X)}")
    print(f"  FALL windows    : {len(fall_X)}")
    print(f"  Feature dims    : {nofall_X.shape[1]}")

    # ── Show feature distributions ────────────────────────────────────────────
    print("\n  Feature distributions (z_range = key feature):")
    print(f"  NO-FALL  z_range  mean={nofall_X[:,0].mean():.3f}  "
          f"std={nofall_X[:,0].std():.3f}  max={nofall_X[:,0].max():.3f}")
    print(f"  FALL     z_range  mean={fall_X[:,0].mean():.3f}  "
          f"std={fall_X[:,0].std():.3f}  max={fall_X[:,0].max():.3f}")
    print(f"\n  NO-FALL  hrng_range mean={nofall_X[:,5].mean():.3f}  max={nofall_X[:,5].max():.3f}")
    print(f"  FALL     hrng_range mean={fall_X[:,5].mean():.3f}  max={fall_X[:,5].max():.3f}")
    print(f"\n  Top 5 FALL windows by z_range:")
    z_ranges = [(i, fall_X[i,0]) for i in range(len(fall_X))]
    for i, zr in sorted(z_ranges, key=lambda x:-x[1])[:5]:
        npts_r = fall_X[i, 11]
        hrng_r = fall_X[i, 5]
        print(f"    [{i:3d}] z_range={zr:.3f}  hrng_range={hrng_r:.3f}  npts_var={npts_r:.3f}"
              f"  ({fall_info[i][0]}/{fall_info[i][1]})")

    # ── Rule-based evaluation ─────────────────────────────────────────────────
    print("\n  [TIER 1] Rule-based performance:")
    rule = RuleDetector()
    all_X   = np.concatenate([nofall_X, fall_X])
    all_y   = np.array([0]*len(nofall_X) + [1]*len(fall_X))
    rb_pred = []
    for fv in all_X:
        # Build a fake minimal "window" struct from feature vector
        votes = (
            (fv[0]  > Z_RANGE_THRESH)     +
            (fv[3]  > Z_RANGE_THRESH*0.6) +
            (fv[7]  > HRNG_DROP_THRESH)   +
            (fv[11] > 0.3)                +
            (fv[13] < VZ_THRESH)
        )
        rb_pred.append(1 if votes >= 2 else 0)
    rb_pred = np.array(rb_pred)
    tp = int(((rb_pred==1)&(all_y==1)).sum())
    fp = int(((rb_pred==1)&(all_y==0)).sum())
    tn = int(((rb_pred==0)&(all_y==0)).sum())
    fn = int(((rb_pred==0)&(all_y==1)).sum())
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    print(f"  Recall={rec:.0%}  Precision={prec:.0%}  "
          f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # ── One-Class SVM (trained on NO-FALL only) ───────────────────────────────
    print("\n  [TIER 2] One-Class SVM (anomaly detection on NO-FALL):")
    # Tune nu = expected fraction of outliers in training set
    best_nu, best_recall, best_model = 0.05, 0, None
    for nu in [0.03, 0.05, 0.08, 0.10, 0.15]:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('ocsvm', OneClassSVM(kernel='rbf', nu=nu, gamma='scale'))
        ])
        pipe.fit(nofall_X)
        # Predict: +1 = normal (NO-FALL), -1 = anomaly (FALL)
        preds_nofall = pipe.predict(nofall_X)
        preds_fall   = pipe.predict(fall_X)
        recall_fall  = int((preds_fall==-1).sum()) / len(fall_X)
        spec_nofall  = int((preds_nofall==1).sum()) / len(nofall_X)
        print(f"    nu={nu:.2f}  FALL recall={recall_fall:.0%}  NO-FALL specificity={spec_nofall:.0%}")
        if recall_fall > best_recall:
            best_recall, best_nu, best_model = recall_fall, nu, pipe

    print(f"\n  Best nu={best_nu}  FALL recall={best_recall:.0%}")

    # ── Final evaluation: TIER1 OR TIER2 ─────────────────────────────────────
    print("\n  [ENSEMBLE] TIER1 OR TIER2 (fire on either):")
    oc_pred_nofall = (best_model.predict(nofall_X) == -1).astype(int)
    oc_pred_fall   = (best_model.predict(fall_X)   == -1).astype(int)
    rb_nofall = rb_pred[:len(nofall_X)]
    rb_fall   = rb_pred[len(nofall_X):]
    ens_nofall = np.clip(rb_nofall + oc_pred_nofall, 0, 1)
    ens_fall   = np.clip(rb_fall   + oc_pred_fall,   0, 1)
    ens_tp = int(ens_fall.sum())
    ens_fn = len(fall_X) - ens_tp
    ens_fp = int(ens_nofall.sum())
    ens_tn = len(nofall_X) - ens_fp
    ens_rec  = ens_tp/(ens_tp+ens_fn) if (ens_tp+ens_fn)>0 else 0
    ens_prec = ens_tp/(ens_tp+ens_fp) if (ens_tp+ens_fp)>0 else 0
    print(f"  Recall={ens_rec:.0%}  Precision={ens_prec:.0%}  "
          f"TP={ens_tp}  FP={ens_fp}  TN={ens_tn}  FN={ens_fn}")

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "ocsvm_pipeline":       best_model,
        "rule_config": {
            "z_range_thresh":   Z_RANGE_THRESH,
            "hrng_drop_thresh": HRNG_DROP_THRESH,
            "npts_drop_thresh": NPTS_DROP_THRESH,
            "vz_thresh":        VZ_THRESH,
            "min_votes":        2,
        },
        "n_features":           nofall_X.shape[1],
        "window_size":          WINDOW_SIZE,
        "ocsvm_fall_recall":    best_recall,
        "ensemble_recall":      ens_rec,
        "classes":              ["NO-FALL", "FALL"],
        "version":              "combined_v2",
    }
    out = OUT_DIR / "combined_detector.pkl"
    with open(str(out), "wb") as f:
        pickle.dump(payload, f)

    print(f"\n  Saved: {out}")
    print(f"  One-Class SVM FALL recall : {best_recall:.0%}")
    print(f"  Ensemble FALL recall      : {ens_rec:.0%}")
    print("="*60)


if __name__ == "__main__":
    main()
