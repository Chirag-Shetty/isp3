"""
streaming_fall_detector.py
==========================
Streaming rule-based fall detector — no ML model needed.
Tracks centroid-z over time and fires when it drops suddenly.

Works regardless of radar coordinate system (uses RELATIVE drops).
Implements CLAUDE.md TIER 1 exactly.
"""
import pickle
import numpy as np
from collections import deque
from pathlib import Path

# Calibrated thresholds (from CLAUDE.md + ESPHome IWR6843 production)
Z_DROP_THRESHOLD  = 0.50   # centroid z must drop >= 0.5m to flag fall
BODY_FLAT_THRESH  = 0.50   # height_range < 0.5m = person horizontal
LOW_POINTS_THRESH = 8      # fewer than 8 points = person on floor
DETECTION_WINDOW  = 20     # frames to look back for drop (~1-2 sec)
MIN_VOTES         = 2      # need 2 of 3 signals


class StreamingFallDetector:
    """
    Feed one frame at a time. Internally tracks z_mean history.
    Falls are detected as sudden drops in centroid height.

    Usage:
        det = StreamingFallDetector()
        for each_frame_20_features:
            is_fall, conf, info = det.update(feat_20)
    """

    def __init__(self,
                 z_drop_threshold=Z_DROP_THRESHOLD,
                 body_flat_thresh=BODY_FLAT_THRESH,
                 low_points_thresh=LOW_POINTS_THRESH,
                 detection_window=DETECTION_WINDOW,
                 min_votes=MIN_VOTES):
        self.z_drop_threshold  = z_drop_threshold
        self.body_flat_thresh  = body_flat_thresh
        self.low_points_thresh = low_points_thresh
        self.detection_window  = detection_window
        self.min_votes         = min_votes
        self._z_buf  = deque(maxlen=detection_window + 5)
        self._fall_frames = 0          # consecutive fall frames
        self._cooldown    = 0          # frames until next alert allowed

    def reset(self):
        self._z_buf.clear()
        self._fall_frames = 0
        self._cooldown    = 0

    def update(self, feat_20: np.ndarray):
        """
        feat_20 : 1-D array of 20 pipeline features (from feature_extract.py)
                  feat[2]  = z centroid (raw, any coordinate system)
                  feat[9]  = n_points
                  feat[11] = height_range
        Returns:
            is_fall   : bool
            confidence: float 0-1
            info      : dict with signal values
        """
        z    = float(feat_20[2])
        npts = float(feat_20[9])
        hrng = float(feat_20[11])

        self._z_buf.append(z)

        if self._cooldown > 0:
            self._cooldown -= 1

        if len(self._z_buf) < self.detection_window:
            return False, 0.0, {"status": "warming_up",
                                 "buffered": len(self._z_buf)}

        # Peak z over detection window (highest recent height)
        window_z  = list(self._z_buf)[-self.detection_window:]
        z_peak    = max(window_z)
        z_current = window_z[-1]
        z_drop    = z_peak - z_current   # positive = centroid dropped

        # Three independent signals
        height_dropped = z_drop   >= self.z_drop_threshold
        body_flat      = hrng     <= self.body_flat_thresh
        few_points     = npts     <= self.low_points_thresh

        votes   = int(height_dropped) + int(body_flat) + int(few_points)
        is_fall = (votes >= self.min_votes) and (self._cooldown == 0)

        if is_fall:
            self._fall_frames += 1
            self._cooldown = 30   # suppress re-alert for ~1.5 sec
        else:
            self._fall_frames = 0

        confidence = min(1.0, votes / 3.0)
        info = {
            "z_drop":        round(z_drop, 3),
            "z_peak":        round(z_peak, 3),
            "z_current":     round(z_current, 3),
            "hrng":          round(hrng, 3),
            "npts":          int(npts),
            "height_dropped": height_dropped,
            "body_flat":      body_flat,
            "few_points":     few_points,
            "votes":          votes,
        }
        return is_fall, confidence, info

    # ── Batch prediction (for window-based API) ────────────────────────────
    def predict_window(self, window: np.ndarray):
        """
        window: (N, 20) array — N frames of pipeline features
        Runs frame-by-frame through the window.
        Returns final (is_fall, confidence, info) after last frame.
        """
        self.reset()
        result = (False, 0.0, {})
        for i in range(len(window)):
            result = self.update(window[i])
        return result


def calibrate_from_data(data_root: str) -> dict:
    """
    Walk Dataset1 subfolders and compute optimal thresholds
    using the median drop for fall vs no-fall recordings.
    Returns calibrated threshold dict.
    """
    import json
    root = Path(data_root)
    SNR  = 10.0; DT = 0.055; WS = 20

    def load_z_trace(path):
        raw = json.load(open(path, 'r', encoding='utf-8'))
        rows = raw['data'] if isinstance(raw, dict) and 'data' in raw else raw
        z_trace, hrng_trace, npts_trace = [], [], []
        for r in rows:
            fd  = r.get('frameData', r)
            pc  = fd.get('pointCloud', [])
            hd  = fd.get('heightData', [])
            if not pc:
                z_trace.append(0.0); hrng_trace.append(0.0); npts_trace.append(0)
                continue
            pts = np.array(pc, np.float32)
            if pts.shape[1] > 4:
                m = pts[:, 4] >= SNR
                if m.sum() > 0: pts = pts[m]
            if len(pts) == 0:
                z_trace.append(0.0); hrng_trace.append(0.0); npts_trace.append(0)
                continue
            z_trace.append(float(pts[:, 2].mean()))
            hrng_trace.append(float(pts[:, 2].max() - pts[:, 2].min()))
            npts_trace.append(len(pts))
        return z_trace, hrng_trace, npts_trace

    fall_drops, nofall_drops = [], []
    fall_hrng,  nofall_hrng  = [], []
    fall_npts,  nofall_npts  = [], []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir(): continue
        nm = folder.name.lower()
        if nm.startswith("fall"):          is_fall = True
        elif nm.startswith(("sit","stand")): is_fall = False
        else: continue

        for jf in sorted(folder.glob("*.json")):
            z_tr, hr_tr, np_tr = load_z_trace(jf)
            if len(z_tr) < WS: continue
            # Max drop in any WS-frame window
            max_drop = max(max(z_tr[i:i+WS]) - min(z_tr[i:i+WS])
                           for i in range(len(z_tr) - WS))
            min_hrng = min(hr_tr) if hr_tr else 0
            min_npts = min(np_tr) if np_tr else 0
            if is_fall:
                fall_drops.append(max_drop)
                fall_hrng.append(min_hrng)
                fall_npts.append(min_npts)
            else:
                nofall_drops.append(max_drop)
                nofall_hrng.append(min_hrng)
                nofall_npts.append(min_npts)

    print(f"  FALL   z_drop: mean={np.mean(fall_drops):.3f}  "
          f"min={np.min(fall_drops):.3f}  max={np.max(fall_drops):.3f}")
    print(f"  NOFALL z_drop: mean={np.mean(nofall_drops):.3f}  "
          f"min={np.min(nofall_drops):.3f}  max={np.max(nofall_drops):.3f}")
    print(f"  FALL   min_hrng: mean={np.mean(fall_hrng):.3f}")
    print(f"  NOFALL min_hrng: mean={np.mean(nofall_hrng):.3f}")
    print(f"  FALL   min_npts: mean={np.mean(fall_npts):.1f}")
    print(f"  NOFALL min_npts: mean={np.mean(nofall_npts):.1f}")

    # Threshold = midpoint between fall/nofall means
    z_thresh   = (np.mean(fall_drops)  + np.mean(nofall_drops))  / 2
    hrng_thresh = (np.mean(fall_hrng)  + np.mean(nofall_hrng))   / 2
    npts_thresh = (np.mean(fall_npts)  + np.mean(nofall_npts))   / 2

    print(f"\n  Calibrated thresholds:")
    print(f"    Z_DROP_THRESHOLD  = {z_thresh:.3f}")
    print(f"    BODY_FLAT_THRESH  = {hrng_thresh:.3f}")
    print(f"    LOW_POINTS_THRESH = {npts_thresh:.1f}")

    return {"z_drop_threshold":  round(z_thresh, 3),
            "body_flat_thresh":  round(hrng_thresh, 3),
            "low_points_thresh": round(npts_thresh, 1)}


if __name__ == "__main__":
    import sys
    data_root = Path(__file__).resolve().parent.parent / "Dataset1"
    print("="*55)
    print("  Calibrating StreamingFallDetector from data...")
    print("="*55)

    thresholds = calibrate_from_data(str(data_root))

    det = StreamingFallDetector(**thresholds)

    # Quick eval on raw files
    import json
    SNR = 10.0
    tp=fp=tn=fn=0
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir(): continue
        nm = folder.name.lower()
        if nm.startswith("fall"):            true_lbl = 1
        elif nm.startswith(("sit","stand")): true_lbl = 0
        else: continue
        for jf in sorted(folder.glob("*.json")):
            raw = json.load(open(jf,'r',encoding='utf-8'))
            rows = raw['data'] if isinstance(raw,dict) and 'data' in raw else raw
            det.reset()
            file_fall = False
            for r in rows:
                fd = r.get('frameData', r)
                pc = fd.get('pointCloud', [])
                hd = fd.get('heightData', [])
                feat = np.zeros(20, np.float32)
                if pc:
                    pts = np.array(pc, np.float32)
                    if pts.shape[1] > 4:
                        m = pts[:,4] >= SNR
                        if m.sum() > 0: pts = pts[m]
                    if len(pts) > 0:
                        feat[2]  = pts[:,2].mean()
                        feat[9]  = float(len(pts))
                        feat[11] = float(pts[:,2].max()-pts[:,2].min()) if len(pts)>1 else 0
                is_fall, conf, _ = det.update(feat)
                if is_fall: file_fall = True
            pred = 1 if file_fall else 0
            if pred==1 and true_lbl==1: tp+=1
            elif pred==1 and true_lbl==0: fp+=1
            elif pred==0 and true_lbl==0: tn+=1
            else: fn+=1

    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    print(f"\n  Evaluation on {tp+fp+tn+fn} files:")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Recall={rec:.0%}  Precision={prec:.0%}")

    # Save
    out = Path(__file__).resolve().parent / "combined_detector.pkl"
    payload = {"detector": det, "thresholds": thresholds,
               "version": "streaming_v1", "classes": ["NO-FALL","FALL"]}
    with open(str(out), "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Saved: {out}")
    print("="*55)
