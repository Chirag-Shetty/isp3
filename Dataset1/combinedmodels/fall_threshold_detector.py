"""
fall_threshold_detector.py
==========================
Simple threshold-based fall detector.
No training. No model file. Pure physics.

When STANDING  -> height_range ~1.0-1.8m, n_pts ~20-80
When SITTING   -> height_range ~0.4-0.8m, n_pts ~10-40
When FALLEN    -> height_range ~0.0-0.4m, n_pts ~2-10

Save: combinedmodels/combined_detector.pkl
"""

import json, pickle, numpy as np
from pathlib import Path
from collections import deque

OUT_DIR  = Path(__file__).resolve().parent
DATA_ROOT = OUT_DIR.parent / "Dataset1"
SNR = 10.0

# ── THRESHOLDS (based on mmWave physics) ──────────────────────────────────────
#
#  height_range = z_max - z_min of all detected points in ONE frame
#  This is offset-invariant (same regardless of sensor height/tilt)
#
#  A person STANDING presents ~1.5m of vertical extent to the radar
#  A person FALLEN is lying flat -> height_range collapses to <0.4m
#
PERSON_TALL_THRESH   = 0.7   # height_range > this = person is upright recently
PERSON_FALLEN_THRESH = 0.35  # height_range < this = person is flat/fallen
N_POINTS_LOW         = 8     # n_points < this = very sparse (person on floor)
Z_DROP_THRESH        = 0.45  # z_mean must drop this much from recent peak
HISTORY_FRAMES       = 25    # frames to look back for "was standing" check
SMOOTH_N             = 5     # smooth z_mean over this many frames
MIN_POINTS_VALID     = 3     # ignore frames with fewer points (noise)
PERSIST_FRAMES       = 4     # flat height_range must persist for this many frames


class FallThresholdDetector:
    """
    Rule-based fall detector using height_range collapse + z-drop.

    Feed frames one at a time via update().
    No training required.
    """

    def __init__(self,
                 person_tall_thresh=PERSON_TALL_THRESH,
                 person_fallen_thresh=PERSON_FALLEN_THRESH,
                 n_points_low=N_POINTS_LOW,
                 z_drop_thresh=Z_DROP_THRESH,
                 history_frames=HISTORY_FRAMES,
                 smooth_n=SMOOTH_N):

        self.tall_thresh   = person_tall_thresh
        self.fallen_thresh = person_fallen_thresh
        self.npts_low      = n_points_low
        self.z_drop_thresh = z_drop_thresh
        self.hist          = history_frames
        self.smooth_n      = smooth_n
        self.min_pts_valid = MIN_POINTS_VALID
        self.persist       = PERSIST_FRAMES

        # Rolling history
        self._hrng_buf       = deque(maxlen=history_frames)
        self._z_buf          = deque(maxlen=history_frames + smooth_n)
        self._npts_buf       = deque(maxlen=history_frames)
        self._flat_streak    = 0   # consecutive frames with low height_range
        self._cooldown       = 0

    def reset(self):
        self._hrng_buf.clear()
        self._z_buf.clear()
        self._npts_buf.clear()
        self._cooldown = 0

    def update(self, z_mean: float, height_range: float, n_points: int):
        """
        z_mean       : centroid z this frame (any units/offset)
        height_range : z_max - z_min of point cloud this frame
        n_points     : number of detected radar points this frame

        Returns: (is_fall: bool, confidence: float, info: dict)
        """
        if self._cooldown > 0:
            self._cooldown -= 1

        # Skip frames with too few points (radar noise, person out of FOV)
        if n_points >= self.min_pts_valid:
            self._hrng_buf.append(height_range)
            self._z_buf.append(z_mean)
            self._npts_buf.append(n_points)

        warmup = len(self._hrng_buf) < max(self.smooth_n, 5)
        if warmup:
            return False, 0.0, {"status": "warming_up"}

        # ── Smooth z to remove frame noise ────────────────────────────────
        recent_z   = list(self._z_buf)
        z_smooth   = np.mean(recent_z[-self.smooth_n:])
        z_peak     = max(recent_z[-self.hist:]) if len(recent_z) >= 5 else z_smooth
        z_drop     = z_peak - z_smooth   # positive = dropped

        # ── Current frame values ──────────────────────────────────────────
        hrng_now   = height_range
        hrng_hist  = list(self._hrng_buf)
        was_tall   = (len(hrng_hist) >= 5 and
                      max(hrng_hist[-min(self.hist, len(hrng_hist)):]) > self.tall_thresh)
        is_flat    = hrng_now < self.fallen_thresh and n_points >= self.min_pts_valid
        z_dropped  = z_drop > self.z_drop_thresh
        few_pts    = (self.min_pts_valid <= n_points < self.npts_low)

        # ── Persistence: flat must last PERSIST_FRAMES consecutive frames ─
        if is_flat:
            self._flat_streak += 1
        else:
            self._flat_streak = 0
        sustained_flat = self._flat_streak >= self.persist

        # ── Fall signals ──────────────────────────────────────────────────
        #  PRIMARY: height_range SUSTAINED collapse AND was previously tall
        #  SECONDARY: z dropped significantly AND few points
        primary   = was_tall and sustained_flat
        secondary = was_tall and z_dropped and few_pts

        votes = int(primary) + int(secondary)

        is_fall = (votes >= 1) and (self._cooldown == 0)

        if is_fall:
            self._cooldown = 40   # suppress for ~2 sec

        confidence = min(1.0, votes / 3.0)

        info = {
            "height_range":  round(hrng_now, 3),
            "was_tall":      was_tall,
            "is_flat":       is_flat,
            "z_drop":        round(z_drop, 3),
            "z_dropped":     z_dropped,
            "n_points":      n_points,
            "few_pts":       few_pts,
            "votes":         votes,
        }
        return is_fall, confidence, info

    def update_from_features(self, feat_20: np.ndarray):
        """
        Convenience: pass the 20-dim pipeline feature vector directly.
        feat_20[2]  = z_mean
        feat_20[9]  = n_points
        feat_20[11] = height_range
        """
        return self.update(
            z_mean       = float(feat_20[2]),
            height_range = float(feat_20[11]),
            n_points     = int(feat_20[9]),
        )

    def update_from_window(self, window: np.ndarray):
        """
        window: (N, 20) — run frame-by-frame through a buffered window.
        Returns result after last frame.
        """
        self.reset()
        result = (False, 0.0, {})
        for i in range(len(window)):
            result = self.update_from_features(window[i])
        return result


# ── Calibration / evaluation on training data ─────────────────────────────────

def get_frame_values(path):
    """Returns (z_mean, height_range, n_points) per frame from a JSON file."""
    raw  = json.load(open(path, 'r', encoding='utf-8'))
    rows = raw['data'] if isinstance(raw, dict) and 'data' in raw else raw
    out  = []
    for r in rows:
        fd  = r.get('frameData', r)
        pc  = fd.get('pointCloud', [])
        if not pc:
            out.append((0.0, 0.0, 0)); continue
        pts = np.array(pc, np.float32)
        if pts.shape[1] > 4:
            m = pts[:, 4] >= SNR
            if m.sum() > 0: pts = pts[m]
        if len(pts) == 0:
            out.append((0.0, 0.0, 0)); continue
        zm   = float(pts[:, 2].mean())
        hrng = float(pts[:, 2].max() - pts[:, 2].min()) if len(pts) > 1 else 0.0
        out.append((zm, hrng, len(pts)))
    return out


def evaluate(det: FallThresholdDetector, data_root: Path):
    fall_px = ("fall",)
    nofall_px = ("sit", "stand", "standing")
    tp = fp = tn = fn = 0
    rows = []
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir(): continue
        nm = folder.name.lower()
        if nm.startswith(fall_px):     true_lbl = 1
        elif nm.startswith(nofall_px): true_lbl = 0
        else: continue
        for jf in sorted(folder.glob("*.json")):
            frames = get_frame_values(jf)
            det.reset()
            file_fall = False
            for zm, hrng, npts in frames:
                is_fall, conf, info = det.update(zm, hrng, npts)
                if is_fall:
                    file_fall = True
            pred = 1 if file_fall else 0
            label = "FALL" if true_lbl else "NO-FALL"
            pred_s = "FALL" if pred else "NO-FALL"
            correct = "OK" if pred == true_lbl else "WRONG"
            rows.append(f"  {folder.name}/{jf.name:<30} true={label:<8} pred={pred_s:<8} {correct}")
            if pred==1 and true_lbl==1: tp+=1
            elif pred==1 and true_lbl==0: fp+=1
            elif pred==0 and true_lbl==0: tn+=1
            else: fn+=1
    return tp, fp, tn, fn, rows


if __name__ == "__main__":
    print("="*60)
    print("  Fall Threshold Detector — Physics-based")
    print("="*60)
    print(f"\n  Thresholds:")
    print(f"    height_range > {PERSON_TALL_THRESH}m  -> person was UPRIGHT recently")
    print(f"    height_range < {PERSON_FALLEN_THRESH}m  -> person is FLAT (fallen)")
    print(f"    z_drop       > {Z_DROP_THRESH}m  -> centroid dropped")
    print(f"    n_points     < {N_POINTS_LOW}     -> very sparse (on floor)")

    # Show what height_range looks like per activity
    print("\n  Per-folder height_range stats:")
    print(f"  {'Folder':<25} {'mean':>7} {'min':>7} {'max':>7}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7}")
    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir(): continue
        nm = folder.name.lower()
        if not (nm.startswith(("fall","sit","stand"))): continue
        all_hrng = []
        for jf in folder.glob("*.json"):
            for zm, hrng, npts in get_frame_values(jf):
                if hrng > 0: all_hrng.append(hrng)
        if all_hrng:
            print(f"  {folder.name:<25} {np.mean(all_hrng):>7.3f} "
                  f"{np.min(all_hrng):>7.3f} {np.max(all_hrng):>7.3f}")

    det = FallThresholdDetector()
    print(f"\n  Evaluation on all {47} files:")
    tp, fp, tn, fn, detail = evaluate(det, DATA_ROOT)
    for row in detail:
        print(row)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f"\n  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Recall={rec:.0%}  Precision={prec:.0%}  F1={f1:.2f}")

    # Save
    out = OUT_DIR / "combined_detector.pkl"
    payload = {
        "detector":   det,
        "thresholds": {
            "person_tall_thresh":   PERSON_TALL_THRESH,
            "person_fallen_thresh": PERSON_FALLEN_THRESH,
            "n_points_low":         N_POINTS_LOW,
            "z_drop_thresh":        Z_DROP_THRESH,
            "history_frames":       HISTORY_FRAMES,
        },
        "version":  "threshold_v1",
        "classes":  ["NO-FALL", "FALL"],
        "recall":   rec,
        "precision": prec,
    }
    with open(str(out), "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Saved: {out}")
    print("="*60)
