"""
build_velocity_detector.py
==========================
Fall detection based on RATE OF Z CHANGE (velocity), not total drop.

Key insight from data:
  FALL:          z drops 0.5-1.5m in 3-8 frames (~0.1-0.3 sec) -> FAST
  Sit-to-stand:  z changes 0.3-0.5m over 25-50 frames (~2-3 sec) -> SLOW

=> Peak negative z-velocity separates falls from transitions.

Run from: Dataset1/
    python combinedmodels/build_velocity_detector.py
"""
import json, pickle
import numpy as np
from pathlib import Path
from collections import deque

DATA_ROOT = Path(__file__).resolve().parent.parent / "Dataset1"
OUT_DIR   = Path(__file__).resolve().parent
SNR       = 10.0
FRAME_DT  = 0.055   # seconds between frames

FALL_PREFIXES   = ("fall",)
NOFALL_PREFIXES = ("sit", "stand", "standing")


def get_z_trace(path):
    """Returns per-frame z_mean, height_range, n_points from a JSON file."""
    raw  = json.load(open(path, 'r', encoding='utf-8'))
    rows = raw['data'] if isinstance(raw, dict) and 'data' in raw else raw
    z_tr, hr_tr, np_tr = [], [], []
    for r in rows:
        fd  = r.get('frameData', r)
        pc  = fd.get('pointCloud', [])
        if not pc:
            z_tr.append(None); hr_tr.append(0.0); np_tr.append(0)
            continue
        pts = np.array(pc, np.float32)
        if pts.shape[1] > 4:
            m = pts[:, 4] >= SNR
            if m.sum() > 0: pts = pts[m]
        if len(pts) == 0:
            z_tr.append(None); hr_tr.append(0.0); np_tr.append(0)
            continue
        z_tr.append(float(pts[:, 2].mean()))
        hr_tr.append(float(pts[:, 2].max() - pts[:, 2].min()) if len(pts) > 1 else 0.0)
        np_tr.append(len(pts))
    return z_tr, hr_tr, np_tr


def peak_velocities(z_trace):
    """Compute frame-to-frame z velocity and return min (most negative = fastest fall)."""
    z_valid = [z for z in z_trace if z is not None]
    if len(z_valid) < 2:
        return 0.0, 0.0
    arr = np.array(z_valid, dtype=np.float64)
    vel = np.diff(arr) / FRAME_DT   # m/s per frame
    return float(vel.min()), float(vel.max())


def main():
    print("="*58)
    print("  Velocity-based fall detector — calibration")
    print("="*58)

    fall_vels, nofall_vels = [], []
    print("\n  Per-file peak z-velocity (m/s, negative = falling):")
    print(f"  {'File':<35} {'Label':<8} {'PeakVel':>9}")
    print(f"  {'-'*35} {'-'*8} {'-'*9}")

    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir(): continue
        nm = folder.name.lower()
        if nm.startswith(FALL_PREFIXES):       label, is_fall = "FALL",    True
        elif nm.startswith(NOFALL_PREFIXES):   label, is_fall = "NO-FALL", False
        else: continue

        for jf in sorted(folder.glob("*.json")):
            z_tr, hr_tr, np_tr = get_z_trace(jf)
            peak_neg, peak_pos = peak_velocities(z_tr)
            name = f"{folder.name}/{jf.name}"
            print(f"  {name:<35} {label:<8} {peak_neg:>9.3f}")
            if is_fall: fall_vels.append(peak_neg)
            else:       nofall_vels.append(peak_neg)

    print(f"\n  FALL   peak_vel: mean={np.mean(fall_vels):.3f}  "
          f"min={np.min(fall_vels):.3f}  max={np.max(fall_vels):.3f}")
    print(f"  NOFALL peak_vel: mean={np.mean(nofall_vels):.3f}  "
          f"min={np.min(nofall_vels):.3f}  max={np.max(nofall_vels):.3f}")

    # Best threshold = midpoint + bias toward recall
    thresh = (np.mean(fall_vels) + np.mean(nofall_vels)) / 2.0
    print(f"\n  Midpoint threshold: {thresh:.3f} m/s")

    # Sweep thresholds to find best F1
    best_f1, best_t = 0, thresh
    for t in np.linspace(min(fall_vels+nofall_vels), max(fall_vels+nofall_vels), 200):
        tp = sum(1 for v in fall_vels   if v <= t)
        fp = sum(1 for v in nofall_vels if v <= t)
        fn = len(fall_vels) - tp
        p  = tp/(tp+fp) if (tp+fp)>0 else 0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # Evaluate best threshold
    tp = sum(1 for v in fall_vels   if v <= best_t)
    fp = sum(1 for v in nofall_vels if v <= best_t)
    tn = sum(1 for v in nofall_vels if v >  best_t)
    fn = sum(1 for v in fall_vels   if v >  best_t)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    print(f"\n  Best threshold: {best_t:.3f} m/s  (F1={best_f1:.2f})")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Recall={rec:.0%}  Precision={prec:.0%}")

    # Save detector
    payload = {
        "version":           "velocity_v1",
        "z_vel_threshold":   round(best_t, 3),
        "frame_dt":          FRAME_DT,
        "classes":           ["NO-FALL", "FALL"],
        "calibration": {
            "fall_vel_mean":   round(float(np.mean(fall_vels)), 3),
            "nofall_vel_mean": round(float(np.mean(nofall_vels)), 3),
        }
    }
    out = OUT_DIR / "combined_detector.pkl"
    with open(str(out), "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Saved: {out}")
    print("="*58)


class VelocityFallDetector:
    """
    Streaming fall detector based on z-velocity.
    Feed one frame's z_mean at a time.
    Fires when z drops faster than threshold (m/s).

    Works with ANY coordinate system — uses velocity, not absolute z.
    """
    def __init__(self, z_vel_threshold=-1.5, frame_dt=0.055,
                 cooldown_frames=30, confirm_frames=2):
        self.z_vel_threshold = z_vel_threshold
        self.frame_dt        = frame_dt
        self.cooldown_frames = cooldown_frames
        self.confirm_frames  = confirm_frames
        self._prev_z         = None
        self._fast_frames    = 0
        self._cooldown       = 0

    def reset(self):
        self._prev_z      = None
        self._fast_frames = 0
        self._cooldown    = 0

    def update(self, z_mean: float, n_points: int = 99):
        """Single frame update. Returns (is_fall, velocity, info)."""
        if self._cooldown > 0:
            self._cooldown -= 1

        if self._prev_z is None or n_points == 0:
            self._prev_z = z_mean
            return False, 0.0, {"status": "init"}

        velocity = (z_mean - self._prev_z) / self.frame_dt
        self._prev_z = z_mean

        is_falling_fast = velocity <= self.z_vel_threshold
        if is_falling_fast:
            self._fast_frames += 1
        else:
            self._fast_frames = max(0, self._fast_frames - 1)

        is_fall = (self._fast_frames >= self.confirm_frames
                   and self._cooldown == 0)
        if is_fall:
            self._cooldown    = self.cooldown_frames
            self._fast_frames = 0

        return is_fall, round(velocity, 3), {
            "z_vel":       round(velocity, 3),
            "threshold":   self.z_vel_threshold,
            "fast_frames": self._fast_frames,
        }


if __name__ == "__main__":
    main()
