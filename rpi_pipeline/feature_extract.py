"""
feature_extract.py
------------------
Extracts the exact same 20-dimensional feature vector used during
binary fall detection training (run_fall_detection.py).

Features [0-11]: Point cloud statistics
  0: x_mean         - centroid x
  1: y_mean         - centroid y
  2: z_mean         - centroid z
  3: vx_angular     - mean(doppler * sin(azimuth))
  4: vy_angular     - mean(doppler * cos(azimuth))
  5: vz_angular     - mean(doppler * sin(elevation))
  6: ax             - frame-to-frame delta of feat[3] / FRAME_DT
  7: ay             - frame-to-frame delta of feat[4] / FRAME_DT
  8: az             - frame-to-frame delta of feat[5] / FRAME_DT
  9: n_points       - number of points passing SNR filter
  10: spread_xy     - sqrt(var(x) + var(y))
  11: height_range  - max(z) - min(z)

Features [12-17]: Tracker output (track x,y,z,vx,vy,vz)
Features [18-19]: HeightData (maxZ, minZ)

IMPORTANT: Send RAW (unscaled) features to the cloud API.
           The API applies the StandardScaler internally.
"""

import numpy as np
from config import SNR_THRESHOLD, FRAME_DT


def extract_frame_features(point_cloud, track_data, height_data, prev_feat=None):
    """
    Extract 20-dim feature vector from a single radar frame.

    Args:
        point_cloud : list of [x, y, z, doppler, snr, ...]
        track_data  : list of tracker rows [trackId, x, y, z, vx, vy, vz, ...]
        height_data : list of height rows  [trackId, maxZ, minZ, ...]
        prev_feat   : previous frame's feature vector (np.ndarray shape 20),
                      used for acceleration (feats 6-8). Pass None for first frame.

    Returns:
        feat          : np.ndarray shape (20,) — raw unscaled features
        feat          : same array (returned twice for API compat with old callers
                        that did:  feat, prev_vel = extract_frame_features(...))
    """
    pts = np.array(point_cloud, dtype=np.float32) if point_cloud else np.empty((0, 5))

    # Apply SNR filter if SNR column exists
    if pts.ndim == 2 and pts.shape[1] > 4:
        mask = pts[:, 4] >= SNR_THRESHOLD
        pts = pts[mask]

    feat = np.zeros(20, dtype=np.float32)

    if len(pts) > 0:
        # ── Centroid ──────────────────────────────────────────────────────────
        feat[0] = np.mean(pts[:, 0])   # x_mean
        feat[1] = np.mean(pts[:, 1])   # y_mean
        feat[2] = np.mean(pts[:, 2])   # z_mean

        # ── Angular velocity decomposition (MATCHES TRAINING EXACTLY) ─────────
        angles = np.arctan2(pts[:, 0], pts[:, 1])            # azimuth
        elev   = np.arctan2(pts[:, 2],
                            np.sqrt(pts[:, 0]**2 + pts[:, 1]**2))  # elevation
        doppler = pts[:, 3]

        feat[3] = np.mean(doppler * np.sin(angles))   # vx_angular
        feat[4] = np.mean(doppler * np.cos(angles))   # vy_angular
        feat[5] = np.mean(doppler * np.sin(elev))     # vz_angular

        # ── Acceleration (delta from previous frame) ──────────────────────────
        if prev_feat is not None:
            feat[6] = (feat[3] - prev_feat[3]) / FRAME_DT
            feat[7] = (feat[4] - prev_feat[4]) / FRAME_DT
            feat[8] = (feat[5] - prev_feat[5]) / FRAME_DT
        # else feats 6-8 remain 0 (first frame)

        # ── Point cloud shape ─────────────────────────────────────────────────
        feat[9]  = float(len(pts))
        feat[10] = float(np.sqrt(np.var(pts[:, 0]) + np.var(pts[:, 1])))  # spread_xy
        feat[11] = float(np.max(pts[:, 2]) - np.min(pts[:, 2]))           # height_range

    # ── Tracker features [12-17] ──────────────────────────────────────────────
    if track_data and len(track_data) > 0:
        td = track_data[0]
        # td layout: [trackId, x, y, z, vx, vy, vz, ...]
        for i, idx in enumerate(range(12, 18)):
            feat[idx] = float(td[i + 1]) if len(td) > i + 1 else 0.0

    # ── Height features [18-19] ───────────────────────────────────────────────
    if height_data and len(height_data) > 0:
        hd = height_data[0]
        # hd layout: [trackId, maxZ, minZ, ...]
        feat[18] = float(hd[1]) if len(hd) > 1 else 0.0
        feat[19] = float(hd[2]) if len(hd) > 2 else 0.0

    # Return feat twice: new callers use feat only; old callers unpack (feat, prev)
    return feat, feat


def extract_recording_features(frames):
    """
    Process all frames in a recording -> (T, 20) feature array.
    """
    features = []
    prev = None
    for frame_obj in frames:
        fd = frame_obj.get("frameData", frame_obj)
        pc = fd.get("pointCloud", [])
        td = fd.get("trackData", [])
        hd = fd.get("heightData", [])

        feat, prev = extract_frame_features(pc, td, hd, prev)
        features.append(feat)
    return np.array(features, dtype=np.float32)


def build_sliding_windows(feature_buffer, window_size, stride):
    """
    Build all complete sliding windows from a growing feature buffer.

    Args:
        feature_buffer : list of np.array(20,) frames
        window_size    : int, frames per window
        stride         : int, frames to slide

    Returns:
        list of np.array(window_size, 20)
    """
    T = len(feature_buffer)
    windows = []
    if T < window_size:
        return windows
    for start in range(0, T - window_size + 1, stride):
        windows.append(np.array(feature_buffer[start:start + window_size],
                                dtype=np.float32))
    return windows
