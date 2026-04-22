"""
feature_extract.py
------------------
Extracts the same 12-dimensional feature vector used during training.
Mirrors the logic from the Google Colab notebook exactly.
"""

import numpy as np
from config import SNR_THRESHOLD, NUM_FEATURES, FRAME_DT


def extract_frame_features(point_cloud, track_data, height_data, prev_velocity=None):
    """
    Extract 20-dim feature vector from a single radar frame.

    Features:
      0-11: PointCloud (x, y, z, vx, vy, vz, ax, ay, az, n_points, spread_xy, height_range)
      12-17: TrackData (track_x, track_y, track_z, track_vx, track_vy, track_vz)
      18-19: HeightData (person_height, bottom_height)
    """
    # 1) Point Cloud Features (12)
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

            acc = (current_velocity - prev_velocity) / FRAME_DT if prev_velocity is not None else np.zeros(3, dtype=np.float32)

            spread_xy = float(np.sqrt(x.var() + y.var())) if n_points > 1 else 0.0
            height_range = float(z.max() - z.min()) if n_points > 1 else 0.0

            pc_feat = np.array([
                x_mean, y_mean, z_mean,
                vx_mean, vy_mean, vz_mean,
                acc[0], acc[1], acc[2],
                float(n_points), spread_xy, height_range
            ], dtype=np.float32)

    # 2) Track Features (6)
    track_feat = np.zeros(6, dtype=np.float32)
    if track_data and len(track_data) > 0:
        # Standard tracker shape: [trackId, x, y, z, vx, vy, vz, ax, ay, az, ...]
        td = track_data[0]
        if len(td) >= 7:
            track_feat = np.array(td[1:7], dtype=np.float32)

    # 3) Height Features (2)
    height_feat = np.zeros(2, dtype=np.float32)
    if height_data and len(height_data) > 0:
        # Standard height shape: [trackId, person_height, bottom_height]
        hd = height_data[0]
        if len(hd) >= 3:
            height_feat = np.array(hd[1:3], dtype=np.float32)

    # Combine into 20-dim feature vector
    feat = np.concatenate([pc_feat, track_feat, height_feat])
    return feat, current_velocity


def extract_recording_features(frames):
    """
    Process all frames in a recording → (T, 20) feature array.
    """
    features = []
    prev_vel = None
    for frame_obj in frames:
        fd = frame_obj.get("frameData", frame_obj)
        pc = fd.get("pointCloud", [])
        td = fd.get("trackData", [])
        hd = fd.get("heightData", [])
        
        feat, prev_vel = extract_frame_features(pc, td, hd, prev_vel)
        features.append(feat)
    return np.array(features, dtype=np.float32)


def build_sliding_windows(feature_buffer, window_size, stride):
    """
    Build all complete sliding windows from a growing feature buffer.

    Args:
        feature_buffer: list of np.array(12,) frames accumulated so far
        window_size: int, frames per window
        stride: int, frames to slide

    Returns:
        list of np.array(window_size, 12) — newly extractable windows
    """
    T = len(feature_buffer)
    windows = []
    if T < window_size:
        return windows
    # Extract all windows that are now completable
    for start in range(0, T - window_size + 1, stride):
        window = np.array(feature_buffer[start:start + window_size], dtype=np.float32)
        windows.append(window)
    return windows
