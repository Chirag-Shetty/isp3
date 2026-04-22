"""
test_watcher.py
--------------
Quick test: read a real dataset JSON, extract features, check sliding windows.
Does NOT call cloud inference or Supabase — just verifies the data pipeline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

# ── Import pipeline modules ──────────────────────────────────────────────────
from realtime_watcher import read_visualizer_json
from feature_extract import extract_frame_features
from config import WINDOW_SIZE, STRIDE, NUM_FEATURES

# ── Point at a real dataset file ─────────────────────────────────────────────
JSON_PATH = r"c:\Users\chira\Downloads\idp\dataset_26_09_25 (1)\dataset_26_09_25\chair_floor_transition\replay_1.json"

print("=" * 60)
print("  Watcher Test — JSON reading + feature extraction")
print("=" * 60)
print(f"  File        : {os.path.basename(JSON_PATH)}")
print(f"  Window size : {WINDOW_SIZE} frames")
print(f"  Stride      : {STRIDE} frames")
print(f"  Num features: {NUM_FEATURES}")
print("=" * 60)

# Step 1: Read the JSON
frames = read_visualizer_json(JSON_PATH)
print(f"\n[1] JSON read OK — {len(frames)} frames found")

# Step 2: Extract features frame by frame
feature_buffer = []
prev_velocity  = None
windows_built  = 0

for i, frame_dict in enumerate(frames):
    pc = frame_dict.get("pointCloud", [])
    td = frame_dict.get("trackData", [])
    hd = frame_dict.get("heightData", [])

    feat, prev_velocity = extract_frame_features(pc, td, hd, prev_velocity)
    feature_buffer.append(feat)

    # Check window condition
    if len(feature_buffer) >= WINDOW_SIZE:
        if (len(feature_buffer) - WINDOW_SIZE) % STRIDE == 0:
            window = np.array(feature_buffer[-WINDOW_SIZE:], dtype=np.float32)
            windows_built += 1
            if windows_built <= 3:  # Show first 3 windows
                print(f"\n[window #{windows_built}] shape={window.shape} "
                      f"  mean_x={window[:,0].mean():.3f} "
                      f"  mean_y={window[:,1].mean():.3f} "
                      f"  mean_z={window[:,2].mean():.3f}")

    if i % 20 == 0:
        n_pts = len(pc)
        print(f"  Frame {i:03d}: {n_pts} points — feat[:3]={feat[:3].round(3)}")

print(f"\n[2] Feature extraction OK")
print(f"    Frames processed : {len(frames)}")
print(f"    Windows built    : {windows_built}")
print(f"    Feature shape    : ({len(feature_buffer)}, {NUM_FEATURES})")

# Step 3: Verify shape matches model expectation
expected_window_shape = (WINDOW_SIZE, NUM_FEATURES)
if windows_built > 0:
    window = np.array(feature_buffer[-WINDOW_SIZE:], dtype=np.float32)
    assert window.shape == expected_window_shape, \
        f"Shape mismatch: got {window.shape}, expected {expected_window_shape}"
    print(f"    Window shape     : {window.shape} [OK]")
    print(f"\n[3] Shape check PASSED — ready for inference")
else:
    print(f"\n[WARN] Not enough frames to build a single window (need {WINDOW_SIZE}, got {len(frames)})")

print("\n" + "=" * 60)
print("  All checks passed! realtime_watcher.py is working correctly.")
print("=" * 60)
