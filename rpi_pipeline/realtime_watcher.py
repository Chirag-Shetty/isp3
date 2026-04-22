"""
realtime_watcher.py
-------------------
Watches a folder on the RPi for new JSON files produced by the
TI mmWave Visualizer in real time. Each time a new .json file
appears, it reads the frames and feeds them through the full
pipeline (feature extraction -> inference -> Supabase upload).

The feature buffer is maintained across files so sliding windows
span continuously across recordings.

Usage (on the RPi):
    python realtime_watcher.py /path/to/visualizer/output/

    # Optional flags:
    python realtime_watcher.py ./json_output --poll-interval 0.5 --process-existing

Stop with Ctrl-C.
"""

import sys
# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
import json
import time
import glob
import argparse
import pickle
from collections import deque
from datetime import datetime, timezone

import numpy as np
import requests
import torch

from config import (
    SUPABASE_URL, SUPABASE_ANON_KEY,
    WINDOW_SIZE, STRIDE, NUM_FEATURES,
    MODEL_PATH, SCALER_PATH, RUN_INFERENCE,
    CLASSES, FALL_CLASS_IDS,
    INFERENCE_URL,
)
from feature_extract import extract_frame_features

# ── Cloud vs local inference ────────────────────────────────────────────────
_use_cloud = bool(INFERENCE_URL and "YOUR_HF_SPACE" not in INFERENCE_URL)
if RUN_INFERENCE and not _use_cloud:
    from model import TransformerCNNLSTM


# ══════════════════════════════════════════════════════════════════════════════
#  Supabase helpers (same as rpi_sender.py)
# ══════════════════════════════════════════════════════════════════════════════

HEADERS = {
    "apikey":        SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal",
}


def send_to_supabase(table: str, payload: dict) -> bool:
    """POST a single row to a Supabase table via REST API."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=5)
        if r.status_code in (200, 201):
            return True
        print(f"  [cloud] Supabase error {r.status_code}: {r.text[:120]}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"  [cloud] Network error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Model loader (same as rpi_sender.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_scaler():
    """Load the trained .pth model and sklearn scaler from disk."""
    device = torch.device("cpu")
    model = TransformerCNNLSTM(num_features=NUM_FEATURES, num_classes=len(CLASSES))
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        state_dict = checkpoint.get('sd') or checkpoint.get('model_state_dict')
        epoch      = checkpoint.get('ep') or checkpoint.get('epoch', '?')
        val_acc    = checkpoint.get('va') or checkpoint.get('val_acc', 0)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"[model] Loaded weights from {MODEL_PATH}  "
              f"(epoch {epoch}, val_acc={val_acc:.2%})")
    except FileNotFoundError:
        print(f"[model] WARNING: {MODEL_PATH} not found — running without weights.")
        model.eval()

    scaler = None
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[model] Loaded scaler from {SCALER_PATH}")
    except FileNotFoundError:
        print(f"[model] WARNING: {SCALER_PATH} not found — features will NOT be scaled.")

    return model.to(device), scaler, device


def run_inference(model, scaler, device, window: np.ndarray):
    """
    Run inference — cloud API or local model.
    Returns dict with class_id, class_name, confidence, is_fall, probs.
    Returns {} on failure.
    """
    if _use_cloud:
        try:
            payload = {"window": window.tolist()}
            r = requests.post(INFERENCE_URL, json=payload, timeout=10)
            if r.status_code == 200:
                data = r.json()
                return {
                    "class_id":   data["class_id"],
                    "class_name": data["class_name"],
                    "confidence": data["confidence"],
                    "is_fall":    data["is_fall"],
                    "probs":      data["probs"],
                }
            else:
                print(f"  [cloud-infer] API error {r.status_code}: {r.text[:120]}")
                return {}
        except requests.exceptions.RequestException as e:
            print(f"  [cloud-infer] Network error: {e}")
            return {}

    # Local inference fallback
    W, F = window.shape
    if scaler is not None:
        flat = window.reshape(-1, F)
        flat = scaler.transform(flat)
        window = flat.reshape(W, F)

    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model.predict_proba(x)[0].cpu().numpy()

    class_id = int(np.argmax(probs))
    return {
        "class_id":   class_id,
        "class_name": CLASSES[class_id],
        "confidence": float(probs[class_id]),
        "is_fall":    class_id in FALL_CLASS_IDS,
        "probs":      probs.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  JSON file processing
# ══════════════════════════════════════════════════════════════════════════════

def read_visualizer_json(json_path: str):
    """
    Read a JSON file from the TI visualizer and yield frame dicts.
    
    Supports two formats:
      1. Wrapped: { "data": [ { "frameData": { "pointCloud": [...] } }, ... ] }
      2. Single frame: { "frameData": { "pointCloud": [...] } }
      3. Direct: { "pointCloud": [...] }
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  [watcher] Skipping {os.path.basename(json_path)}: invalid JSON ({e})")
        return []
    except PermissionError:
        # File may still be being written
        print(f"  [watcher] Skipping {os.path.basename(json_path)}: file locked (still writing?)")
        return []

    frames = []

    # Format 1: Full recording with 'data' array (replay_*.json format)
    if isinstance(data, dict) and 'data' in data:
        for row in data['data']:
            fd = row.get("frameData", {})
            frames.append({
                "pointCloud": fd.get("pointCloud", []),
                "trackData":  fd.get("trackData", []),
                "heightData": fd.get("heightData", []),
            })

    # Format 2: Single frame with frameData wrapper
    elif isinstance(data, dict) and 'frameData' in data:
        fd = data['frameData']
        frames.append({
            "pointCloud": fd.get("pointCloud", []),
            "trackData":  fd.get("trackData", []),
            "heightData": fd.get("heightData", []),
        })

    # Format 3: Direct frame dict with pointCloud at top level
    elif isinstance(data, dict) and 'pointCloud' in data:
        frames.append({
            "pointCloud": data.get("pointCloud", []),
            "trackData":  data.get("trackData", []),
            "heightData": data.get("heightData", []),
        })

    # Format 4: Array of frames
    elif isinstance(data, list):
        for item in data:
            fd = item.get("frameData", item)
            frames.append({
                "pointCloud": fd.get("pointCloud", []),
                "trackData":  fd.get("trackData", []),
                "heightData": fd.get("heightData", []),
            })

    else:
        print(f"  [watcher] Unknown JSON format in {os.path.basename(json_path)}")

    return frames


# ══════════════════════════════════════════════════════════════════════════════
#  Main watcher loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Watch a folder for new JSON files from the TI visualizer "
                    "and run the fall-detection pipeline in real time."
    )
    parser.add_argument(
        "watch_dir",
        help="Path to the folder where the visualizer saves JSON files."
    )
    parser.add_argument(
        "--poll-interval", type=float, default=0.5,
        help="Seconds between folder scans (default: 0.5s)"
    )
    parser.add_argument(
        "--process-existing", action="store_true",
        help="Process JSON files already in the folder on startup "
             "(default: only process new files that appear after start)"
    )
    args = parser.parse_args()

    watch_dir = os.path.abspath(args.watch_dir)
    if not os.path.isdir(watch_dir):
        print(f"ERROR: Watch directory does not exist: {watch_dir}")
        print("       Create the folder or check the path.")
        sys.exit(1)

    # ── Banner ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Real-Time JSON Watcher -> Pipeline")
    print("=" * 60)
    print(f"  Watch dir    : {watch_dir}")
    print(f"  Poll interval: {args.poll_interval}s")
    print(f"  Supabase URL : {SUPABASE_URL}")

    inference_mode = "Cloud API" if _use_cloud else ("Local" if RUN_INFERENCE else "OFF")
    print(f"  Inference    : {inference_mode}")
    if _use_cloud:
        print(f"  Cloud URL    : {INFERENCE_URL}")
    print("=" * 60)

    # ── Load model (only for local inference) ────────────────────────────────
    model, scaler, device = (None, None, None)
    if RUN_INFERENCE and not _use_cloud:
        model, scaler, device = load_model_and_scaler()

    # ── Track which files we've already processed ────────────────────────────
    seen_files = set()
    if not args.process_existing:
        # Mark all existing files as "seen" so we skip them
        existing = glob.glob(os.path.join(watch_dir, "*.json"))
        seen_files = set(existing)
        print(f"\n[watcher] Skipping {len(seen_files)} existing file(s). "
              f"Waiting for new ones...")
    else:
        print(f"\n[watcher] Will process existing + new files...")

    # ── Pipeline state (persists across files) ───────────────────────────────
    feature_buffer = []
    prev_velocity  = None
    frame_count    = 0
    windows_sent   = 0
    files_processed = 0
    session_start  = datetime.now(timezone.utc).isoformat()

    print("[watcher] Watching for new JSON files... (Ctrl-C to stop)\n")

    try:
        while True:
            # Scan for .json files
            current_files = set(glob.glob(os.path.join(watch_dir, "*.json")))
            new_files = sorted(current_files - seen_files)  # sorted by name for order

            for json_path in new_files:
                seen_files.add(json_path)

                # Small delay to let the file finish writing
                time.sleep(0.1)

                fname = os.path.basename(json_path)
                frames = read_visualizer_json(json_path)

                if not frames:
                    print(f"  [watcher] {fname}: no frames found, skipping.")
                    continue

                files_processed += 1
                print(f"\n{'-'*50}")
                print(f"  [watcher] New file: {fname} ({len(frames)} frames)")
                print(f"{'-'*50}")

                # ── Process each frame ───────────────────────────────────────
                for frame_dict in frames:
                    frame_count += 1

                    pc = frame_dict.get("pointCloud", [])
                    td = frame_dict.get("trackData", [])
                    hd = frame_dict.get("heightData", [])

                    feat, prev_velocity = extract_frame_features(pc, td, hd, prev_velocity)
                    feature_buffer.append(feat)

                    # Check if we have enough frames for a window
                    if len(feature_buffer) < WINDOW_SIZE:
                        continue

                    if (len(feature_buffer) - WINDOW_SIZE) % STRIDE != 0:
                        continue

                    # Build window
                    window = np.array(feature_buffer[-WINDOW_SIZE:], dtype=np.float32)
                    windows_sent += 1
                    ts = datetime.now(timezone.utc).isoformat()

                    # ── Run inference ────────────────────────────────────────
                    result = {}
                    if RUN_INFERENCE and (_use_cloud or model is not None):
                        result = run_inference(model, scaler, device, window)

                        if not result:
                            print(f"  [{ts[11:19]}] frame={frame_count:05d} | "
                                  f"[infer] No response — skipping window #{windows_sent}")
                            continue

                        label = result["class_name"]
                        conf  = result["confidence"]
                        fall  = "[FALL DETECTED]" if result["is_fall"] else "[SAFE]"
                        print(f"  [{ts[11:19]}] frame={frame_count:05d} | "
                              f"pred={label:<15s} conf={conf:.1%}  {fall}")
                    else:
                        n_pts = len(pc)
                        print(f"  [{ts[11:19]}] frame={frame_count:05d} | "
                              f"{n_pts} points | window #{windows_sent}")

                    # ── Build & send Supabase payload ───────────────────────
                    payload = {
                        "timestamp":       ts,
                        "session_start":   session_start,
                        "frame_count":     frame_count,
                        "window_index":    windows_sent,
                        "n_points":        int(len(pc)),
                        "source_file":     fname,
                        "x_mean":          float(feat[0]),
                        "y_mean":          float(feat[1]),
                        "z_mean":          float(feat[2]),
                        "height_range":    float(feat[11]),
                        "window_features": window.tolist(),
                    }

                    if result:
                        payload.update({
                            "class_id":   result["class_id"],
                            "class_name": result["class_name"],
                            "confidence": result["confidence"],
                            "is_fall":    result["is_fall"],
                            "probs":      result["probs"],
                        })

                    ok = send_to_supabase("radar_predictions", payload)
                    if not ok:
                        print("  [cloud] WARNING: Failed to send — will retry next window")

                    # Keep buffer manageable
                    if len(feature_buffer) > WINDOW_SIZE * 4:
                        feature_buffer = feature_buffer[-WINDOW_SIZE:]

                print(f"  [watcher] Done with {fname}: "
                      f"total frames={frame_count}, windows={windows_sent}")

            # Sleep before next poll
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("  [watcher] Stopped.")
        print(f"  Files processed : {files_processed}")
        print(f"  Total frames    : {frame_count}")
        print(f"  Windows sent    : {windows_sent}")
        print("=" * 60)
        sys.exit(0)


if __name__ == '__main__':
    main()
