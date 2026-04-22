"""
rpi_sender.py
-------------
Main pipeline script. Run this on the Raspberry Pi.

What it does:
  1. Configures the IWR6843 sensor
  2. Reads radar frames in real-time
  3. Extracts 12-dim features per frame
  4. Batches frames into sliding windows of 40
  5. Runs Transformer-CNN-LSTM inference (on-device)
  6. Sends predictions + features to Supabase cloud

Usage:
    python rpi_sender.py

Stop with Ctrl-C.
"""

import sys
import time
import pickle
import json
from collections import deque
from datetime import datetime, timezone

import numpy as np
import requests
import torch

from config import (
    SUPABASE_URL, SUPABASE_ANON_KEY,
    SERIAL_PORT_DATA, SERIAL_PORT_CFG,
    WINDOW_SIZE, STRIDE, NUM_FEATURES,
    MODEL_PATH, SCALER_PATH, RUN_INFERENCE,
    CLASSES, FALL_CLASS_IDS,
    INFERENCE_URL,
)
from feature_extract import extract_frame_features
from radar_capture import send_config, frame_generator

# ── Only import torch/model if cloud URL is empty (local fallback) ────────────
_use_cloud = bool(INFERENCE_URL and "YOUR_HF_SPACE" not in INFERENCE_URL)
if RUN_INFERENCE and not _use_cloud:
    from model import TransformerCNNLSTM


# ══════════════════════════════════════════════════════════════════════════════
#  Supabase helpers
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
#  Model loader
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_scaler():
    """Load the trained .pth model and sklearn scaler from disk."""
    device = torch.device("cpu")           # RPi uses CPU
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
    Run inference — either via the cloud API (Hugging Face Space)
    or locally on the RPi if INFERENCE_URL is not set.

    Returns:
        class_id  : int
        class_name: str
        confidence: float (0..1)
        is_fall   : bool
        probs     : list of floats (one per class)
    """

    # ── Cloud inference (preferred) ──────────────────────────────────────────
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

    # ── Local inference fallback (RPi runs PyTorch) ──────────────────────────
    W, F = window.shape
    if scaler is not None:
        flat = window.reshape(-1, F)
        flat = scaler.transform(flat)
        window = flat.reshape(W, F)

    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, W, F)
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
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  IWR6843 -> Supabase Fall Detection Pipeline")
    print("=" * 60)
    print(f"  Supabase URL : {SUPABASE_URL}")
    print(f"  Serial port  : {SERIAL_PORT_DATA}")
    print(f"  Inference    : {'ON  (' + MODEL_PATH + ')' if RUN_INFERENCE else 'OFF (raw features only)'}")
    print("=" * 60)

    # Load model (only if running in local fallback mode)
    model, scaler, device = (None, None, None)
    if RUN_INFERENCE and not _use_cloud:
        model, scaler, device = load_model_and_scaler()

    inference_mode = "Cloud API" if _use_cloud else ("Local RPi" if RUN_INFERENCE else "OFF")
    print(f"  Inference    : {inference_mode}")
    if _use_cloud:
        print(f"  Cloud URL    : {INFERENCE_URL}")

    # Configure sensor
    send_config(SERIAL_PORT_CFG)

    # Feature buffer — grows with each frame, windowed every STRIDE frames
    feature_buffer = []   # list of np.array(12,)
    prev_velocity  = None
    frame_count    = 0
    windows_sent   = 0
    session_start  = datetime.now(timezone.utc).isoformat()

    print("\n[pipeline] Streaming ... (Ctrl-C to stop)\n")

    try:
        for frame_dict in frame_generator(SERIAL_PORT_DATA):
            frame_count += 1

            # ── 1. Extract features for this frame ──────────────────────────
            pc = frame_dict.get("pointCloud", [])
            td = frame_dict.get("trackData", [])
            hd = frame_dict.get("heightData", [])
            
            feat, prev_velocity = extract_frame_features(pc, td, hd, prev_velocity)
            feature_buffer.append(feat)

            # ── 2. Check if we have enough frames for a window ──────────────
            #    We extract a new window every STRIDE frames once we have WINDOW_SIZE
            if len(feature_buffer) < WINDOW_SIZE:
                continue

            if (len(feature_buffer) - WINDOW_SIZE) % STRIDE != 0:
                continue

            # The latest complete window
            window = np.array(feature_buffer[-WINDOW_SIZE:], dtype=np.float32)
            windows_sent += 1
            ts = datetime.now(timezone.utc).isoformat()

            # ── 3. Run inference ─────────────────────────────────────────────
            result = {}
            if RUN_INFERENCE and (_use_cloud or model is not None):
                result = run_inference(model, scaler, device, window)

                if not result:
                    print(f"  [{ts[11:19]}] frame={frame_count:05d} | "
                          f"[cloud-infer] No response (timeout?) — skipping window #{windows_sent}")
                    continue

                label = result["class_name"]
                conf  = result["confidence"]
                fall  = "[FALL DETECTED]" if result["is_fall"] else "[SAFE] normal"
                print(f"  [{ts[11:19]}] frame={frame_count:05d} | "
                      f"pred={label:<15s} conf={conf:.1%}  {fall}")

            else:
                # No inference — just show point count
                n_pts = len(pc)
                print(f"  [{ts[11:19]}] frame={frame_count:05d} | "
                      f"{n_pts} points | window #{windows_sent}")

            # ── 4. Build Supabase payload ────────────────────────────────────
            payload = {
                "timestamp":      ts,
                "session_start":  session_start,
                "frame_count":    frame_count,
                "window_index":   windows_sent,
                "n_points":       int(len(pc)),

                # Feature summary (last frame centroid)
                "x_mean":         float(feat[0]),
                "y_mean":         float(feat[1]),
                "z_mean":         float(feat[2]),
                "height_range":   float(feat[11]),

                # Full window as JSON (for re-inference later)
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

            # ── 5. Send to Supabase ──────────────────────────────────────────
            ok = send_to_supabase("radar_predictions", payload)
            if not ok:
                print("  [cloud] WARNING: Failed to send - will retry next window")

            # ── 6. Keep buffer from growing too large ────────────────────────
            # Keep only the last WINDOW_SIZE frames to avoid memory creep
            if len(feature_buffer) > WINDOW_SIZE * 4:
                feature_buffer = feature_buffer[-WINDOW_SIZE:]

    except KeyboardInterrupt:
        print(f"\n[pipeline] Stopped. Sent {windows_sent} windows over {frame_count} frames.")
        sys.exit(0)


if __name__ == "__main__":
    # Quick config check before starting
    if "YOUR_PROJECT_ID" in SUPABASE_URL:
        print("ERROR: Please edit config.py and fill in your Supabase URL and key.")
        print("       Sign up free at https://supabase.com")
        sys.exit(1)

    main()
