"""
aws_watcher.py
--------------
Watches a folder for new JSON files produced by the TI mmWave Visualizer,
extracts per-frame features, and streams them to the AWS EC2 API.

This is the folder-watcher equivalent of cloud_stream_sender.py — use it
when the radar is NOT directly connected via serial (e.g. you are saving
JSON files from the TI GUI and want to stream them to AWS).

Usage (on the RPi or any machine with JSON files):
    python aws_watcher.py /path/to/json/output/folder

    # Process files already in the folder on startup:
    python aws_watcher.py ./json_output --process-existing

    # Faster polling:
    python aws_watcher.py ./json_output --poll-interval 0.2

Stop with Ctrl-C.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import json
import time
import glob
import argparse
from datetime import datetime, timezone

import numpy as np
import requests

from config import (
    CLOUD_API_URL,
    DEVICE_ID,
    CLOUD_TIMEOUT,
)
from feature_extract import extract_frame_features


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_visualizer_json(json_path: str):
    """
    Read a JSON file from the TI visualizer and return a list of frame dicts.

    Supports four formats:
      1. Wrapped:      { "data": [ { "frameData": { "pointCloud": [...] } }, ... ] }
      2. Single frame: { "frameData": { "pointCloud": [...] } }
      3. Direct:       { "pointCloud": [...] }
      4. List:         [ { "frameData": {...} }, ... ]
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  [watcher] Skipping {os.path.basename(json_path)}: invalid JSON ({e})")
        return []
    except PermissionError:
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
                    "and stream per-frame features to the AWS EC2 API."
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

    if "YOUR_EC2_IP" in CLOUD_API_URL:
        print("ERROR: Set CLOUD_API_URL in config.py or via env var CLOUD_API_URL")
        sys.exit(1)

    # ── Banner ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Real-Time JSON Watcher -> AWS EC2")
    print("=" * 60)
    print(f"  Watch dir    : {watch_dir}")
    print(f"  Poll interval: {args.poll_interval}s")
    print(f"  API URL      : {CLOUD_API_URL}")
    print(f"  Device ID    : {DEVICE_ID}")
    print("=" * 60)

    # ── HTTP session (reuse connection) ──────────────────────────────────────
    session = requests.Session()

    # ── Track which files we've already processed ────────────────────────────
    seen_files = set()
    if not args.process_existing:
        existing = glob.glob(os.path.join(watch_dir, "*.json"))
        seen_files = set(existing)
        print(f"\n[watcher] Skipping {len(seen_files)} existing file(s). "
              f"Waiting for new ones...")
    else:
        print(f"\n[watcher] Will process existing + new files...")

    # ── Pipeline state ───────────────────────────────────────────────────────
    prev_velocity  = None
    frame_count    = 0
    frames_sent    = 0
    files_processed = 0
    last_warn       = 0.0

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

                # ── Process each frame and send to AWS ───────────────────────
                for frame_dict in frames:
                    frame_count += 1

                    pc = frame_dict.get("pointCloud", [])
                    td = frame_dict.get("trackData", [])
                    hd = frame_dict.get("heightData", [])

                    feat, prev_velocity = extract_frame_features(pc, td, hd, prev_velocity)

                    payload = {
                        "device_id": DEVICE_ID,
                        "timestamp": now_iso(),
                        "features":  feat.tolist(),
                    }

                    try:
                        session.post(CLOUD_API_URL, json=payload, timeout=CLOUD_TIMEOUT)
                        frames_sent += 1
                    except requests.exceptions.RequestException as exc:
                        now_t = time.time()
                        if now_t - last_warn > 5:
                            print(f"  [cloud] send failed: {exc}")
                            last_warn = now_t

                    if frame_count % 50 == 0:
                        ts = now_iso()[11:19]
                        print(f"  [{ts}] frames processed: {frame_count}, sent: {frames_sent}")

                print(f"  [watcher] Done with {fname}: "
                      f"frames={frame_count}, sent={frames_sent}")

            # Sleep before next poll
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("  [watcher] Stopped.")
        print(f"  Files processed : {files_processed}")
        print(f"  Total frames    : {frame_count}")
        print(f"  Frames sent     : {frames_sent}")
        print("=" * 60)
        sys.exit(0)


if __name__ == '__main__':
    main()
