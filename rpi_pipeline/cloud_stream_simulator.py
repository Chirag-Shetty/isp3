"""
cloud_stream_simulator.py
-------------------------
Simulate the RPi by streaming features from a replay JSON file
to the cloud rule-based API (AWS).
"""

import argparse
import json
import time
from datetime import datetime, timezone

import requests

from config import CLOUD_API_URL, DEVICE_ID, CLOUD_TIMEOUT, FRAME_DT
from feature_extract import extract_frame_features


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_frames(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = []
    if isinstance(data, dict) and "data" in data:
        for row in data["data"]:
            fd = row.get("frameData", {})
            frames.append({
                "pointCloud": fd.get("pointCloud", []),
                "trackData": fd.get("trackData", []),
                "heightData": fd.get("heightData", []),
            })
    elif isinstance(data, dict) and "frameData" in data:
        fd = data["frameData"]
        frames.append({
            "pointCloud": fd.get("pointCloud", []),
            "trackData": fd.get("trackData", []),
            "heightData": fd.get("heightData", []),
        })
    elif isinstance(data, dict) and "pointCloud" in data:
        frames.append({
            "pointCloud": data.get("pointCloud", []),
            "trackData": data.get("trackData", []),
            "heightData": data.get("heightData", []),
        })
    elif isinstance(data, list):
        for item in data:
            fd = item.get("frameData", item)
            frames.append({
                "pointCloud": fd.get("pointCloud", []),
                "trackData": fd.get("trackData", []),
                "heightData": fd.get("heightData", []),
            })
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Stream a replay JSON file to the cloud rule-based API."
    )
    parser.add_argument("json_path", help="Path to replay_*.json file")
    parser.add_argument("--loop", action="store_true", help="Loop forever")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0x)")
    args = parser.parse_args()

    if "YOUR_EC2_IP" in CLOUD_API_URL:
        print("ERROR: Set CLOUD_API_URL in config.py or via env var CLOUD_API_URL")
        return

    frames = load_frames(args.json_path)
    if not frames:
        print("No frames found in JSON.")
        return

    print("=" * 60)
    print("  Cloud Stream Simulator")
    print("=" * 60)
    print(f"  API URL   : {CLOUD_API_URL}")
    print(f"  Device ID : {DEVICE_ID}")
    print(f"  Frames    : {len(frames)}")
    print(f"  Speed     : {args.speed}x")
    print("=" * 60)

    session = requests.Session()
    prev_velocity = None
    sent = 0

    while True:
        for frame_dict in frames:
            pc = frame_dict.get("pointCloud", [])
            td = frame_dict.get("trackData", [])
            hd = frame_dict.get("heightData", [])

            feat, prev_velocity = extract_frame_features(pc, td, hd, prev_velocity)
            payload = {
                "device_id": DEVICE_ID,
                "timestamp": now_iso(),
                "features": feat.tolist(),
            }

            try:
                session.post(CLOUD_API_URL, json=payload, timeout=CLOUD_TIMEOUT)
            except requests.exceptions.RequestException as exc:
                print(f"[cloud] send failed: {exc}")

            sent += 1
            if sent % 50 == 0:
                print(f"[cloud] sent frames: {sent}")

            if args.speed > 0:
                time.sleep(FRAME_DT / args.speed)

        if not args.loop:
            break


if __name__ == "__main__":
    main()
