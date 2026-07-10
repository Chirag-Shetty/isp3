"""
cloud_stream_sender.py
----------------------
Streams per-frame features to the cloud rule-based API (AWS).
Use this when inference runs in the cloud and you want <1s latency.
"""

import time
from datetime import datetime, timezone

import requests

from config import (
    SERIAL_PORT_DATA,
    SERIAL_PORT_CFG,
    CLOUD_API_URL,
    DEVICE_ID,
    CLOUD_TIMEOUT,
)
from feature_extract import extract_frame_features
from radar_capture import send_config, frame_generator


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def main():
    if "YOUR_EC2_IP" in CLOUD_API_URL:
        print("ERROR: Set CLOUD_API_URL in config.py or via env var CLOUD_API_URL")
        return

    print("=" * 60)
    print("  RPi -> Cloud Stream Sender")
    print("=" * 60)
    print(f"  API URL   : {CLOUD_API_URL}")
    print(f"  Device ID : {DEVICE_ID}")
    print(f"  Serial    : {SERIAL_PORT_DATA}")
    print("=" * 60)

    send_config(SERIAL_PORT_CFG)

    session = requests.Session()
    prev_velocity = None
    frame_count = 0
    last_warn = 0.0

    for frame_dict in frame_generator(SERIAL_PORT_DATA):
        frame_count += 1
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
            now = time.time()
            if now - last_warn > 5:
                print(f"[cloud] send failed: {exc}")
                last_warn = now

        if frame_count % 50 == 0:
            print(f"[cloud] sent frames: {frame_count}")


if __name__ == "__main__":
    main()
