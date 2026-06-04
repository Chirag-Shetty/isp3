"""
aws_watcher.py
--------------
Watches a folder for new JSON files from the TI mmWave Visualizer and
streams per-frame features to the AWS EC2 API immediately as files appear.

Uses inotify (Linux) via the `watchdog` library for instant file detection
— no polling delay. Falls back to fast polling if watchdog is unavailable.

Usage:
    pip install watchdog requests numpy
    python aws_watcher.py /path/to/radar/json/folder

    # Also process files already in the folder:
    python aws_watcher.py /path/to/folder --process-existing

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
import threading
import queue
from datetime import datetime, timezone

import numpy as np
import requests

from config import CLOUD_API_URL, DEVICE_ID, CLOUD_TIMEOUT
from feature_extract import extract_frame_features


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_visualizer_json(json_path: str):
    """Read a TI visualizer JSON file and return a list of frame dicts."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  [watcher] Skipping {os.path.basename(json_path)}: invalid JSON ({e})")
        return []
    except PermissionError:
        print(f"  [watcher] Skipping {os.path.basename(json_path)}: file locked")
        return []

    frames = []

    if isinstance(data, dict) and 'data' in data:
        for row in data['data']:
            fd = row.get("frameData", {})
            frames.append({
                "pointCloud":        fd.get("pointCloud", []),
                "trackData":         fd.get("trackData", []),
                "heightData":        fd.get("heightData", []),
                "numDetectedTracks": int(fd.get("numDetectedTracks", 0)),
            })
    elif isinstance(data, dict) and 'frameData' in data:
        fd = data['frameData']
        frames.append({
            "pointCloud":        fd.get("pointCloud", []),
            "trackData":         fd.get("trackData", []),
            "heightData":        fd.get("heightData", []),
            "numDetectedTracks": int(fd.get("numDetectedTracks", 0)),
        })
    elif isinstance(data, dict) and 'pointCloud' in data:
        frames.append({
            "pointCloud":        data.get("pointCloud", []),
            "trackData":         data.get("trackData", []),
            "heightData":        data.get("heightData", []),
            "numDetectedTracks": int(data.get("numDetectedTracks", 0)),
        })
    elif isinstance(data, list):
        for item in data:
            fd = item.get("frameData", item)
            frames.append({
                "pointCloud":        fd.get("pointCloud", []),
                "trackData":         fd.get("trackData", []),
                "heightData":        fd.get("heightData", []),
                "numDetectedTracks": int(fd.get("numDetectedTracks", 0)),
            })
    else:
        print(f"  [watcher] Unknown JSON format in {os.path.basename(json_path)}")
    return frames


def process_file(json_path, session, prev_velocity_ref, counters):
    """Extract features from every frame in a JSON file and POST as one batch to AWS."""
    fname  = os.path.basename(json_path)
    frames = read_visualizer_json(json_path)
    if not frames:
        return

    counters['files'] += 1

    # ── Build batch payload ───────────────────────────────────────────────────
    batch_frames = []
    for frame_dict in frames:
        counters['frames'] += 1
        pc = frame_dict.get("pointCloud", [])
        td = frame_dict.get("trackData",  [])
        hd = frame_dict.get("heightData", [])
        hc = frame_dict.get("numDetectedTracks", 0)

        feat, prev_velocity_ref[0] = extract_frame_features(
            pc, td, hd, prev_velocity_ref[0]
        )
        batch_frames.append({
            "features":    feat.tolist(),
            "timestamp":   now_iso(),
            "human_count": hc,
        })

    # ── Send entire file as one HTTP request ──────────────────────────────────
    batch_url = CLOUD_API_URL.replace("/frame", "/frames/batch")
    payload   = {"device_id": DEVICE_ID, "frames": batch_frames}

    t0 = time.perf_counter()
    try:
        session.post(batch_url, json=payload, timeout=CLOUD_TIMEOUT * 10)
        counters['sent'] += len(batch_frames)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [cloud] {fname}  {len(batch_frames)} frames → 1 request  {ms:.0f}ms  "
              f"(total sent: {counters['sent']})")
    except requests.exceptions.RequestException as exc:
        counters['errors'] += 1
        print(f"  [cloud] batch send failed: {exc}")


def wait_for_stable_file(path, stable_ms=80, max_wait_s=15):
    """
    Wait until the file size stops changing — i.e. the writer has finished.
    Polls every `stable_ms` milliseconds. Returns True when stable, False on timeout.
    """
    last_size = -1
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            time.sleep(0.05)
            continue
        if size > 0 and size == last_size:
            return True   # size unchanged — file is fully written
        last_size = size
        time.sleep(stable_ms / 1000.0)
    return False  # timed out


def run_with_watchdog(watch_dir, session, prev_velocity_ref, counters, seen_files):
    """Use watchdog (inotify on Linux) for zero-delay file detection."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    file_queue = queue.Queue()

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith('.json'):
                file_queue.put(event.src_path)

        def on_moved(self, event):
            # Some apps write to .tmp then rename to .json
            if not event.is_directory and event.dest_path.endswith('.json'):
                file_queue.put(event.dest_path)

    observer = Observer()
    observer.schedule(Handler(), watch_dir, recursive=False)
    observer.start()
    print("[watcher] inotify active — zero-delay detection ✓")

    try:
        while True:
            try:
                path = file_queue.get(timeout=1.0)
                if path in seen_files:
                    continue
                seen_files.add(path)
                if wait_for_stable_file(path):
                    process_file(path, session, prev_velocity_ref, counters)
                else:
                    print(f"  [watcher] Timed out waiting for {os.path.basename(path)} — skipping")
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ── Fallback fast-polling watcher ─────────────────────────────────────────────

def run_with_polling(watch_dir, session, prev_velocity_ref, counters, seen_files,
                     poll_interval=0.1):
    """Poll every 100ms as a fallback when watchdog isn't available."""
    print(f"[watcher] polling every {poll_interval*1000:.0f}ms (install watchdog for instant detection)")
    try:
        while True:
            current = set(glob.glob(os.path.join(watch_dir, "*.json")))
            for path in sorted(current - seen_files):
                seen_files.add(path)
                time.sleep(0.02)
                process_file(path, session, prev_velocity_ref, counters)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Watch a folder for new JSON files and stream features to AWS instantly."
    )
    parser.add_argument("watch_dir", help="Folder where TI visualizer saves JSON files")
    parser.add_argument("--process-existing", action="store_true",
                        help="Also process JSON files already in the folder on startup")
    parser.add_argument("--poll-interval", type=float, default=0.1,
                        help="Polling fallback interval in seconds (default 0.1)")
    args = parser.parse_args()

    watch_dir = os.path.abspath(args.watch_dir)
    if not os.path.isdir(watch_dir):
        print(f"ERROR: Directory does not exist: {watch_dir}")
        sys.exit(1)

    if "YOUR_EC2_IP" in CLOUD_API_URL:
        print("ERROR: Set CLOUD_API_URL in config.py")
        sys.exit(1)

    print("=" * 60)
    print("  Real-Time Watcher → AWS EC2")
    print("=" * 60)
    print(f"  Watch dir : {watch_dir}")
    print(f"  API URL   : {CLOUD_API_URL}")
    print(f"  Device ID : {DEVICE_ID}")
    print("=" * 60)

    session          = requests.Session()
    prev_velocity    = [None]   # mutable ref so process_file can update it
    counters         = {'files': 0, 'frames': 0, 'sent': 0, 'errors': 0}

    # Mark existing files as seen (skip them unless --process-existing)
    seen_files = set()
    existing = set(glob.glob(os.path.join(watch_dir, "*.json")))
    if args.process_existing:
        print(f"[watcher] Processing {len(existing)} existing file(s) first...")
        for path in sorted(existing):
            seen_files.add(path)
            process_file(path, session, prev_velocity, counters)
    else:
        seen_files = existing
        print(f"[watcher] Skipping {len(seen_files)} existing file(s). Waiting for new ones...")

    print("[watcher] Ready — waiting for new JSON files (Ctrl-C to stop)\n")

    # Try inotify first, fall back to polling
    try:
        import watchdog
        run_with_watchdog(watch_dir, session, prev_velocity, counters, seen_files)
    except ImportError:
        print("[watcher] watchdog not installed — using polling fallback")
        print("[watcher] For instant detection: pip install watchdog")
        run_with_polling(watch_dir, session, prev_velocity, counters, seen_files,
                         args.poll_interval)

    print("\n" + "=" * 60)
    print(f"  Files processed : {counters['files']}")
    print(f"  Total frames    : {counters['frames']}")
    print(f"  Frames sent     : {counters['sent']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
