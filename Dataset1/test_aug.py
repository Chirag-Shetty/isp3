"""
mmWave Radar — Test Dataset Generator
======================================
Creates a FRESH test dataset the model has NEVER seen.

Strategy:
- Uses the SAME original JSON files as training
- BUT applies DIFFERENT augmentations (combinations + stronger params)
- NEVER uses the exact same aug+file combo that went into aug_dataset_binary/

New augmentations used for test set:
  combo_noise_mirror   : mirror + stronger noise
  combo_scale_rotate   : scale + rotate together
  combo_drop_doppler   : drop points + doppler shift
  stronger_noise       : higher sigma than training
  stronger_scale       : more aggressive scale (0.75-1.25)
  time_shift           : shift window start by random offset
  point_jitter         : per-point independent noise (different from global noise)
  flip_doppler_sign    : negate all doppler (person moving in opposite direction)

OUTPUT:
  test_dataset_binary/
    class_0_nofall/
    class_1_fall/
    test_manifest.csv
"""

import json
import os
import copy
import random
import csv
import math
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_ROOT = "./Dataset1"
OUTPUT_DIR   = "./test_dataset_binary"
RANDOM_SEED  = 99          # DIFFERENT seed from training (was 42)

random.seed(RANDOM_SEED)

FALL_PREFIXES   = ["fall_stand", "fall_walk"]
NOFALL_PREFIXES = ["sitchair_stand_tr", "sitting_chair",
                   "stand_sitchair_tr", "standing_still"]
FALL_LOOSE_FILES = {"1data.json", "2data.json", "4data.json"}

IDX_X, IDX_Y, IDX_Z, IDX_DOPPLER, IDX_SNR = 0, 1, 2, 3, 4


def get_label(folder_name):
    fl = folder_name.lower()
    for p in FALL_PREFIXES:
        if fl.startswith(p): return 1
    for p in NOFALL_PREFIXES:
        if fl.startswith(p): return 0
    return None

def load_frames(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"], raw
    elif isinstance(raw, list):
        return raw, {"data": raw}
    elif isinstance(raw, dict) and "frameData" in raw:
        return [raw], {"data": [raw]}
    raise ValueError(f"Unknown format: {path}")

def save_frames(path, template, frames):
    out = copy.deepcopy(template)
    if "data" in out:
        out["data"] = frames
    else:
        out = frames
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)

def get_points(frame): return frame.get("frameData", {}).get("pointCloud", [])
def set_points(frame, pts): frame["frameData"]["pointCloud"] = pts


# ─────────────────────────────────────────────
# TEST-ONLY AUGMENTATIONS
# (different from training augmentations)
# ─────────────────────────────────────────────

def aug_combo_noise_mirror(data):
    """Mirror X + stronger noise than training (sigma=0.04 vs 0.02)."""
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        for p in pts:
            p[IDX_X]       = -p[IDX_X] + random.gauss(0, 0.04)
            p[IDX_Y]       += random.gauss(0, 0.04)
            p[IDX_Z]       += random.gauss(0, 0.04)
            p[IDX_DOPPLER] = -p[IDX_DOPPLER] + random.gauss(0, 0.02)
        set_points(frame, pts)
    return d


def aug_combo_scale_rotate(data):
    """Scale XYZ AND rotate azimuth simultaneously."""
    scale = random.uniform(0.85, 1.15)
    angle = math.radians(random.uniform(-12.0, 12.0))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        for p in pts:
            # scale first
            p[IDX_X] *= scale; p[IDX_Y] *= scale; p[IDX_Z] *= scale
            # then rotate
            x, y = p[IDX_X], p[IDX_Y]
            p[IDX_X] = x * cos_a - y * sin_a
            p[IDX_Y] = x * sin_a + y * cos_a
        set_points(frame, pts)
    return d


def aug_combo_drop_doppler(data):
    """Drop 20% of points AND scale doppler."""
    drop_rate = 0.20
    dop_scale = random.uniform(0.80, 1.20)
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        if len(pts) > 4:
            pts = [p for p in pts if random.random() > drop_rate] or pts[:2]
        for p in pts:
            p[IDX_DOPPLER] *= dop_scale
        set_points(frame, pts)
        frame["frameData"]["numDetectedPoints"] = len(pts)
    return d


def aug_stronger_noise(data):
    """Much stronger Gaussian noise — sigma=0.05 (was 0.02 in training)."""
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        for p in pts:
            p[IDX_X]       += random.gauss(0, 0.05)
            p[IDX_Y]       += random.gauss(0, 0.05)
            p[IDX_Z]       += random.gauss(0, 0.03)
            p[IDX_DOPPLER] += random.gauss(0, 0.02)
        set_points(frame, pts)
    return d


def aug_stronger_scale(data):
    """More aggressive scale: 0.75 to 1.25 (was 0.88-1.12 in training)."""
    scale = random.uniform(0.75, 1.25)
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        for p in pts:
            p[IDX_X] *= scale; p[IDX_Y] *= scale; p[IDX_Z] *= scale
        set_points(frame, pts)
    return d


def aug_time_shift(data):
    """
    Skip the first N frames (1-10) — different temporal window.
    Simulates catching the activity mid-way through.
    """
    if len(data) < 15:
        return copy.deepcopy(data)
    shift = random.randint(1, min(10, len(data) - 10))
    return copy.deepcopy(data[shift:])


def aug_point_jitter(data):
    """
    Per-point INDEPENDENT noise (different magnitude per point).
    Different from global noise where all points shift the same way.
    """
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        for p in pts:
            # each point gets its own random sigma between 0.01 and 0.04
            s = random.uniform(0.01, 0.04)
            p[IDX_X]       += random.gauss(0, s)
            p[IDX_Y]       += random.gauss(0, s)
            p[IDX_Z]       += random.gauss(0, s * 0.5)
        set_points(frame, pts)
    return d


def aug_flip_doppler(data):
    """
    Negate all doppler values.
    Simulates radar mounted on the opposite side,
    or person facing the opposite direction.
    """
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        for p in pts:
            p[IDX_DOPPLER] = -p[IDX_DOPPLER]
        set_points(frame, pts)
    return d


TEST_AUGMENTATIONS = [
    ("combo_noise_mirror",  aug_combo_noise_mirror),
    ("combo_scale_rotate",  aug_combo_scale_rotate),
    ("combo_drop_doppler",  aug_combo_drop_doppler),
    ("stronger_noise",      aug_stronger_noise),
    ("stronger_scale",      aug_stronger_scale),
    ("time_shift",          aug_time_shift),
    ("point_jitter",        aug_point_jitter),
    ("flip_doppler",        aug_flip_doppler),
]


# ─────────────────────────────────────────────
# PROCESS ONE FILE
# ─────────────────────────────────────────────
def process_file(json_path, label, out_dir, stem_prefix, manifest_rows, stats):
    try:
        frames, template = load_frames(json_path)
    except Exception as e:
        print(f"    [ERROR] {json_path.name}: {e}")
        return

    label_name = "FALL" if label == 1 else "NO-FALL"
    print(f"    {json_path.name}: {len(frames)} frames → {label_name}")

    def save_and_record(aug_name, aug_frames):
        fname = f"test_{aug_name}_{stem_prefix}_{json_path.stem}.json"
        out_path = out_dir / fname
        save_frames(out_path, template, aug_frames)
        manifest_rows.append({
            "file":        str(out_path),
            "label":       label,
            "label_name":  label_name,
            "source":      str(json_path),
            "augmentation": aug_name,
            "n_frames":    len(aug_frames),
        })
        stats[label]["frames"] += len(aug_frames)

    for aug_name, aug_fn in TEST_AUGMENTATIONS:
        try:
            save_and_record(aug_name, aug_fn(frames))
            stats[label]["files"] += 1
        except Exception as e:
            print(f"    [WARN] {aug_name} failed: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run():
    root    = Path(DATASET_ROOT)
    out_dir = Path(OUTPUT_DIR)

    if not root.exists():
        print(f"ERROR: DATASET_ROOT not found: {root.resolve()}")
        return

    out_fall   = out_dir / "class_1_fall"
    out_nofall = out_dir / "class_0_nofall"
    out_fall.mkdir(parents=True, exist_ok=True)
    out_nofall.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    stats = {
        0: {"files": 0, "frames": 0},
        1: {"files": 0, "frames": 0},
    }

    print("\n=== GENERATING TEST DATASET (unseen augmentations) ===\n")

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        label = get_label(folder.name)
        if label is None:
            continue

        out_class  = out_fall if label == 1 else out_nofall
        json_files = sorted(folder.glob("*.json"))
        if not json_files:
            continue

        label_name = "FALL" if label == 1 else "NO-FALL"
        print(f"  [{folder.name}] → {label_name}")
        for jf in json_files:
            process_file(jf, label, out_class, folder.name, manifest_rows, stats)

    # Loose files
    for jf in sorted(root.glob("*.json")):
        if jf.name in FALL_LOOSE_FILES:
            print(f"\n  [{jf.name}] → FALL (loose file)")
            process_file(jf, 1, out_fall, "loose", manifest_rows, stats)

    # Save manifest
    manifest_path = out_dir / "test_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "label", "label_name", "source", "augmentation", "n_frames"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Summary
    print("\n" + "="*55)
    print("  TEST DATASET COMPLETE")
    print("="*55)
    print(f"  {'Class':<20} {'Files':>6} {'Frames':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*8}")
    for lbl, name in [(0, "NO-FALL"), (1, "FALL")]:
        s = stats[lbl]
        print(f"  {name:<20} {s['files']:>6} {s['frames']:>8}")
    total_f = stats[0]["files"]  + stats[1]["files"]
    total_fr= stats[0]["frames"] + stats[1]["frames"]
    print(f"  {'TOTAL':<20} {total_f:>6} {total_fr:>8}")
    print(f"\n  Output:   {out_dir.resolve()}")
    print(f"  Manifest: {manifest_path.resolve()}")
    print()
    print("  Augmentations used (ALL different from training):")
    for name, _ in TEST_AUGMENTATIONS:
        print(f"    - {name}")
    print()
    print("NEXT: Run evaluate_model.py to test your trained model on this data.")


if __name__ == "__main__":
    run()