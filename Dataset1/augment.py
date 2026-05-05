"""
mmWave Radar Binary Augmentation Script
=======================================
2-Class Problem: FALL (1) vs NO-FALL (0)

FALL folders:
  fall_stand, fall_stand1, fall_stand2
  fall_walk, fall_walk1
  1data.json, 2data.json, 4data.json  ← loose files directly in Dataset1/

NO-FALL folders:
  sitchair_stand_tr, sitchair_stand_tr1
  sitting_chair
  stand_sitchair_tr, stand_sitchair_tr1
  standing_still

OUTPUT:
  aug_dataset_binary/
    class_0_nofall/   ← all no-fall files (original + augmented)
    class_1_fall/     ← all fall files    (original + augmented)
    dataset_manifest.csv
"""

import json
import os
import copy
import random
import csv
import math
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION — edit DATASET_ROOT if needed
# ─────────────────────────────────────────────
DATASET_ROOT = "./Dataset1"        # folder containing all subfolders + loose JSONs
OUTPUT_DIR   = "./aug_dataset_binary"
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# FALL / NO-FALL folder prefix mapping
# ─────────────────────────────────────────────
FALL_PREFIXES = [
    "fall_stand",
    "fall_walk",
]

NOFALL_PREFIXES = [
    "sitchair_stand_tr",
    "sitting_chair",
    "stand_sitchair_tr",
    "standing_still",
]

# Loose JSON files directly in Dataset1/ root that are FALL
FALL_LOOSE_FILES = {"1data.json", "2data.json", "4data.json"}


def get_label_for_folder(folder_name: str):
    """Returns 1 (FALL), 0 (NO-FALL), or None (skip)."""
    fl = folder_name.lower()
    for p in FALL_PREFIXES:
        if fl.startswith(p.lower()):
            return 1
    for p in NOFALL_PREFIXES:
        if fl.startswith(p.lower()):
            return 0
    return None


# ─────────────────────────────────────────────
# POINT CLOUD INDICES
# [x, y, z, doppler, snr, field6, trackIndex]
# ─────────────────────────────────────────────
IDX_X, IDX_Y, IDX_Z = 0, 1, 2
IDX_DOPPLER = 3
IDX_SNR = 4


# ─────────────────────────────────────────────
# JSON LOAD / SAVE
# ─────────────────────────────────────────────
def load_frames(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"], raw
    elif isinstance(raw, list):
        return raw, {"data": raw}
    elif isinstance(raw, dict) and "frameData" in raw:
        return [raw], {"data": [raw]}
    else:
        raise ValueError(f"Unknown format: {path}")

def save_frames(path: Path, template: dict, frames: list):
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
# AUGMENTATION FUNCTIONS
# ─────────────────────────────────────────────
def aug_mirror_x(data):
    """Flip person left↔right. Negate X and doppler."""
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        for p in pts:
            p[IDX_X] = -p[IDX_X]
            p[IDX_DOPPLER] = -p[IDX_DOPPLER]
        set_points(frame, pts)
        for t in frame.get("frameData", {}).get("trackData", []):
            if len(t) > 4: t[1] = -t[1]; t[4] = -t[4]
    return d

def aug_gaussian_noise(data, sigma_xyz=0.02, sigma_dop=0.01):
    """Add small Gaussian noise to XYZ and doppler."""
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        for p in pts:
            p[IDX_X] += random.gauss(0, sigma_xyz)
            p[IDX_Y] += random.gauss(0, sigma_xyz)
            p[IDX_Z] += random.gauss(0, sigma_xyz)
            p[IDX_DOPPLER] += random.gauss(0, sigma_dop)
        set_points(frame, pts)
    return d

def aug_spatial_scale(data, lo=0.88, hi=1.12):
    """Scale XYZ — person closer or farther."""
    s = random.uniform(lo, hi)
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        for p in pts:
            p[IDX_X] *= s; p[IDX_Y] *= s; p[IDX_Z] *= s
        set_points(frame, pts)
    return d

def aug_drop_points(data, rate=0.15):
    """Randomly drop ~15% of points per frame (occlusion)."""
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        if len(pts) > 3:
            pts = [p for p in pts if random.random() > rate] or pts[:2]
            set_points(frame, pts)
            frame["frameData"]["numDetectedPoints"] = len(pts)
    return d

def aug_doppler_scale(data, lo=0.88, hi=1.12):
    """Scale doppler — person moving slightly faster/slower."""
    s = random.uniform(lo, hi)
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        for p in pts: p[IDX_DOPPLER] *= s
        set_points(frame, pts)
    return d

def aug_speed_up(data):
    """Keep every 2nd frame — activity done faster."""
    if len(data) < 10: return copy.deepcopy(data)
    result = [copy.deepcopy(f) for i, f in enumerate(data) if i % 2 == 0]
    return result if len(result) >= 5 else copy.deepcopy(data)

def aug_slow_down(data):
    """Duplicate each frame with tiny noise — activity done slower."""
    result = []
    for frame in data:
        f1 = copy.deepcopy(frame)
        f2 = copy.deepcopy(frame)
        pts = get_points(f2)
        for p in pts:
            p[IDX_X] += random.gauss(0, 0.005)
            p[IDX_Y] += random.gauss(0, 0.005)
        set_points(f2, pts)
        result.extend([f1, f2])
    return result

def aug_rotate_azimuth(data, max_deg=8.0):
    """Rotate point cloud around Z — sensor at a slight angle."""
    angle = math.radians(random.uniform(-max_deg, max_deg))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    d = copy.deepcopy(data)
    for frame in d:
        pts = get_points(frame)
        for p in pts:
            x, y = p[IDX_X], p[IDX_Y]
            p[IDX_X] = x * cos_a - y * sin_a
            p[IDX_Y] = x * sin_a + y * cos_a
        set_points(frame, pts)
    return d

AUGMENTATIONS = [
    ("mirror",    aug_mirror_x),
    ("noise",     aug_gaussian_noise),
    ("scale",     aug_spatial_scale),
    ("dropp",     aug_drop_points),
    ("doppler",   aug_doppler_scale),
    ("speedup",   aug_speed_up),
    ("slowdown",  aug_slow_down),
    ("rotate",    aug_rotate_azimuth),
]


# ─────────────────────────────────────────────
# PROCESS ONE FILE
# ─────────────────────────────────────────────
def process_file(json_path: Path, label: int, out_dir: Path,
                 stem_prefix: str, manifest_rows: list, stats: dict):
    try:
        frames, template = load_frames(json_path)
    except Exception as e:
        print(f"    [ERROR] {json_path.name}: {e}")
        return

    n = len(frames)
    label_name = "FALL" if label == 1 else "NO-FALL"
    print(f"    {json_path.name}: {n} frames → {label_name}")

    def save_and_record(aug_name, aug_frames):
        fname = f"{aug_name}_{stem_prefix}_{json_path.stem}.json"
        out_path = out_dir / fname
        save_frames(out_path, template, aug_frames)
        manifest_rows.append({
            "file": str(out_path),
            "label": label,
            "label_name": label_name,
            "source": str(json_path),
            "augmentation": aug_name,
            "n_frames": len(aug_frames),
        })
        stats[label]["total_frames"] += len(aug_frames)

    # Original
    save_and_record("original", frames)
    stats[label]["files"] += 1

    # Augmentations
    for aug_name, aug_fn in AUGMENTATIONS:
        try:
            save_and_record(aug_name, aug_fn(frames))
            stats[label]["augmented"] += 1
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
        print("  → Edit DATASET_ROOT at the top of this script.")
        return

    out_fall   = out_dir / "class_1_fall"
    out_nofall = out_dir / "class_0_nofall"
    out_fall.mkdir(parents=True, exist_ok=True)
    out_nofall.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    stats = {
        0: {"files": 0, "augmented": 0, "total_frames": 0},
        1: {"files": 0, "augmented": 0, "total_frames": 0},
    }

    # ── 1. Process subfolders ──────────────────
    print("\n=== PROCESSING SUBFOLDERS ===")
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        label = get_label_for_folder(folder.name)
        if label is None:
            print(f"  [SKIP folder] '{folder.name}' — no mapping")
            continue

        label_name = "FALL" if label == 1 else "NO-FALL"
        out_class = out_fall if label == 1 else out_nofall
        json_files = sorted(folder.glob("*.json"))

        if not json_files:
            print(f"  [WARN] '{folder.name}' has no JSON files")
            continue

        print(f"\n  [{folder.name}] → {label_name}")
        for jf in json_files:
            process_file(jf, label, out_class, folder.name, manifest_rows, stats)

    # ── 2. Process loose JSON files in root ───
    print("\n=== PROCESSING LOOSE JSON FILES IN DATASET ROOT ===")
    for jf in sorted(root.glob("*.json")):
        if jf.name in FALL_LOOSE_FILES:
            label = 1
        else:
            print(f"  [SKIP file] '{jf.name}' — not in FALL_LOOSE_FILES, ignoring")
            continue
        label_name = "FALL"
        print(f"\n  [{jf.name}] → {label_name}")
        process_file(jf, label, out_fall, "loose", manifest_rows, stats)

    # ── 3. Save manifest ──────────────────────
    manifest_path = out_dir / "dataset_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "label", "label_name", "source", "augmentation", "n_frames"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # ── 4. Summary ────────────────────────────
    print("\n" + "="*55)
    print("  AUGMENTATION COMPLETE")
    print("="*55)
    print(f"  {'Class':<20} {'Orig':>6} {'Aug':>6} {'Frames':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*8}")
    for lbl, name in [(0, "NO-FALL"), (1, "FALL")]:
        s = stats[lbl]
        print(f"  {name:<20} {s['files']:>6} {s['augmented']:>6} {s['total_frames']:>8}")
    total = sum(s["files"] for s in stats.values())
    total_aug = sum(s["augmented"] for s in stats.values())
    total_frames = sum(s["total_frames"] for s in stats.values())
    print(f"  {'TOTAL':<20} {total:>6} {total_aug:>6} {total_frames:>8}")
    print(f"\n  Output:   {out_dir.resolve()}")
    print(f"  Manifest: {manifest_path.resolve()}")

    # ── 5. Class balance warning ───────────────
    f0 = stats[0]["files"]
    f1 = stats[1]["files"]
    if f0 > 0 and f1 > 0:
        ratio = max(f0, f1) / min(f0, f1)
        if ratio > 1.5:
            minority = "FALL" if f1 < f0 else "NO-FALL"
            print(f"\n  [WARNING] Class imbalance detected (ratio {ratio:.1f}x)")
            print(f"  → {minority} has fewer files. Class weights will handle this in training.")
    print()


if __name__ == "__main__":
    run()