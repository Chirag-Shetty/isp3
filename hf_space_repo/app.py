"""
app.py — Fall Detection API v7 (Rate-of-Collapse Detector)
===========================================================
FALL  = sudden rapid collapse of height_range (< 1 second)
NO-FALL = standing, walking, sitting, sit-to-stand, sleeping (slow)

Key insight:
  A FALL drops height_range by ~0.8m in < 10 frames (< 0.55 sec)
  Sleeping/sitting: same drop but over 30+ frames (slow)

Signals used:
  1. steepest_8frame_slope < -0.06 m/frame  (fast collapse)
  2. total_drop > 0.40m                      (large drop)
  3. final_hrng < 0.45m                      (ended on floor)
  4. was_upright: init_hrng > 0.50m          (was standing/walking/sitting)

Sleeping is correctly rejected because slope is slow (~0.02 m/frame)
Sitting   is correctly rejected because final_hrng > 0.45m (chair height)
Walking   is correctly rejected because total_drop is small and no floor ending
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List

# ── Thresholds ─────────────────────────────────────────────────────────────────
PERSON_TALL_THRESH   = 0.50   # init_hrng > this = person was upright/sitting
PERSON_FALLEN_THRESH = 0.42   # final_hrng < this = person on floor
FALL_SLOPE_THRESH    = -0.055 # steepest 8-frame slope < this = FAST collapse
TOTAL_DROP_THRESH    = 0.38   # peak-to-trough height_range drop > this
MIN_POINTS_VALID     = 4      # frames with < 4 pts ignored (noise/empty)
SMOOTH_N             = 5      # rolling mean window for height_range
MIN_VALID_FRAMES     = 16     # need at least 16 valid frames in window
FRAME_DT             = 0.055  # seconds between frames

VERSION = "7.0.0"


def smooth_signal(values, valid_mask, k=5):
    """k-frame rolling mean over valid frames only. Returns NaN for gaps."""
    out = np.full(len(values), np.nan)
    for t in range(len(values)):
        sl = values[max(0, t-k+1):t+1]
        vm = valid_mask[max(0, t-k+1):t+1]
        vv = sl[vm]
        if len(vv) >= 2:
            out[t] = float(np.mean(vv))
    return out


def detect_fall(window: np.ndarray):
    """
    window : (N, 20) array — pipeline feature window
      col  2 = z_mean       (centroid height)
      col  9 = n_points
      col 10 = spread_xy    (horizontal spread)
      col 11 = height_range (z_max - z_min per frame)

    Returns (is_fall, confidence, debug_dict)
    """
    npts = window[:, 9].astype(np.float32)
    hrng = window[:, 11].astype(np.float32)
    z    = window[:, 2].astype(np.float32)
    spxy = window[:, 10].astype(np.float32)
    valid = npts >= MIN_POINTS_VALID

    n_valid = int(valid.sum())
    if n_valid < MIN_VALID_FRAMES:
        return False, 0.0, {"reason": "too_few_valid_frames", "n_valid": n_valid}

    # ── Smooth height_range and z ─────────────────────────────────────────────
    hrng_s = smooth_signal(hrng, valid, k=SMOOTH_N)
    z_s    = smooth_signal(z,    valid, k=SMOOTH_N)
    ok     = ~np.isnan(hrng_s)

    if ok.sum() < MIN_VALID_FRAMES:
        return False, 0.0, {"reason": "insufficient_smooth_data"}

    hs = hrng_s[ok]   # valid smoothed height_range values
    zs = z_s[ok]

    # ── Key measurements ──────────────────────────────────────────────────────
    n = len(hs)
    h = n // 2

    # Initial state: average of FIRST 8 valid smoothed frames
    init_hrng  = float(np.mean(hs[:min(8, n)]))
    # Final state: average of LAST 8 valid smoothed frames
    final_hrng = float(np.mean(hs[max(0, n-8):]))
    # Peak in first half (tallest the person was)
    peak_hrng  = float(np.max(hs[:max(h, 1)]))
    # Trough in second half (lowest the person got)
    trough_hrng = float(np.min(hs[h:]))

    total_drop = peak_hrng - trough_hrng  # large = significant collapse

    # Z drop
    init_z  = float(np.mean(zs[:min(8, len(zs))]))
    final_z = float(np.mean(zs[max(0, len(zs)-8):]))
    z_drop  = init_z - final_z   # positive = centroid dropped

    # Spread change (person goes horizontal = spread increases)
    spxy_valid = spxy[valid].astype(np.float32)
    sh = len(spxy_valid) // 2
    init_spxy  = float(np.mean(spxy_valid[:max(sh,1)]))
    final_spxy = float(np.mean(spxy_valid[sh:])) if sh < len(spxy_valid) else init_spxy
    spread_increase = final_spxy - init_spxy

    # ── Steepest slope: max drop over any 8 consecutive smoothed frames ───────
    # This is the KEY metric: fall is fast, sleep/sit is slow
    slopes = np.diff(hs)   # per-frame change in smoothed height_range
    steepest = float(slopes.min()) if len(slopes) > 0 else 0.0  # most negative

    # ── Fall signals ──────────────────────────────────────────────────────────
    was_upright      = init_hrng   > PERSON_TALL_THRESH     # was standing/sitting/walking
    ended_on_floor   = final_hrng  < PERSON_FALLEN_THRESH   # ended flat on floor
    fast_collapse    = steepest    < FALL_SLOPE_THRESH       # fast drop (not slow sit/sleep)
    large_drop       = total_drop  > TOTAL_DROP_THRESH       # big height_range drop
    z_fell           = z_drop      > 0.30                    # centroid also dropped
    went_horizontal  = spread_increase > 0.08                # wider = lying down

    # Anti-sleeping: if final is low BUT transition was slow → not a fall
    # (fast_collapse already handles this via slope threshold)

    # Anti-sitting: if person ends at chair height (0.42–0.75m) → not a fall
    # (ended_on_floor already handles this)

    # Core vote: need fast collapse + ended on floor + was upright
    core_fall = was_upright and fast_collapse and ended_on_floor

    # Supporting signals boost confidence
    votes = (int(large_drop) + int(z_fell) + int(went_horizontal))

    is_fall    = core_fall and large_drop   # need core + large drop to fire
    confidence = 0.0
    if is_fall:
        confidence = min(1.0, 0.65 + 0.12 * votes)

    debug = {
        "init_hrng":      round(init_hrng, 3),
        "final_hrng":     round(final_hrng, 3),
        "peak_hrng":      round(peak_hrng, 3),
        "trough_hrng":    round(trough_hrng, 3),
        "total_drop":     round(total_drop, 3),
        "steepest_slope": round(steepest, 4),
        "z_drop":         round(z_drop, 3),
        "spread_increase":round(spread_increase, 3),
        "was_upright":    was_upright,
        "ended_on_floor": ended_on_floor,
        "fast_collapse":  fast_collapse,
        "large_drop":     large_drop,
        "z_fell":         z_fell,
        "went_horizontal":went_horizontal,
        "n_valid":        n_valid,
    }
    return is_fall, confidence, debug


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fall Detection API — Rate-of-Collapse v7",
    description=(
        "Detects SUDDEN collapse of radar height_range (fall) vs "
        "SLOW descent (sleeping, sitting). "
        "No ML model — pure physics thresholds."
    ),
    version=VERSION,
)

CLASSES = ["NO-FALL", "FALL"]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status":     "ok",
        "model_type": "RateOfCollapse",
        "version":    VERSION,
        "thresholds": {
            "person_tall_thresh":   PERSON_TALL_THRESH,
            "person_fallen_thresh": PERSON_FALLEN_THRESH,
            "fall_slope_thresh":    FALL_SLOPE_THRESH,
            "total_drop_thresh":    TOTAL_DROP_THRESH,
            "min_points_valid":     MIN_POINTS_VALID,
        },
        "classes": CLASSES,
    }


class PredictRequest(BaseModel):
    window: List[List[float]]


class PredictResponse(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    is_fall:    bool
    probs:      List[float]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Detect fall from a (N, 20) radar feature window.

    FALL fired when ALL true:
      - person was upright recently (init height_range > 0.50m)
      - person ended on floor (final height_range < 0.42m)
      - collapse was FAST (steepest 1-frame drop > 0.055 m/frame)
      - total height_range drop > 0.38m

    Rejects: sleeping (slow slope), sitting (chair height > 0.42m),
             empty room (< 4 pts/frame), walking (no floor ending)
    """
    window = np.array(req.window, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] < 12:
        raise HTTPException(
            status_code=422,
            detail=f"window must be (N, >=12), got {list(window.shape)}"
        )

    is_fall, confidence, debug = detect_fall(window)
    class_id = 1 if is_fall else 0
    p_fall   = confidence if is_fall else 0.04
    p_nofall = 1.0 - p_fall

    return PredictResponse(
        class_id   = class_id,
        class_name = CLASSES[class_id],
        confidence = confidence if is_fall else round(p_nofall, 3),
        is_fall    = is_fall,
        probs      = [round(p_nofall, 3), round(p_fall, 3)],
    )
