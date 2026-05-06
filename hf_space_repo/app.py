"""
app.py — Fall Detection API v8 (Z-Drop Primary)
================================================
z_mean (centroid height) drops from ~3m to ~0.7m during a fall.
This is the PRIMARY signal. height_range is secondary.

FALL  = z_mean drops > 1.5m AND ends at low position AND was fast
NO-FALL examples:
  sitting:     z drops ~1m, ends at chair height (> 1.2m)  → rejected by final_z
  sleeping:    z drops > 1.5m, ends low BUT slope is slow  → rejected by speed
  walking:     z stable, no large drop                      → rejected by magnitude
  empty room:  < 4 pts/frame                                → rejected by presence
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List

# ── Thresholds ─────────────────────────────────────────────────────────────────
Z_DROP_THRESH        = 1.5    # total z_mean drop (init - final) > this = fall candidate
Z_DROP_FAST_THRESH   = 0.8    # fast z_drop (init - final) if slope also fast
FINAL_Z_THRESH       = 1.5    # final z_mean must be < this (person on floor, not chair)
INIT_Z_THRESH        = 1.5    # init z_mean must be > this (was upright/present)
Z_SLOPE_THRESH       = -0.05  # smoothed z slope < this m/frame = fast fall
MIN_POINTS_VALID     = 4      # frames with < 4 pts ignored
SMOOTH_N             = 5      # rolling mean window
MIN_VALID_FRAMES     = 16     # minimum valid frames needed
FRAME_DT             = 0.055  # seconds per frame

VERSION = "8.0.0"


def smooth_signal(values, valid_mask, k=5):
    """k-frame rolling mean over valid frames only."""
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
    window : (N, 20) — pipeline feature window
      col  2 = z_mean       (centroid height — PRIMARY signal)
      col  9 = n_points
      col 10 = spread_xy
      col 11 = height_range (z_max - z_min — secondary signal)
    """
    npts = window[:, 9].astype(np.float32)
    z    = window[:, 2].astype(np.float32)
    hrng = window[:, 11].astype(np.float32)
    spxy = window[:, 10].astype(np.float32)
    valid = npts >= MIN_POINTS_VALID

    n_valid = int(valid.sum())
    if n_valid < MIN_VALID_FRAMES:
        return False, 0.0, {"reason": "too_few_valid_frames", "n_valid": n_valid}

    # ── Smooth z_mean and height_range ────────────────────────────────────────
    z_s    = smooth_signal(z,    valid, k=SMOOTH_N)
    hrng_s = smooth_signal(hrng, valid, k=SMOOTH_N)
    ok     = ~np.isnan(z_s)

    if ok.sum() < MIN_VALID_FRAMES:
        return False, 0.0, {"reason": "insufficient_smooth_data"}

    zs = z_s[ok]
    hs = hrng_s[ok]

    n = len(zs)

    # ── Z measurements ────────────────────────────────────────────────────────
    init_z  = float(np.mean(zs[:min(10, n)]))   # first 10 frames avg
    final_z = float(np.mean(zs[max(0, n-10):]))  # last 10 frames avg
    peak_z  = float(np.max(zs[:max(n//2, 1)]))   # highest z in first half
    trough_z = float(np.min(zs[n//2:]))           # lowest z in second half

    total_z_drop = peak_z - trough_z    # peak in 1st half → trough in 2nd half
    init_final_drop = init_z - final_z  # simple first→last comparison

    # Steepest z slope (most negative = fastest fall)
    z_slopes = np.diff(zs)
    steepest_z = float(z_slopes.min()) if len(z_slopes) > 0 else 0.0

    # ── Height-range measurements (secondary) ─────────────────────────────────
    init_hrng  = float(np.mean(hs[:min(10, n)]))
    final_hrng = float(np.mean(hs[max(0, n-10):]))
    hrng_drop  = init_hrng - final_hrng

    # ── Spread (horizontal extent grows when lying down) ─────────────────────
    spxy_valid = spxy[valid]
    sh = len(spxy_valid) // 2
    init_spxy  = float(np.mean(spxy_valid[:max(sh, 1)]))
    final_spxy = float(np.mean(spxy_valid[sh:])) if sh < len(spxy_valid) else init_spxy
    spread_delta = final_spxy - init_spxy

    # ── Fall signals ──────────────────────────────────────────────────────────
    #  Primary (z_mean based)
    was_high        = init_z   > INIT_Z_THRESH           # person was upright (z high)
    ended_low       = final_z  < FINAL_Z_THRESH          # person ended at floor level
    big_drop        = total_z_drop > Z_DROP_THRESH        # large z drop (> 1.5m)
    moderate_drop   = total_z_drop > Z_DROP_FAST_THRESH   # moderate drop (> 0.8m) if fast
    fast_fall       = steepest_z   < Z_SLOPE_THRESH       # steep slope = fast

    #  Secondary (height_range, spread)
    hrng_collapsed  = hrng_drop    > 0.20                 # height_range also collapsed
    went_horizontal = spread_delta > 0.05                 # wider footprint

    # ── Decision ─────────────────────────────────────────────────────────────
    # CASE 1: Large slow/fast drop → definitely a fall
    case1 = was_high and ended_low and big_drop

    # CASE 2: Moderate drop but very fast → also a fall
    case2 = was_high and ended_low and moderate_drop and fast_fall

    is_fall    = case1 or case2
    confidence = 0.0
    if is_fall:
        support = int(hrng_collapsed) + int(went_horizontal) + int(fast_fall)
        confidence = min(1.0, 0.70 + 0.10 * support)

    debug = {
        "init_z":        round(init_z, 3),
        "final_z":       round(final_z, 3),
        "peak_z":        round(peak_z, 3),
        "trough_z":      round(trough_z, 3),
        "total_z_drop":  round(total_z_drop, 3),
        "init_final_drop": round(init_final_drop, 3),
        "steepest_z_slope": round(steepest_z, 4),
        "init_hrng":     round(init_hrng, 3),
        "final_hrng":    round(final_hrng, 3),
        "hrng_drop":     round(hrng_drop, 3),
        "spread_delta":  round(spread_delta, 3),
        "was_high":      was_high,
        "ended_low":     ended_low,
        "big_drop":      big_drop,
        "fast_fall":     fast_fall,
        "case1":         case1,
        "case2":         case2,
        "n_valid":       n_valid,
    }
    return is_fall, confidence, debug


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fall Detection API — Z-Drop v8",
    description=(
        "PRIMARY signal: z_mean (centroid height) drops > 1.5m. "
        "Rejects sitting (ends at chair height > 1.5m), "
        "sleeping (slow slope), empty room (< 4 pts/frame)."
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
        "model_type": "ZDropPrimary",
        "version":    VERSION,
        "thresholds": {
            "z_drop_thresh":      Z_DROP_THRESH,
            "z_drop_fast_thresh": Z_DROP_FAST_THRESH,
            "final_z_thresh":     FINAL_Z_THRESH,
            "init_z_thresh":      INIT_Z_THRESH,
            "z_slope_thresh":     Z_SLOPE_THRESH,
            "min_points_valid":   MIN_POINTS_VALID,
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
    Detect fall from (N, 20) radar feature window.

    FALL when:
      z_mean drops > 1.5m (peak→trough) AND ends at z < 1.5m
      OR drops > 0.8m AND fast slope < -0.05 m/frame AND ends low

    Rejects:
      sitting   — ends at chair height (z > 1.5m)
      sleeping  — slow slope
      walking   — z stays high, no large drop
      empty room — < 4 pts/frame
    """
    window = np.array(req.window, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] < 12:
        raise HTTPException(status_code=422,
            detail=f"window must be (N, >=12), got {list(window.shape)}")

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
