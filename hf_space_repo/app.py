"""
app.py — Fall Detection API (Window-based Physics Detector v6)
==============================================================
Detects the TRANSITION from upright to flat within a 40-frame window.
Compares first-half vs second-half of the window.

Key rule:
  FALL = first_half_hrng > 0.60m  (was upright)
       AND second_half_hrng < 0.45m (now flat)
       AND hrng_collapsed > 35%    (significant drop)
       AND NOT entire_flat          (not already on floor)
       AND person_present           (enough radar points)

This prevents:
  - Sustained-flat repeating (person already on floor)
  - Empty-frame triggers (no person detected)
  - Walking sparse triggers (not sustained flat in 2nd half)
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List

# ── Thresholds ─────────────────────────────────────────────────────────────────
PERSON_TALL_THRESH   = 0.60   # first-half avg height_range > this = was upright
PERSON_FALLEN_THRESH = 0.45   # second-half avg height_range < this = now flat
COLLAPSE_RATIO       = 0.65   # second/first height_range ratio < this = collapsed
Z_DROP_THRESH        = 0.35   # z_mean drop first→second half confirms fall
MIN_POINTS_VALID     = 5      # frames with < 5 pts ignored (noise/empty)
MIN_VALID_FRAMES     = 8      # each half needs at least this many valid frames

VERSION = "6.0.0"


def detect_fall_in_window(window: np.ndarray):
    """
    window: (N, 20) numpy array — N frames of pipeline features
      col 2  = z_mean
      col 9  = n_points
      col 11 = height_range (z_max - z_min per frame)

    Returns: (is_fall: bool, confidence: float, debug: dict)

    Logic: compare first half vs second half of the window.
    FALL = person was tall in first half AND flat in second half.
    Anti-repeat: if entire window is already flat -> NOT a new fall.
    """
    n    = len(window)
    half = n // 2

    npts = window[:, 9].astype(np.float32)
    hrng = window[:, 11].astype(np.float32)
    z    = window[:, 2].astype(np.float32)

    valid = npts >= MIN_POINTS_VALID

    f_valid = valid[:half]
    s_valid = valid[half:]

    f_count = int(f_valid.sum())
    s_count = int(s_valid.sum())

    debug = {"f_valid": f_count, "s_valid": s_count}

    # Not enough valid frames in either half → can't decide
    if f_count < MIN_VALID_FRAMES or s_count < MIN_VALID_FRAMES:
        debug["reason"] = "insufficient_valid_frames"
        return False, 0.0, debug

    first_hrng  = float(np.mean(hrng[:half][f_valid]))
    second_hrng = float(np.mean(hrng[half:][s_valid]))
    first_z     = float(np.mean(z[:half][f_valid]))
    second_z    = float(np.mean(z[half:][s_valid]))

    hrng_ratio  = second_hrng / (first_hrng + 1e-8)
    z_drop      = first_z - second_z   # positive = dropped

    # ── Four conditions for FALL ──────────────────────────────────────────────
    was_tall      = first_hrng  > PERSON_TALL_THRESH    # person upright in first half
    is_now_flat   = second_hrng < PERSON_FALLEN_THRESH  # person flat in second half
    hrng_collapsed = hrng_ratio < COLLAPSE_RATIO         # 35%+ collapse
    z_dropped     = z_drop      > Z_DROP_THRESH          # centroid dropped

    # Anti-repeat: if entire window has low hrng → person already on floor → skip
    entire_flat = first_hrng < PERSON_FALLEN_THRESH * 1.5   # both halves flat

    is_fall = (was_tall and is_now_flat and hrng_collapsed and not entire_flat)

    # Confidence
    if is_fall:
        confidence = 0.75 + (0.25 if z_dropped else 0.0)
    else:
        confidence = 0.0

    debug.update({
        "first_hrng":    round(first_hrng, 3),
        "second_hrng":   round(second_hrng, 3),
        "hrng_ratio":    round(hrng_ratio, 3),
        "z_drop":        round(z_drop, 3),
        "was_tall":      was_tall,
        "is_now_flat":   is_now_flat,
        "hrng_collapsed": hrng_collapsed,
        "entire_flat":   entire_flat,
        "z_dropped":     z_dropped,
    })
    return is_fall, confidence, debug


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fall Detection API (Window-based Physics)",
    description=(
        "Detects fall TRANSITION: person was upright in first half of window, "
        "flat in second half. Prevents repeat-FALL from sustained fallen state."
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
        "status":      "ok",
        "model_type":  "WindowPhysics",
        "version":     VERSION,
        "thresholds": {
            "person_tall_thresh":   PERSON_TALL_THRESH,
            "person_fallen_thresh": PERSON_FALLEN_THRESH,
            "collapse_ratio":       COLLAPSE_RATIO,
            "z_drop_thresh":        Z_DROP_THRESH,
            "min_points_valid":     MIN_POINTS_VALID,
            "min_valid_frames":     MIN_VALID_FRAMES,
        },
        "num_classes": 2,
        "classes":     CLASSES,
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
    Detect fall in a feature window.

    Body:
        window — shape (N, 20), col2=z_mean, col9=n_points, col11=height_range

    Returns FALL only if:
      - First half of window has high height_range (person was upright)
      - Second half has low height_range (person is flat)
      - NOT already flat in first half (anti-repeat)
      - Enough valid radar points in both halves
    """
    window = np.array(req.window, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] < 12:
        raise HTTPException(
            status_code=422,
            detail=f"window must be shape (N, >=12), got {list(window.shape)}"
        )

    is_fall, confidence, debug = detect_fall_in_window(window)

    class_id = 1 if is_fall else 0
    p_fall   = float(confidence) if is_fall else 0.05
    p_nofall = 1.0 - p_fall

    return PredictResponse(
        class_id   = class_id,
        class_name = CLASSES[class_id],
        confidence = confidence if is_fall else (1.0 - p_fall),
        is_fall    = is_fall,
        probs      = [p_nofall, p_fall],
    )
