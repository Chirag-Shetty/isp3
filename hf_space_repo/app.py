"""
app.py — Fall Detection API (Physics-based Threshold Detector)
==============================================================
No ML model file needed. Pure thresholds based on mmWave physics.

FALL when (sustained for 4+ frames):
  height_range < 0.35m  (person is flat/horizontal)
  AND was_tall (height_range > 0.7m) recently

Endpoints:
  GET  /health   -> status
  POST /predict  -> {"window": [[20 floats] x N]}
"""

import numpy as np
from collections import deque
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List

# ── Thresholds (from mmWave physics + ESPHome IWR6843 production) ─────────────
PERSON_TALL_THRESH   = 0.60   # height_range > this = person was upright
PERSON_FALLEN_THRESH = 0.50   # height_range < this = person is flat/fallen
N_POINTS_LOW         = 10     # n_points < this = sparse (floor reflection)
Z_DROP_THRESH        = 0.40   # z_mean must drop this much from recent peak
HISTORY_FRAMES       = 25
SMOOTH_N             = 5
MIN_POINTS_VALID     = 2      # ignore frames with < 2 points (noise)
PERSIST_FRAMES       = 2      # flat height_range must persist N frames


# ── Embedded detector (no pkl needed) ─────────────────────────────────────────
class FallThresholdDetector:
    """
    Physics-based streaming fall detector.
    Feed window frames one at a time via update_from_features().

    Window column mapping (from rpi_pipeline/feature_extract.py):
      col 2  = z_mean        (centroid height)
      col 9  = n_points
      col 11 = height_range  (z_max - z_min of point cloud)
    """

    def __init__(self):
        self._hrng_buf    = deque(maxlen=HISTORY_FRAMES)
        self._z_buf       = deque(maxlen=HISTORY_FRAMES + SMOOTH_N)
        self._npts_buf    = deque(maxlen=HISTORY_FRAMES)
        self._flat_streak = 0
        self._cooldown    = 0

    def reset(self):
        self._hrng_buf.clear()
        self._z_buf.clear()
        self._npts_buf.clear()
        self._flat_streak = 0
        self._cooldown    = 0

    def update(self, z_mean: float, height_range: float, n_points: int):
        if self._cooldown > 0:
            self._cooldown -= 1

        # Only buffer valid frames (enough points to be reliable)
        if n_points >= MIN_POINTS_VALID:
            self._hrng_buf.append(height_range)
            self._z_buf.append(z_mean)
            self._npts_buf.append(n_points)

        if len(self._hrng_buf) < max(SMOOTH_N, 5):
            return False, 0.0

        # Smoothed z and peak
        recent_z  = list(self._z_buf)
        z_smooth  = float(np.mean(recent_z[-SMOOTH_N:]))
        z_peak    = float(max(recent_z[-HISTORY_FRAMES:]))
        z_drop    = z_peak - z_smooth

        hrng_hist = list(self._hrng_buf)
        was_tall  = max(hrng_hist[-min(HISTORY_FRAMES, len(hrng_hist)):]) > PERSON_TALL_THRESH
        is_flat   = (height_range < PERSON_FALLEN_THRESH
                     and n_points >= MIN_POINTS_VALID)
        z_dropped = z_drop > Z_DROP_THRESH
        few_pts   = MIN_POINTS_VALID <= n_points < N_POINTS_LOW

        # Persistence — must be flat for PERSIST_FRAMES consecutive frames
        if is_flat:
            self._flat_streak += 1
        else:
            self._flat_streak = 0
        sustained_flat = self._flat_streak >= PERSIST_FRAMES

        primary   = was_tall and sustained_flat
        secondary = was_tall and z_dropped and few_pts
        tertiary  = was_tall and z_dropped and sustained_flat  # z drop + flat

        is_fall   = (primary or secondary or tertiary) and self._cooldown == 0

        if is_fall:
            self._cooldown = 40

        votes = int(primary) + int(secondary)
        confidence = min(1.0, votes / 2.0)
        return is_fall, confidence

    def predict_window(self, window: np.ndarray):
        """
        window: (N, 20) — N frames of pipeline features.
        Runs frame-by-frame. Returns final (is_fall, confidence).
        """
        self.reset()
        is_fall, conf = False, 0.0
        for i in range(len(window)):
            f, c = self.update(
                z_mean       = float(window[i, 2]),
                height_range = float(window[i, 11]),
                n_points     = int(window[i, 9]),
            )
            if f:
                is_fall, conf = True, c
        return is_fall, conf


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fall Detection API (Physics-based Threshold)",
    description=(
        "IWR6843 radar fall detector using height_range collapse threshold. "
        "No ML model — pure physics. Falls when height_range < 0.35m sustained."
    ),
    version="5.0.0",
)

CLASSES = ["NO-FALL", "FALL"]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_type":   "PhysicsThreshold",
        "version":      "5.0.0",
        "thresholds": {
            "person_tall_thresh":   PERSON_TALL_THRESH,
            "person_fallen_thresh": PERSON_FALLEN_THRESH,
            "n_points_low":         N_POINTS_LOW,
            "z_drop_thresh":        Z_DROP_THRESH,
            "persist_frames":       PERSIST_FRAMES,
        },
        "num_classes":  2,
        "classes":      CLASSES,
    }


class PredictRequest(BaseModel):
    window: List[List[float]]   # shape (N, 20)


class PredictResponse(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    is_fall:    bool
    probs:      List[float]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Run physics-based fall detection on a feature window.

    Body:
        window — list of lists, shape (N, 20)
                 col 2  = z_mean
                 col 9  = n_points
                 col 11 = height_range (z_max - z_min per frame)

    Returns:
        class_id   : 0=NO-FALL, 1=FALL
        class_name : "NO-FALL" or "FALL"
        confidence : 0.0 – 1.0
        is_fall    : bool
        probs      : [P(NO-FALL), P(FALL)]
    """
    window = np.array(req.window, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] < 12:
        raise HTTPException(
            status_code=422,
            detail=f"window must be shape (N, 20), got {list(window.shape)}"
        )

    det = FallThresholdDetector()
    is_fall, confidence = det.predict_window(window)

    class_id = 1 if is_fall else 0
    p_fall   = float(confidence) if is_fall else 0.1
    p_nofall = 1.0 - p_fall

    return PredictResponse(
        class_id   = class_id,
        class_name = CLASSES[class_id],
        confidence = float(confidence) if is_fall else (1.0 - confidence),
        is_fall    = is_fall,
        probs      = [p_nofall, p_fall],
    )
