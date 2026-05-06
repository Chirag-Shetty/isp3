"""
app.py
------
FastAPI inference server — Binary Fall Detection via Random Forest.
Runs on Hugging Face Spaces (Docker).

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> {"window": [[20 floats] x 40]}
                   <- {"class_id", "class_name", "confidence", "is_fall", "probs"}
"""

import pickle
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List

# ── Config ────────────────────────────────────────────────────────────────────
CLASSES        = ["NO-FALL", "FALL"]
FALL_CLASS_IDS = {1}
NUM_FEATURES   = 20
WINDOW_SIZE    = 40
MODEL_PATH     = "./fall_rf_model.pkl"

# ── Load model at startup ─────────────────────────────────────────────────────
print("[startup] Loading Random Forest fall detection model ...")
model_payload = None
model_loaded  = False
try:
    with open(MODEL_PATH, "rb") as f:
        model_payload = pickle.load(f)
    model_loaded = True
    print(f"[startup] Model loaded: {model_payload['model_name']}"
          f"  CV ROC-AUC={model_payload['cv_roc_auc']:.4f}")
except FileNotFoundError:
    print(f"[startup] WARNING: {MODEL_PATH} not found")
except Exception as e:
    print(f"[startup] WARNING: load error: {e}")


# ── Window feature extraction (IDENTICAL to rpi_pipeline/feature_extract.py) ─
SNR_THRESHOLD = 10.0
FRAME_DT      = 0.055

def window_to_features(window: np.ndarray) -> np.ndarray:
    """
    Compress (40, 20) window -> 124-dim feature vector for RF.
    Must match train_rf_fall.py exactly.
    """
    T, F = window.shape
    xs   = np.arange(T, dtype=np.float32)
    feats = []
    for col in range(F):
        v = window[:, col].astype(np.float32)
        feats += [v.mean(), v.std(), v.min(), v.max(), v.max()-v.min()]
        slope = float(np.polyfit(xs, v, 1)[0]) if v.std() > 1e-8 else 0.0
        feats.append(slope)

    # Physics extras
    feats.append(float(np.polyfit(xs, window[:, 2],  1)[0]))  # z_slope
    feats.append(float(np.polyfit(xs, window[:, 11], 1)[0]))  # height_slope
    feats.append(float(np.max(np.abs(window[:, 5]))))          # peak_vz
    feats.append(float(np.max(np.abs(window[:, 8]))))          # peak_az

    return np.array(feats, dtype=np.float32)


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Binary Fall Detection API (Random Forest)",
    description="RF model for IWR6843 radar — NO-FALL vs FALL (2 classes)",
    version="4.0.0",
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


class PredictRequest(BaseModel):
    window: List[List[float]]   # shape (40, 20)


class PredictResponse(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    is_fall:    bool
    probs:      List[float]


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_loaded":  model_loaded,
        "model_type":    model_payload["model_name"] if model_payload else "none",
        "cv_roc_auc":    model_payload["cv_roc_auc"] if model_payload else 0,
        "num_classes":   2,
        "classes":       CLASSES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Run binary fall detection on a single feature window.

    Body:
        window -- list of lists, shape (40, 20)

    Returns:
        class_id   : 0=NO-FALL, 1=FALL
        class_name : "NO-FALL" or "FALL"
        confidence : probability of predicted class
        is_fall    : True if FALL
        probs      : [P(NO-FALL), P(FALL)]
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    window = np.array(req.window, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] != NUM_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"window must be shape (N, {NUM_FEATURES}), got {list(window.shape)}"
        )

    # Summarise window -> 124-dim feature vector
    feat_vec = window_to_features(window).reshape(1, -1)

    # RF predict
    clf   = model_payload["model"]
    probs = clf.predict_proba(feat_vec)[0]   # [P(NO-FALL), P(FALL)]
    class_id = int(np.argmax(probs))

    return PredictResponse(
        class_id   = class_id,
        class_name = CLASSES[class_id],
        confidence = float(probs[class_id]),
        is_fall    = class_id in FALL_CLASS_IDS,
        probs      = probs.tolist(),
    )
