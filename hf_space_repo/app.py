"""
app.py
------
FastAPI inference server for BINARY fall detection.
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

from model import FallDetectionTransformerCNNLSTM

# -- Config -------------------------------------------------------------------
CLASSES = [
    "NO-FALL",   # 0
    "FALL",      # 1
]

FALL_CLASS_IDS = {1}   # class 1 is always FALL

NUM_FEATURES = 20
NUM_CLASSES  = 2

MODEL_PATH  = "./fall_detection_model_best.pth"
SCALER_PATH = "./fall_scaler.pkl"

# -- Load model once at startup -----------------------------------------------
print("[startup] Loading binary fall detection model ...")
device = torch.device("cpu")

model = FallDetectionTransformerCNNLSTM(input_size=NUM_FEATURES, num_classes=NUM_CLASSES)
model_loaded = False
try:
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model_loaded = True
    print("[startup] Model loaded OK (binary: NO-FALL / FALL)")
except FileNotFoundError:
    print(f"[startup] WARNING: {MODEL_PATH} not found - upload it to the Space repo")
except Exception as e:
    print(f"[startup] WARNING: Model load error: {e}")

model.eval()

scaler = None
try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("[startup] Scaler loaded OK")
except FileNotFoundError:
    print(f"[startup] WARNING: {SCALER_PATH} not found - features will not be scaled")

# -- FastAPI ------------------------------------------------------------------
app = FastAPI(
    title="Binary Fall Detection API",
    description="Transformer-CNN-LSTM model for IWR6843 radar — NO-FALL vs FALL (2 classes)",
    version="3.0.0",
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


class PredictRequest(BaseModel):
    window: List[List[float]]   # shape (WINDOW_SIZE, 20)


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
        "scaler_loaded": scaler is not None,
        "num_classes":   NUM_CLASSES,
        "classes":       CLASSES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Run binary fall detection on a single feature window.

    Body:
        window  -- list of lists, shape (40, 20)

    Returns:
        class_id   : 0 = NO-FALL, 1 = FALL
        class_name : "NO-FALL" or "FALL"
        confidence : probability of the predicted class
        is_fall    : True if class_id == 1
        probs      : [P(NO-FALL), P(FALL)]
    """
    window = np.array(req.window, dtype=np.float32)

    if window.ndim != 2 or window.shape[1] != NUM_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"window must be shape (N, {NUM_FEATURES}), got {list(window.shape)}",
        )

    W, F = window.shape

    # Scale features (same StandardScaler as training)
    if scaler is not None:
        window = scaler.transform(window.reshape(-1, F)).reshape(W, F)

    # Inference
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, W, F)
    with torch.no_grad():
        probs = model.predict_proba(x)[0].cpu().numpy()

    class_id = int(np.argmax(probs))

    return PredictResponse(
        class_id   = class_id,
        class_name = CLASSES[class_id],
        confidence = float(probs[class_id]),
        is_fall    = class_id in FALL_CLASS_IDS,
        probs      = probs.tolist(),
    )
