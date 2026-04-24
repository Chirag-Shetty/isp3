"""
app.py
------
FastAPI inference server for multi-class activity detection.
Runs on Hugging Face Spaces (Docker).

Endpoints:
    GET  /health   → {"status": "ok"}
    POST /predict  → {"window": [[20 floats] × 40]}
                   ← {"class_id", "class_name", "confidence", "is_fall", "probs"}
"""

import pickle
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List

from model import MultiClassTransformerCNNLSTM

# ── Config ─────────────────────────────────────────────────────────────────────
# Multi-class model: 6 activities
CLASSES = [
    "Standing_walk",              # 0
    "Sitting_chair",              # 1
    "sitting_floor",              # 2
    "Stand_Sit_chair_transition", # 3
    "chair_floor_transition",     # 4
    "stand_floor_transition",     # 5
]

# Classes that should be flagged as "fall" events
FALL_CLASS_IDS = {5}  # stand_floor_transition can indicate a fall

NUM_FEATURES = 20
NUM_CLASSES  = len(CLASSES)

MODEL_PATH  = "./multi_class_model_best.pth"
SCALER_PATH = "./multi_class_scaler.pkl"

# ── Load model once at startup ─────────────────────────────────────────────────
print("[startup] Loading model and scaler …")
device = torch.device("cpu")

model = MultiClassTransformerCNNLSTM(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
model_loaded = False
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    # The new model saves state_dict directly (OrderedDict), not wrapped in a dict
    if isinstance(checkpoint, dict) and not any(k.startswith(('input_proj', 'transformer', 'cnn', 'lstm', 'fc')) for k in checkpoint.keys()):
        # Wrapped format
        state_dict = checkpoint.get('sd') or checkpoint.get('model_state_dict') or checkpoint
    else:
        # Direct state_dict (OrderedDict of tensors)
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model_loaded = True
    print(f"[startup] ✓ Weights loaded ({NUM_CLASSES}-class model)")
except FileNotFoundError:
    print(f"[startup] ⚠ {MODEL_PATH} not found — upload it to the Space repo")
except Exception as e:
    print(f"[startup] ⚠ Model load error: {e}")

model.eval()

scaler = None
try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("[startup] ✓ Scaler loaded")
except FileNotFoundError:
    print(f"[startup] ⚠ {SCALER_PATH} not found — features will not be scaled")

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Class Activity Detection API",
    description="Transformer-CNN-LSTM model for IWR6843 radar activity recognition (6 classes)",
    version="2.0.0",
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect base URL to interactive API docs."""
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
    """Quick liveness check — also shows whether model/scaler loaded."""
    return {
        "status":       "ok",
        "model_loaded": model_loaded,
        "scaler_loaded": scaler is not None,
        "num_classes":  NUM_CLASSES,
        "classes":      CLASSES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Run inference on a single feature window.

    Body:
        window  — list of lists, shape (N, 20).
                  N is typically 40 (WINDOW_SIZE from config.py).

    Returns:
        class_id, class_name, confidence, is_fall, probs
    """
    window = np.array(req.window, dtype=np.float32)

    # Validate shape
    if window.ndim != 2 or window.shape[1] != NUM_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"window must be shape (N, {NUM_FEATURES}), got {list(window.shape)}",
        )

    W, F = window.shape

    # Scale features (same as training)
    if scaler is not None:
        flat   = window.reshape(-1, F)
        flat   = scaler.transform(flat)
        window = flat.reshape(W, F)

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
