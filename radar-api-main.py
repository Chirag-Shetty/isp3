from boto3.dynamodb.conditions import Key
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from collections import deque

import numpy as np
import boto3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
TABLE_NAME = os.getenv("RADAR_EVENTS_TABLE", "radar_events")

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)


class FrameRequest(BaseModel):
    device_id: str
    timestamp: str | None = None
    features: list[float]


class WSManager:
    def __init__(self):
        self.clients = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, payload: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# ── Rule-based fall detector ─────────────────────────────────────────────────
Z_DROP_THRESHOLD = 0.50
BODY_FLAT_THRESH = 0.50
LOW_POINTS_THRESH = 8
DETECTION_WINDOW = 20
MIN_VOTES = 2


class StreamingFallDetector:
    def __init__(self):
        self.z_buf = deque(maxlen=DETECTION_WINDOW + 5)
        self.cooldown = 0

    def update(self, feat_20: np.ndarray):
        z    = float(feat_20[2])
        npts = float(feat_20[9])
        hrng = float(feat_20[11])

        self.z_buf.append(z)

        if self.cooldown > 0:
            self.cooldown -= 1

        if len(self.z_buf) < DETECTION_WINDOW:
            return False, 0.0, {"status": "warming_up", "buffered": len(self.z_buf)}

        window_z  = list(self.z_buf)[-DETECTION_WINDOW:]
        z_peak    = max(window_z)
        z_current = window_z[-1]
        z_drop    = z_peak - z_current

        height_dropped = z_drop >= Z_DROP_THRESHOLD
        body_flat      = hrng  <= BODY_FLAT_THRESH
        few_points     = npts  <= LOW_POINTS_THRESH

        votes   = int(height_dropped) + int(body_flat) + int(few_points)
        is_fall = (votes >= MIN_VOTES) and (self.cooldown == 0)

        if is_fall:
            self.cooldown = 30

        confidence = min(1.0, votes / 3.0)
        info = {
            "z_drop":    round(z_drop, 3),
            "z_peak":    round(z_peak, 3),
            "z_current": round(z_current, 3),
            "hrng":      round(hrng, 3),
            "npts":      int(npts),
            "votes":     votes,
        }
        return is_fall, confidence, info


detectors  = {}
ws_manager = WSManager()
app        = FastAPI(title="Radar Rule-Based API", version="1.0.0")

# ── CORS — must be added before routes ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def ddb_num(x, digits=4):
    return Decimal(str(round(float(x), digits)))


def _json_safe(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "table": TABLE_NAME, "region": AWS_REGION}


@app.post("/frame")
async def frame(req: FrameRequest):
    if len(req.features) != 20:
        raise HTTPException(status_code=422, detail="features must be length 20")

    feat      = np.array(req.features, dtype=np.float32)
    device_id = req.device_id
    ts        = req.timestamp or now_iso()

    det = detectors.get(device_id)
    if det is None:
        det = StreamingFallDetector()
        detectors[device_id] = det

    is_fall, conf, info = det.update(feat)

    p_fall   = conf if is_fall else 0.05
    p_nofall = 1.0 - p_fall

    event = {
        "device_id":   device_id,
        "ts":          ts,
        "timestamp":   ts,           # also include 'timestamp' for frontend compatibility
        "class_id":    1 if is_fall else 0,
        "class_name":  "FALL" if is_fall else "NO-FALL",
        "confidence":  float(conf if is_fall else p_nofall),
        "is_fall":     bool(is_fall),
        "probs":       [round(p_nofall, 3), round(p_fall, 3)],
        "z_mean":      float(feat[2]),
        "height_range": float(feat[11]),
        "n_points":    int(feat[9]),
        "x_mean":      float(feat[0]),
        "y_mean":      float(feat[1]),
        "frame_count": 0,            # not tracked per-frame in streaming mode
        "debug":       info,
    }

    # Broadcast every event over WebSocket (not just falls)
    await ws_manager.broadcast(event)

    # Save ALL events to DynamoDB (not just falls) so /history works
    ttl = int(time.time()) + 86400  # 24h TTL
    item = {
        "device_id":   device_id,
        "ts":          ts,
        "pk":          "all",
        "class_id":    event["class_id"],
        "class_name":  event["class_name"],
        "confidence":  ddb_num(event["confidence"]),
        "is_fall":     bool(is_fall),
        "z_mean":      ddb_num(feat[2]),
        "x_mean":      ddb_num(feat[0]),
        "y_mean":      ddb_num(feat[1]),
        "height_range": ddb_num(feat[11]),
        "n_points":    int(feat[9]),
        "expire_at":   ttl,
    }
    table.put_item(Item=item)

    return event


@app.get("/history")
def history(device_id: str, limit: int = 60):
    resp = table.query(
        KeyConditionExpression=Key("device_id").eq(device_id),
        ScanIndexForward=False,
        Limit=limit,
    )
    items = resp.get("Items", [])
    return [_json_safe(x) for x in items]


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
