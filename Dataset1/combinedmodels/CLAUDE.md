# CLAUDE.md — Antigravity Fall Detection Agent
## IWR6843AOP mmWave Radar · Fall / No-Fall Classifier

> **Context for this agent:** We have very limited labeled data (≈10 samples per class).
> The strategy is: pretrained models first → statistical/threshold methods second → learned
> model only as a final head fine-tuned on top of frozen weights.
> Do NOT try to train anything from scratch on this data.

---

## 1. MISSION

Build a fall / no-fall binary classifier for the **Texas Instruments IWR6843AOP** mmWave radar.
The radar outputs sparse **3D point clouds** per frame — each point has `(x, y, z, Doppler velocity, SNR, noise)`.
We typically get **≤ 10 cloud points per frame** in a room setting.

The pipeline must work **reliably with small data** using:
1. Rule-based / statistical thresholds — zero training needed
2. Pretrained model weights from published open-source repos
3. Anomaly detection trained only on no-fall data — no fall labels needed
4. Fine-tuning only the final classification head using the 10 labeled samples

---

## 2. INPUT DATA FORMAT

Every radar frame from IWR6843AOP produces a list of detected points.
Each point is a vector:

```
point = [x, y, z, doppler, SNR, noise]
```

| Field    | Unit    | Meaning                                     |
|----------|---------|---------------------------------------------|
| x        | meters  | lateral position (left/right)               |
| y        | meters  | depth (distance from radar along beam)      |
| z        | meters  | height above floor (after correction)       |
| doppler  | m/s     | radial velocity (negative = moving toward radar / falling) |
| SNR      | dB×0.1  | signal-to-noise ratio                       |
| noise    | dB×0.1  | noise floor level                           |

The **centroid** of each frame is the mean of all point positions:
```
centroid = (mean_x, mean_y, mean_z)
```

From the mmFall repo `data_pre.py`, the working feature set is:
```
features_per_point = 4     # (delta_x, delta_y, z, doppler) — drop SNR/noise
frames_per_pattern = 10    # sliding window = 1 second at 10 FPS radar rate
points_per_frame   = 64    # oversample every sparse frame to exactly 64 points
```

---

## 3. COORDINATE CORRECTION (MANDATORY FIRST STEP)

The IWR6843AOP is mounted tilted at **-10 degrees** from horizontal.
Apply this rotation matrix to every point before using any z values:

```python
import numpy as np

TILT_ANGLE   = -10.0   # degrees — radar tilt from horizontal
SENSOR_HEIGHT = 1.8    # meters  — how high the radar is mounted

R = np.array([
    [1, 0,                              0                             ],
    [0, np.cos(np.deg2rad(TILT_ANGLE)), -np.sin(np.deg2rad(TILT_ANGLE))],
    [0, np.sin(np.deg2rad(TILT_ANGLE)),  np.cos(np.deg2rad(TILT_ANGLE))]
])

def correct_point(x, y, z):
    rotated = R @ np.array([x, y, z])
    corrected_z = rotated[2] + SENSOR_HEIGHT   # ground is now z=0
    return rotated[0], rotated[1], corrected_z
```

After correction: `z ≈ 1.7 m` means person is standing; `z < 0.5 m` means person is on/near ground.

---

## 4. STRATEGY (ordered by priority — do in this order)

---

### TIER 1 — Rule-Based Threshold Detector
**No training. No model. Run this first always.**

Extract per-frame features:
```python
def extract_frame_features(points):
    """points: np.array shape (N, 4) — [x, y, z, doppler]"""
    if len(points) == 0:
        return 0.0, 0.0, 0.0
    centroid_z  = float(np.mean(points[:, 2]))
    doppler_avg = float(np.mean(points[:, 3]))
    z_spread    = float(np.std(points[:, 2]))
    return centroid_z, doppler_avg, z_spread
```

**Fall decision over a sliding window of 20 frames (2 seconds at 10 FPS):**

```python
# Thresholds — sourced from mmFall paper + ESPHome IWR6843 production code
Z_DROP_THRESHOLD    = 0.6   # meters — how much centroid must drop to flag as fall
DOPPLER_THRESHOLD   = -0.5  # m/s    — sudden downward velocity confirms fall
Z_SPREAD_THRESHOLD  = 0.3   # meters — cloud is flat = person is on floor
DETECTION_WINDOW    = 20    # frames — sliding window size (2 seconds)
SUSTAINED_FRAMES    = 10    # frames — must stay low for ~1 sec

def rule_based_fall(centroidZ_history, doppler_avg, z_spread):
    """
    centroidZ_history: list of recent centroid_z values (most recent last)
    Returns: True if fall detected
    """
    if len(centroidZ_history) < DETECTION_WINDOW:
        return False

    left_edge  = centroidZ_history[-DETECTION_WINDOW]
    right_edge = centroidZ_history[-1]
    z_drop     = left_edge - right_edge          # positive = dropped

    height_dropped  = z_drop >= Z_DROP_THRESHOLD
    fast_descent    = doppler_avg < DOPPLER_THRESHOLD
    body_flat       = z_spread < Z_SPREAD_THRESHOLD

    return height_dropped and (fast_descent or body_flat)
```

**Reference threshold values from open-source IWR6843 code (ESPHome repo):**

| Parameter             | Value    | What it means                              |
|-----------------------|----------|--------------------------------------------|
| centroid_z drop       | 0.6 m    | person must drop ≥60 cm to be a fall       |
| detection window      | 20 frames| 2-second look-back window                  |
| presence z_min        | 0.5 m    | below this = person is on floor            |
| presence z_max        | 2.5 m    | above this = not a person                  |
| tracking z_min        | -0.5 m   | extended lower bound for tracking          |
| tracking z_max        | 3.0 m    | extended upper bound for tracking          |
| ceiling height        | 290 cm   | sensor mounted at 2.9 m                    |
| anomaly threshold     | 0.3      | mmFall HVRAE loss spike threshold          |

---

### TIER 2 — mmFall HVRAE (Pretrained Anomaly Model, No Fall Labels Needed)

**Repo:** `github.com/radar-lab/mmfall`
**Pretrained weights:** `mmfall/saved_model/VRAE_mdl_local4.h5`
**Paper:** `arxiv.org/abs/2003.02386`
**Key insight:** Trained ONLY on normal activities. Falls are detected as anomalies — you need ZERO fall-labeled data to use this.

**Step 1 — Oversample your sparse frames to 64 points (exact method from data_pre.py):**

```python
def oversample_frame_to_64(frame_points, N=64):
    """
    frame_points: np.array shape (M, 4) — your actual M detected points
    Returns:      np.array shape (64, 4) — oversampled, same mean and variance
    """
    frame_np = np.array(frame_points, dtype=float)
    M = frame_np.shape[0]

    if M == 0:
        return np.zeros((N, 4))
    if M >= N:
        return frame_np[:N]                          # trim if somehow too many

    mean  = np.mean(frame_np, axis=0)

    # Rescale: preserves mean and standard deviation after padding
    scaled = np.sqrt(N / M) * frame_np + mean * (1 - np.sqrt(N / M))

    # Pad remaining rows with mean vector
    padding   = np.tile(mean, (N - M, 1))
    oversampled = np.vstack([scaled, padding])

    return oversampled                               # shape (64, 4)
```

**Step 2 — Build a sliding window of 10 frames:**

```python
def build_pattern(frame_buffer_10):
    """
    frame_buffer_10: list of 10 frames, each is np.array (M_i, 4)
    Returns: np.array shape (10, 64, 4) ready for mmFall model
    """
    return np.array([oversample_frame_to_64(f) for f in frame_buffer_10])
```

**Step 3 — Load pretrained model and compute anomaly loss:**

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.losses import mse

def load_mmfall_model(weights_path='mmfall/saved_model/VRAE_mdl_local4.h5'):
    def sampling_predict(args):
        Z_mean, Z_log_var = args
        batch  = tf.shape(Z_mean)[0]
        frames = Z_mean.shape[1]
        latent = Z_mean.shape[2]
        eps = tf.random.normal(shape=(batch, frames, latent))
        return Z_mean + tf.exp(0.5 * Z_log_var) * eps

    model = load_model(
        weights_path,
        compile=False,
        custom_objects={'sampling': sampling_predict, 'tf': tf}
    )
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=mse)
    return model


def compute_anomaly_loss(model, pattern_10_64_4):
    """
    pattern_10_64_4: np.array shape (10, 64, 4)
    Returns: float — anomaly loss. Higher = more abnormal = likely fall.
    """
    x = np.expand_dims(pattern_10_64_4, axis=0)    # (1, 10, 64, 4)

    prediction = model.predict(x, verbose=0)

    get_zmean   = tf.keras.Model(model.input, model.get_layer('qzx_mean').output)
    get_zlogvar = tf.keras.Model(model.input, model.get_layer('qzx_log_var').output)
    Z_mean   = get_zmean.predict(x, verbose=0)
    Z_logvar = get_zlogvar.predict(x, verbose=0)

    n_feat = x.shape[-1]
    pred_mean   = prediction[:, :, :, :n_feat].reshape(1, 10, -1)
    pred_logvar = prediction[:, :, :, n_feat:].reshape(1, 10, -1)
    pred_var    = np.exp(pred_logvar)
    x_flat      = x.reshape(1, 10, -1)

    log_pXz = np.sum(0.5 * np.square(x_flat - pred_mean) / (pred_var + 1e-8), axis=-1)
    kl_loss = -0.5 * np.sum(1 + Z_logvar - np.square(Z_mean) - np.exp(Z_logvar), axis=-1)
    loss = float(np.mean(log_pXz + kl_loss))
    return loss


def mmfall_detect(loss_history, centroidZ_history,
                  anomaly_threshold=0.3, z_drop_threshold=0.6, window=20):
    """
    Returns True if the current position (end of histories) is a fall.
    Logic: centroid dropped AND anomaly loss spiked within the same window.
    """
    if len(loss_history) < window or len(centroidZ_history) < window:
        return False

    z_drop    = centroidZ_history[-window] - centroidZ_history[-1]
    loss_peak = max(loss_history[-window:])

    return z_drop >= z_drop_threshold and loss_peak >= anomaly_threshold
```

---

### TIER 3 — Statistical Feature SVM (Trained on Your 10 Samples)

With only 10 samples, handcrafted features + SVM beats any deep learning approach.
This is your "learning from your own data" tier.

```python
def extract_statistical_features(frame_buffer_10):
    """
    frame_buffer_10: list of 10 frames, each frame is np.array (M_i, 4)
    Returns: 1D feature vector capturing the fall signature
    """
    centroids_z  = []
    doppler_vals = []
    spreads_z    = []
    n_points     = []

    for frame in frame_buffer_10:
        pts = np.array(frame)
        if len(pts) == 0:
            continue
        centroids_z.append(float(np.mean(pts[:, 2])))
        doppler_vals.append(float(np.mean(pts[:, 3])))
        spreads_z.append(float(np.std(pts[:, 2])))
        n_points.append(len(pts))

    cz = np.array(centroids_z) if centroids_z else np.zeros(1)
    dv = np.array(doppler_vals) if doppler_vals else np.zeros(1)
    sz = np.array(spreads_z)   if spreads_z   else np.zeros(1)

    feats = [
        float(np.min(cz)),                                       # lowest height reached
        float(np.max(cz) - np.min(cz)),                         # height range (drop size)
        float(np.mean(cz)),                                      # mean height
        float(np.std(cz)),                                       # height variability
        float(cz[0] - cz[-1]),                                   # net drop start→end
        float(np.min(dv)),                                       # most negative velocity
        float(np.mean(dv)),                                      # mean velocity
        float(np.max(np.abs(dv))),                              # peak velocity magnitude
        float(np.mean(sz)),                                      # mean vertical spread
        float(np.min(sz)),                                       # flattest moment
        float(np.mean(n_points)) if n_points else 0.0,          # mean point count
        float(np.min(np.diff(cz))) if len(cz) > 1 else 0.0,   # fastest descent
        # skewness of height trace — falls are asymmetric (sharp down, slow up)
        float(np.mean((cz - np.mean(cz))**3) / (np.std(cz)**3 + 1e-8)),
        # autocorrelation lag-1 of height trace
        float(np.corrcoef(cz[:-1], cz[1:])[0, 1]) if len(cz) > 2 else 0.0,
    ]
    return np.array(feats, dtype=float)


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_svm(X_features, y_labels):
    """
    X_features: np.array (n_samples, n_features) — typically (10, 14)
    y_labels:   np.array (n_samples,)             — 0=no_fall, 1=fall
    """
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=1.0,
            probability=True,       # needed for confidence scores
            class_weight='balanced' # handles any class imbalance
        ))
    ])
    clf.fit(X_features, y_labels)
    return clf
```

---

## 5. FULL ENSEMBLE PIPELINE

```python
class AntiFallDetector:
    """
    Three-tier ensemble detector for IWR6843AOP.
    Final decision = majority vote of active tiers.
    """

    def __init__(self, mmfall_weights_path=None, svm_model=None):
        self.mmfall_model = None
        if mmfall_weights_path:
            self.mmfall_model = load_mmfall_model(mmfall_weights_path)
        self.svm = svm_model

        # Rolling history buffers
        self.centroidZ_history = []
        self.loss_history      = []
        self.frame_buffer      = []   # last 10 raw frames

    def process_frame(self, raw_points_Nx6):
        """
        raw_points_Nx6: np.array shape (N, 6) — [x,y,z,doppler,snr,noise]
        Returns: dict with per-tier and ensemble fall decisions
        """
        pts = self._correct_coordinates(raw_points_Nx6)
        pts4 = pts[:, :4]   # keep x,y,z,doppler only

        cz, dv, sz = extract_frame_features(pts4)
        self.centroidZ_history.append(cz)
        self.frame_buffer.append(pts4)
        if len(self.frame_buffer) > 10:
            self.frame_buffer.pop(0)

        result = {
            'centroid_z': cz, 'doppler': dv, 'z_spread': sz,
            'rule_fall':  False,
            'mmfall_fall': False,
            'svm_fall':   False,
            'ensemble_fall': False,
            'confidence': 0.0,
        }

        # TIER 1
        result['rule_fall'] = rule_based_fall(self.centroidZ_history, dv, sz)

        # TIER 2 — needs 10 frames buffered
        if self.mmfall_model and len(self.frame_buffer) == 10:
            pattern = build_pattern(self.frame_buffer)
            loss = compute_anomaly_loss(self.mmfall_model, pattern)
            self.loss_history.append(loss)
            result['mmfall_fall'] = mmfall_detect(
                self.loss_history, self.centroidZ_history)

        # TIER 3 — SVM on statistical features
        if self.svm and len(self.frame_buffer) == 10:
            feats = extract_statistical_features(self.frame_buffer)
            prob  = float(self.svm.predict_proba([feats])[0][1])
            result['svm_fall'] = prob > 0.5

        # Ensemble majority vote
        votes = sum([result['rule_fall'], result['mmfall_fall'], result['svm_fall']])
        n_active = sum([True, self.mmfall_model is not None, self.svm is not None])
        result['ensemble_fall'] = votes > n_active / 2
        result['confidence']    = round(votes / n_active, 2)

        return result

    def _correct_coordinates(self, raw_pts):
        R = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(TILT_ANGLE)), -np.sin(np.deg2rad(TILT_ANGLE))],
            [0, np.sin(np.deg2rad(TILT_ANGLE)),  np.cos(np.deg2rad(TILT_ANGLE))]
        ])
        pts = raw_pts.copy().astype(float)
        pts[:, :3] = (R @ pts[:, :3].T).T
        pts[:, 2] += SENSOR_HEIGHT
        return pts
```

---

## 6. CALIBRATION WITH YOUR 10 SAMPLES

Use your labeled samples to tune thresholds — do NOT use them to train deep models.

```python
def calibrate_from_10_samples(fall_windows, nofall_windows):
    """
    fall_windows:   list of 10-frame windows where falls happen (your fall samples)
    nofall_windows: list of 10-frame windows of normal activity (your no-fall samples)

    Each window is a list of 10 frames, each frame = np.array (M, 4)
    """
    import matplotlib.pyplot as plt

    # --- Find optimal Z_DROP threshold ---
    min_z_fall   = [min(np.mean(f[:, 2]) for f in w if len(f) > 0)
                    for w in fall_windows]
    min_z_nofall = [min(np.mean(f[:, 2]) for f in w if len(f) > 0)
                    for w in nofall_windows]

    mean_fall_z   = np.mean(min_z_fall)
    mean_nofall_z = np.mean(min_z_nofall)
    suggested_z   = (mean_fall_z + mean_nofall_z) / 2.0   # midpoint

    print(f"Min centroid_z  FALL: {mean_fall_z:.2f} m")
    print(f"Min centroid_z  NO-FALL: {mean_nofall_z:.2f} m")
    print(f"Suggested Z_DROP threshold: {suggested_z:.2f} m")

    # --- Train SVM on statistical features ---
    all_wins = fall_windows + nofall_windows
    all_lbls = [1] * len(fall_windows) + [0] * len(nofall_windows)
    feats    = np.array([extract_statistical_features(w) for w in all_wins])
    svm      = train_svm(feats, np.array(all_lbls))

    # --- Leave-one-out evaluation (max you can do with 10 samples) ---
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score
    loo = LeaveOneOut()
    preds = []
    for train_idx, test_idx in loo.split(feats):
        m = train_svm(feats[train_idx], np.array(all_lbls)[train_idx])
        preds.append(int(m.predict([feats[test_idx[0]]])[0]))
    loo_acc = accuracy_score(all_lbls, preds)
    print(f"SVM LOO accuracy on 10 samples: {loo_acc:.0%}")

    return svm, suggested_z
```

---

## 7. OUTPUT FORMAT PER FRAME

```python
{
    "frame_id":       int,    # frame counter
    "timestamp":      float,  # seconds since start
    "num_points":     int,    # raw detected points in this frame
    "centroid_z":     float,  # meters — person height estimate
    "centroid_x":     float,  # meters
    "centroid_y":     float,  # meters
    "doppler":        float,  # m/s — average radial velocity
    "z_spread":       float,  # meters — vertical spread of cloud

    "rule_fall":      bool,   # TIER 1 decision (always active)
    "mmfall_fall":    bool,   # TIER 2 decision (active if weights loaded)
    "svm_fall":       bool,   # TIER 3 decision (active if model trained)
    "ensemble_fall":  bool,   # FINAL decision — majority vote
    "confidence":     float,  # 0.0 / 0.33 / 0.67 / 1.0
}
```

---

## 8. DEPENDENCIES

```
tensorflow>=2.8
numpy
scipy
scikit-learn
matplotlib
```

Install:
```bash
pip install tensorflow numpy scipy scikit-learn matplotlib
```

Get pretrained weights:
```bash
git clone https://github.com/radar-lab/mmfall.git
# weights at: mmfall/saved_model/VRAE_mdl_local4.h5
```

---

## 9. RULES FOR THIS AGENT

- NEVER train a deep neural network from scratch on 10 samples
- ALWAYS run TIER 1 rule-based first — it is the fastest and most reliable with small data
- ALWAYS apply coordinate correction (tilt rotation + sensor height) before any feature extraction
- ALWAYS oversample frames to 64 points before feeding to mmFall model
- Use `class_weight='balanced'` in SVM to handle any class imbalance
- Use Leave-One-Out cross-validation (not train/test split) when evaluating on 10 samples
- Tune `Z_DROP_THRESHOLD` using midpoint between your fall and no-fall min-z values
- If only TIER 1 is available, still report ensemble_fall = rule_fall

---

## 10. REFERENCES

| Resource | URL | What it contains |
|---|---|---|
| mmFall paper + code | github.com/radar-lab/mmfall | HVRAE pretrained weights, data_pre.py, thresholds |
| RadHAR pretrained BiLSTM | github.com/nesl/RadHAR | Activity recognition weights for TI mmWave |
| ESPHome IWR6843 | github.com/bytelink-ai/esphome-iwr6843 | fall_detection.h, real production thresholds |
| MiliPoint dataset | github.com/yizzfz/MiliPoint | Large radar HAR dataset (DGCNN, PointNet++ baselines) |
| mmFall arxiv | arxiv.org/abs/2003.02386 | Algorithm details, ROC curves, threshold justification |

---

*Sensor: TI IWR6843AOP | Task: Fall Detection | Data regime: Few-shot (≤10 labeled samples) | Agent: Antigravity*