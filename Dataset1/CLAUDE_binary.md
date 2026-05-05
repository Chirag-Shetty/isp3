# CLAUDE.md — Binary Fall Detection Training Agent

## YOUR JOB
Create and run `train_fall_detection.ipynb` that:
1. Reads `aug_dataset_binary/` (augmented radar data)
2. Extracts 20-dim features per frame from raw JSON
3. Builds sliding windows of shape `(40, 20)`
4. Trains a binary `FallDetectionTransformerCNNLSTM` model
5. Saves model + scaler ready for deployment on Raspberry Pi

---

## THIS IS A BINARY PROBLEM

| Label | Class | Meaning |
|-------|-------|---------|
| 0 | NO-FALL | All normal activities |
| 1 | FALL | Any fall event |

Two output neurons. CrossEntropyLoss with 2 classes. Predict FALL if argmax == 1.

---

## DATASET LOCATION

```
aug_dataset_binary/
  class_0_nofall/     <- label 0
  class_1_fall/       <- label 1
  dataset_manifest.csv
```

Manifest columns: file, label, label_name, source, augmentation, n_frames

---

## FEATURE EXTRACTION — 20 features per frame

FRAME_DT=0.055, SNR_THRESHOLD=10.0. Features [0-11]: point cloud stats,
[12-17]: tracker xyz+velocity, [18-19]: heightData maxZ/minZ.

---

## SLIDING WINDOW

WINDOW_SIZE=40, STRIDE=3. Skip files with < 40 frames silently.

---

## MODEL: FallDetectionTransformerCNNLSTM

Linear(20,64) → TransformerEncoder(2 layers, nhead=4, ff=128) →
Conv1d(64,32,k=3)+BN+Dropout+MaxPool(2) → LSTM(32,64) → Linear(64,32) → Linear(32,2)

---

## TRAINING CONFIG

BATCH=16, EPOCHS=150, LR=0.001, WD=0.001, PATIENCE=25, GRAD_CLIP=1.0
Loss: CrossEntropyLoss(class_weights, label_smoothing=0.1)
Optimizer: AdamW, Scheduler: CosineAnnealingLR

---

## OUTPUT FILES

| File | Purpose |
|------|---------|
| `fall_detection_model_best.pth` | Best model weights |
| `fall_scaler.pkl` | Fitted StandardScaler |
| `evaluation.png` | Confusion matrix + ROC |
| `training_curves.png` | Loss + accuracy curves |
