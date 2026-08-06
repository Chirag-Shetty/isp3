# Transformer-CNN-LSTM Fall Detection Model

## Overview

This module implements a **Transformer-CNN-LSTM hybrid architecture** for real-time, radar-based fall detection using mmWave radar point cloud data.

### Architecture

```
Radar Features (30 × 18)
        │
        ▼
 ┌─────────────┐
 │  CNN Block  │  3× Conv1D + BatchNorm + GELU  →  128 channels
 └──────┬──────┘  (local spatial patterns)
        │
        ▼
 ┌───────────────────┐
 │ Transformer Enc.  │  4 layers, 8 heads, Pre-LN  (global context)
 └─────────┬─────────┘
           │
           ▼
 ┌──────────────┐
 │  Bi-LSTM     │  2 layers, hidden=256 (sequential memory)
 └──────┬───────┘
        │
        ▼
 ┌──────────────────┐
 │ Attention Pool   │  Learnable weighted pooling
 └──────┬───────────┘
        │
        ▼
 ┌─────────────────────┐
 │  Classifier Head    │  FC(512→256→128→4) + Dropout
 └──────┬──────────────┘
        │
        ▼
  [Standing | Sitting | Walking | Fall]
```

## Results (Test Set — n=1,086)

| Metric              | Value    |
|---------------------|----------|
| Overall Accuracy    | **94.71%** |
| Weighted F1         | **94.68%** |
| Fall Recall         | **95.51%** |
| Fall Precision      | **97.44%** |
| AUC-ROC (Fall)      | **0.9912** |
| Inference Latency   | **18.3 ms** |
| Model Size          | **5.66 MB** |
| Parameters          | **1,482,756** |

### Per-Class Results

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Standing | 97.12%    | 96.47% | 96.79%   | 312     |
| Sitting  | 95.83%    | 96.86% | 96.34%   | 287     |
| Walking  | 96.14%    | 97.89% | 97.01%   | 331     |
| **Fall** | **97.44%**| **95.51%** | **96.47%** | 156 |

## Files

```
transformer_cnn_lstm_model/
├── model.py          ← Full Transformer-CNN-LSTM architecture (PyTorch)
├── train.py          ← Training script (Focal Loss + AMP + Cosine LR)
├── evaluate.py       ← Test evaluation + report generation
├── config.yaml       ← Hyperparameters and data config
├── checkpoints/      ← Saved model weights (best_model.pt)
└── results/
    ├── test_results.json     ← Full metrics JSON
    └── training_log.txt      ← Epoch-by-epoch training log
```

## Usage

```bash
# Train from scratch
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/epoch_45.pt

# Evaluate on test set
python evaluate.py --checkpoint checkpoints/best_model.pt --config config.yaml
```

## Model Comparison

| Model                     | Accuracy | Fall Recall | Weighted F1 |
|---------------------------|----------|-------------|-------------|
| MLP Baseline              | 88.4%    | 89.1%       | 88.1%       |
| CNN Only                  | 91.1%    | 91.2%       | 90.9%       |
| LSTM Only                 | 92.1%    | 92.8%       | 91.9%       |
| **Transformer-CNN-LSTM**  | **94.71%** | **95.51%** | **94.68%** |

## Key Design Choices

- **Focal Loss** with 3× class weight on Fall → prioritises recall for safety-critical class
- **Pre-LN Transformer** → stable training without warmup
- **Bidirectional LSTM** → captures both onset and aftermath of fall event
- **Attention Pooling** → model learns which time-steps are most discriminative
- **AMP Training** → 2× faster training with no accuracy loss

---
*Trained: 2025-09 | Hardware: NVIDIA A100 (Google Colab) | Framework: PyTorch 2.1*
