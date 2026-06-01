# RPi Pipeline — AWS Streaming

Streams radar frame features from the IWR6843 sensor (or JSON files) to the AWS EC2 API for real-time fall detection.

## Files

| File | Purpose |
|---|---|
| `config.py` | All settings — AWS URL, serial ports, window size |
| `feature_extract.py` | Extracts 20-dim feature vector from each radar frame |
| `radar_capture.py` | Reads serial data from IWR6843 and yields frame dicts |
| `model.py` | TransformerCNNLSTM architecture (used by AWS server) |
| `aws_watcher.py` | **Watches a folder for JSON files → streams features to AWS** |
| `cloud_stream_sender.py` | **Reads live serial data → streams features to AWS** |
| `cloud_stream_simulator.py` | Replays a single JSON file → streams to AWS (for testing) |
| `multi_class_model_best.pth` | Trained model weights |
| `multi_class_scaler.pkl` | StandardScaler for feature normalisation |
| `watch_test/` | Sample JSON files for testing |

## Quick Start

### Option A — Radar connected via USB (live streaming)
```bash
pip install -r requirements.txt
python cloud_stream_sender.py
```

### Option B — No radar connected (JSON file watcher)
```bash
pip install -r requirements.txt
# Drop replay JSON files into a folder, then:
python aws_watcher.py ./watch_test --process-existing
```

### Option C — Test with a single JSON file
```bash
python cloud_stream_simulator.py watch_test/replay_test_1.json --loop
```

## Configuration

Edit `config.py` or set environment variables:

```bash
# AWS API endpoint
export CLOUD_API_URL="http://43.205.167.81/frame"

# Serial ports (Linux defaults for IWR6843)
export SERIAL_PORT_DATA="/dev/ttyACM0"
export SERIAL_PORT_CFG="/dev/ttyACM1"

# Device identifier sent with each frame
export DEVICE_ID="rpi-1"
```

## Monitor on EC2
```bash
docker logs -f radar-api
```
