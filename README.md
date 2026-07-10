# RadarWatch — Real-Time Fall Detection System

> **Branch: `running-on-aws`**  
> This branch contains everything currently running on AWS.  
> Read this before touching anything.

---

## System Overview

The system detects human falls in real-time using a **TI IWR6843 dual mmWave radar** sensor.  
Data flows: Radar → Raspberry Pi → AWS EC2 (fall detection) → React Dashboard → DynamoDB (fall log).

```
IWR6843 Radar
     │  USB/Serial (JSON frames at ~18 fps)
     ▼
Raspberry Pi 4
  [aws_watcher.py]      ← watches folder for JSON files from TI Visualizer
  [feature_extract.py]  ← extracts 20-dim feature vector per frame
     │  HTTP POST /frames/batch
     ▼
AWS EC2  t3.micro  (43.205.167.81)
  [radar-api-main.backup.py]  ← FastAPI + rule-based fall detector
     │  WebSocket /ws               │
     ▼                              ▼
React Dashboard               DynamoDB radar_events
(hosted on S3)                (FALL events, 24h TTL)
```

---

## Quick Reference

| Item | Value |
|---|---|
| EC2 IP | `43.205.167.81` |
| Dashboard URL | http://radar-frontend-output.s3-website.ap-south-1.amazonaws.com |
| AWS Region | `ap-south-1` (Mumbai) |
| Device ID | `rpi-1` |
| API health check | http://43.205.167.81/health |
| SSH key | `radar-ec2-key.pem` (keep secret — never commit) |

---

## Connecting to EC2

```bash
ssh -i radar-ec2-key.pem ubuntu@43.205.167.81
```

### Useful commands once inside EC2

```bash
# See running Docker containers
sudo docker ps

# View live API logs
sudo docker logs -f <container-id>

# Restart the API (required after any code change)
sudo docker ps -q | xargs sudo docker restart

# Check which API file is deployed
head -5 ~/radar-api-main.py

# Edit the API directly on the server (last resort)
nano ~/radar-api-main.py
```

---

## File Structure

```
.
├── radar-api-main.backup.py   ← The EC2 API — push this to the server
├── s3_upload.py               ← Uploads dashboard build to S3
├── Visualizer.py              ← Radar point cloud visualizer (run locally)
├── currently_running.md       ← Quick reference of what is deployed
├── fall_thresholds_backup.md  ← History of threshold changes
├── AWS_Configuration.docx     ← AWS cost breakdown document
│
├── frontend/                  ← React dashboard source
│   ├── src/App.jsx            ← All dashboard UI and logic
│   ├── src/index.css          ← All styles
│   ├── package.json
│   └── vite.config.js
│
└── rpi_pipeline/              ← Raspberry Pi code
    ├── aws_watcher.py         ← Main script: watch folder, send to EC2
    ├── feature_extract.py     ← Extract 20-dim features from each radar frame
    ├── config.py              ← All settings (EC2 IP, serial port, device ID)
    ├── radar_capture.py       ← Low-level serial reader for the radar
    ├── inject_test_data.py    ← Simulate a fall without the physical radar
    └── requirements.txt       ← Python dependencies for the RPi
```

---

## Running the RPi Pipeline

### First-time setup

```bash
cd rpi_pipeline
pip install -r requirements.txt
```

### Start streaming to EC2

```bash
# Replace /path/to/json with the folder where TI Visualizer saves files
python aws_watcher.py /path/to/json/folder

# To also replay files already in the folder:
python aws_watcher.py /path/to/json/folder --process-existing
```

Stop with `Ctrl-C`.

### Edit connection settings (`config.py`)

```python
SERIAL_PORT_DATA = "/dev/ttyACM0"                  # Radar USB data port on RPi
SERIAL_PORT_CFG  = "/dev/ttyACM1"                  # Radar USB config port on RPi
CLOUD_API_URL    = "http://43.205.167.81/frame"     # EC2 API endpoint
DEVICE_ID        = "rpi-1"                          # Identifier for this device
```

### Test without the physical radar

```bash
cd rpi_pipeline
python inject_test_data.py
```

Sends simulated fall frames to EC2. The dashboard should show FALL detected.

---

## 20-Dimension Feature Vector

| Index | Feature | Used by fall detector? |
|---|---|---|
| 0 | x_mean — mean horizontal position (m) | No |
| 1 | y_mean — mean depth (m) | No |
| **2** | **z_mean — centroid height (m)** | **YES** |
| 3–8 | Velocity and acceleration (vx, vy, vz, ax, ay, az) | No |
| **9** | **n_points — number of radar reflection points** | **YES** |
| 10 | spread_xy — horizontal body spread | No |
| **11** | **height_range — max(z) minus min(z)** | **YES** |
| 12–17 | EKF tracker: track_x/y/z, track_vx/vy/vz | No |
| 18 | person_height (head height from TLV) | No |
| 19 | bottom_height (feet height from TLV) | No |

Only indices 2, 9, and 11 are used by the current rule-based fall detector.

---

## Fall Detection Logic

Three conditions must all be **true at the same time**:

### Thresholds (in `radar-api-main.backup.py` lines 56–62)

```python
Z_DROP_THRESHOLD      = 0.90   # z must drop >= 90 cm from its peak
BODY_FLAT_THRESH      = 0.40   # height_range <= 40 cm (person lying flat)
LOW_POINTS_THRESH     = 8      # n_points must be <= 8
MIN_POINTS_VALID      = 3      # n_points must be >= 3
DETECTION_WINDOW      = 20     # Rolling look-back: last 20 frames
SUSTAINED_DROP_FRAMES = 3      # Drop must persist >= 3 consecutive frames
COOLDOWN_FRAMES       = 150    # No re-trigger for ~8 seconds after a fall
```

### Trigger logic

```
Condition 1 (height_dropped):
  z_peak (90th-percentile of last 20 frames)
  minus z_current (mean of last 3 frames) >= 0.90 m
  AND this has been true for >= 3 consecutive frames

Condition 2 (body_flat):
  height_range <= 0.40 m (body is horizontal)

Condition 3 (few_points):
  3 <= n_points <= 8

is_fall = Condition1 AND Condition2 AND Condition3 AND cooldown == 0
```

### Human count guard
If `human_count == 0` (no person detected by TI tracker), detection is **blocked** and the internal z-buffer is **cleared**. This prevents false positives when the room is empty.

### Important: Warm-up period
The detector needs **20 frames (~1–2 seconds)** of standing data before it can detect anything. Ensure the person stands still for **20–25 seconds** before performing a test fall.

### After a fall triggers
There is an **8-second cooldown** (150 frames at 18 fps). The detector will not trigger again during this time.

### Note on z values
The radar z-axis depends on sensor mounting. For the current setup, a standing person appears at ~2.0 m in radar coordinates. A fall produces a ~1.5–2.0 m z-drop, well above the 0.90 m threshold.

---

## Dashboard Layout

| Area | What it shows |
|---|---|
| Top bar | Events logged in session, current status, average confidence |
| Left — top | **Last Fall Detected** with timestamp, metrics, and Acknowledge button |
| Left — middle | **Current Activity** — latest inference label and sensor metrics |
| Left — bottom | **Inference History** — scrollable log of all recent frames |
| Right | **Humans in Frame** count |

### Acknowledge button behaviour
- Fall detected → large red panel with "Acknowledge" button
- Click Acknowledge → collapses to a compact one-liner (time + elapsed + green badge)
- New fall arrives → automatically expands again regardless of acknowledged state

---

## Deploying Code Changes to EC2

### Update the API (fall detector / thresholds)

```bash
# 1. Edit radar-api-main.backup.py locally

# 2. Copy to EC2 (it becomes radar-api-main.py on the server)
scp -i radar-ec2-key.pem radar-api-main.backup.py ubuntu@43.205.167.81:~/radar-api-main.py

# 3. Restart the container (takes ~5 seconds)
ssh -i radar-ec2-key.pem ubuntu@43.205.167.81 "sudo docker ps -q | xargs sudo docker restart"
```

### Update the dashboard (frontend)

```bash
# 1. Edit files in frontend/src/

# 2. Build
cd frontend
npm install       # first time only
npm run build

# 3. Copy dist/ to EC2
scp -r dist ubuntu@43.205.167.81:~/frontend-dist

# 4. Run the upload script on EC2
ssh -i ../radar-ec2-key.pem ubuntu@43.205.167.81 "python3 ~/s3_upload.py"
```

---

## DynamoDB — Fall Event Database

| Setting | Value |
|---|---|
| Table name | `radar_events` |
| Partition key | `device_id` (String) — e.g. `rpi-1` |
| Sort key | `ts` (String, ISO 8601 timestamp) |
| TTL | `expire_at` — events auto-delete after **24 hours** |
| Stores | FALL events only (not NO-FALL frames) |

To browse: AWS Console → DynamoDB → Tables → `radar_events` → Explore Items.

---

## AWS Costs

| Service | Purpose | Cost/month (after free tier ends Nov 2026) |
|---|---|---|
| EC2 t3.micro | Inference server | USD 8.06 |
| EBS gp3 16 GB | Server disk | USD 1.46 |
| Elastic IP | Static public IP | USD 3.60 |
| DynamoDB | Fall event log | ~USD 0.01 |
| S3 | Dashboard hosting | ~USD 0.01 |
| **Total** | | **~USD 13.14/month** |

Free tier ends: **19 November 2026**.

---

## Troubleshooting

**Dashboard shows Offline / no live stream**
1. SSH into EC2 and run `sudo docker ps` — check container is running
2. If stopped: `sudo docker start <container-id>`
3. Test: open http://43.205.167.81/health — should return `{"status":"ok"}`

**False positives when room is empty**
- The human_count guard prevents this — restart the EC2 container to reset the z-buffer

**Fall not detected during a test**
- Person must stand still for 20–25 seconds first (warm-up)
- Stay within radar range (approx 0.5–6 m from sensor)
- Wait 8 seconds after a previous fall (cooldown)

**RPi not sending data**
```bash
curl http://43.205.167.81/health     # is EC2 reachable?
ls /dev/ttyACM*                      # is radar USB connected?
ps aux | grep aws_watcher            # is the script running?
```

---

*Project: Real-Time Fall Detection using Dual mmWave Radar — IDP Phase 2*  
*Developer: Chirag Shetty*
