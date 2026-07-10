# Currently Running — RadarWatch System
_Last updated: 2026-07-10_

---

## EC2 Instance

| Item | Value |
|---|---|
| IP | `43.205.167.81` |
| Region | `ap-south-1` (Mumbai) |
| Instance Type | `t3.micro` |
| OS | Ubuntu 22.04 LTS |
| Container ID | `256eea78ddb9` |
| SSH Key | `radar-ec2-key.pem` |
| SSH Command | `ssh -i radar-ec2-key.pem ubuntu@43.205.167.81` |

---

## Which File Is Running on EC2

**File on EC2:** `~/radar-api-main.py`
**Source:** `radar-api-main.backup.py` (local) — this is the **correct, working version**

> Do NOT push `radar-api-main.py` (local). That has old broken changes.
> Always push `radar-api-main.backup.py` renamed to `radar-api-main.py` on EC2.

**SCP command to deploy:**
```powershell
scp -i "radar-ec2-key.pem" -o StrictHostKeyChecking=no radar-api-main.backup.py ubuntu@43.205.167.81:~/radar-api-main.py
ssh -i "radar-ec2-key.pem" -o StrictHostKeyChecking=no ubuntu@43.205.167.81 "sudo docker ps -q | xargs sudo docker restart"
```

---

## Fall Detector — Current Thresholds

```python
Z_DROP_THRESHOLD      = 0.90   # z must drop >= 90 cm from P90 peak
BODY_FLAT_THRESH      = 0.40   # height_range <= 0.40 m -> body horizontal
LOW_POINTS_THRESH     = 8      # n_points must be <= 8
MIN_POINTS_VALID      = 3      # n_points must be >= 3
DETECTION_WINDOW      = 20     # rolling look-back window (frames)
SUSTAINED_DROP_FRAMES = 3      # drop must persist >= 3 consecutive frames
COOLDOWN_FRAMES       = 150    # ~8s at 18fps — no re-trigger during cooldown
```

**Trigger logic — ALL 3 must be true + cooldown == 0:**
```python
height_dropped = drop_streak >= 3       # z dropped >=90cm for 3 frames
body_flat      = hrng <= 0.40           # body spread < 40cm vertically
few_points     = 3 <= n_points <= 8    # small point cloud (person on ground)
is_fall        = height_dropped and body_flat and few_points and cooldown == 0
```

**Human count guard (added 2026-07-10):**
- If `human_count == 0` -> z buffer is **cleared**, detection blocked immediately
- Prevents false positives when room is empty but buffer has stale standing z values

**z computation:**
```python
z_peak    = np.percentile(last_20_z_values, 90)   # spike-resistant peak
z_current = np.mean(last_3_z_values)               # smoothed current height
z_drop    = z_peak - z_current
```

---

## Frontend (React Dashboard)

| Item | Value |
|---|---|
| S3 Bucket | `radar-frontend-output` |
| URL | http://radar-frontend-output.s3-website.ap-south-1.amazonaws.com/ |
| Local source | `frontend/` |
| Build output | `frontend/dist/` |
| Upload script | `s3_upload.py` |

**Build and deploy:**
```powershell
cd frontend
npm run build
cd ..
python s3_upload.py
```

**Layout (current):**
- Left column (big): Last Fall Detected (with Acknowledge button) -> Current Activity -> Inference History
- Right column: Humans in Frame only

**Acknowledge button behaviour:**
- Fall detected -> big red panel, "Acknowledge" button appears top-right
- Click Acknowledge -> collapses to one-liner (time + elapsed + green badge + View button)
- New fall arrives -> automatically resets to full expanded view

**Last Fall persistence fix:**
- lastFall stored in its own useState — never evicted by the 200-frame rolling array
- Seeded from DynamoDB history on page load
- Updated immediately on every FALL WebSocket message

---

## AWS Services

| Service | Config | Monthly Cost (post free tier) |
|---|---|---|
| EC2 t3.micro | Always-on, Linux | USD 8.06 |
| EBS gp3 ~16 GB | Root volume (OS + Docker + app) | USD 1.46 |
| Elastic IP | Static public IPv4 | USD 3.60 |
| DynamoDB radar_events | On-demand, 24h TTL, FALL events only | ~USD 0.01 |
| S3 radar-frontend-output | Static website, ~580 KB | ~USD 0.01 |
| **Total** | | **~USD 13.14/mo** |

Free tier active until **November 19, 2026**.

---

## DynamoDB Table

| Attribute | Value |
|---|---|
| Table | `radar_events` |
| Partition key | `device_id` (String) |
| Sort key | `ts` (String ISO 8601) |
| TTL field | `expire_at` (Unix epoch, 24h) |
| Stores | FALL events only (not NO-FALL frames) |
| Device ID | `rpi-1` |

---

## RPi Pipeline

| File | Purpose |
|---|---|
| `rpi_pipeline/aws_watcher.py` | Main loop — reads radar serial, extracts features, POSTs to EC2 |
| `rpi_pipeline/feature_extract.py` | Extracts 20-dim feature vector from each radar frame |
| `rpi_pipeline/config.py` | RPi config (serial port, EC2 IP, device ID) |
| `rpi_pipeline/inject_test_data.py` | Test script — injects simulated fall frames to EC2 API |

**20-dim feature vector groups:**
- [0-11]  Point cloud: x, y, z, vx, vy, vz, ax, ay, az, n_points, spread_xy, height_range
- [12-17] EKF tracker: track_x, track_y, track_z, track_vx, track_vy, track_vz
- [18-19] Height TLV: person_height, bottom_height

**Only 3 features used by the fall detector:** index 2 (z_mean), index 9 (n_points), index 11 (height_range)

---

## Key Local Files

| File | What it is |
|---|---|
| `radar-api-main.backup.py` | The working API — push THIS to EC2 |
| `radar-api-main.py` | Old modified version — do NOT push |
| `fall_thresholds_backup.md` | Threshold history and change log |
| `currently_running.md` | This file |
| `radar-ec2-key.pem` | SSH/SCP key for EC2 |
| `frontend/src/App.jsx` | React dashboard main component |
| `frontend/src/index.css` | All dashboard styles |
| `s3_upload.py` | Uploads dist/ to S3 |
| `AWS_Configuration.docx` | AWS cost breakdown document (for mentor) |
