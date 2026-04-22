# IWR6843 → Supabase Cloud Pipeline

Real-time fall detection pipeline: **Raspberry Pi → Supabase (free cloud)**

---

## File Structure

```
rpi_pipeline/
├── config.py            ← ⚠ fill in your Supabase credentials here
├── radar_capture.py     ← reads raw frames from IWR6843 over serial
├── feature_extract.py   ← extracts 12-dim features (mirrors training notebook)
├── model.py             ← Transformer-CNN-LSTM architecture (identical to Colab)
├── rpi_sender.py        ← MAIN script — ties everything together
├── requirements.txt     ← Python dependencies
├── best_model.pth       ← ⚠ copy this from Google Colab / Drive
├── scaler.pkl           ← ⚠ copy this from Google Colab / Drive
└── cloud/
    └── supabase_setup.sql  ← run once in Supabase SQL editor
```

---

## Step 1 — Create a Free Supabase Account

1. Go to **https://supabase.com** → click **Start your project**
2. Sign up with GitHub (takes 30 seconds, no credit card)
3. Click **New Project** → name it (e.g. `fall-detection`) → set a DB password
4. Wait ~2 minutes while it provisions

---

## Step 2 — Create the Database Table

1. In your Supabase dashboard → click **SQL Editor** (left sidebar)
2. Click **New query**
3. Paste the entire contents of `cloud/supabase_setup.sql`
4. Click **Run** (green button, top right)
5. You should see: `Success. No rows returned`

---

## Step 3 — Get Your API Keys

1. In Supabase dashboard → **Settings** (gear icon) → **API**
2. Copy two values:
   - **Project URL** — looks like `https://abcdefg.supabase.co`
   - **anon/public** key — a long JWT string
3. Open `config.py` and paste them:

```python
SUPABASE_URL      = "https://abcdefg.supabase.co"    # ← your URL
SUPABASE_ANON_KEY = "eyJhbGci..."                     # ← your anon key
```

---

## Step 4 — Copy Model Files to RPi

From **Google Colab/Drive**, download:
- `best_model.pth` (saved during training)
- `scaler.pkl` (saved during data pipeline step)

Copy them to the **same folder** as `rpi_sender.py`:

```bash
# From your PC (Windows):
scp best_model.pth pi@<rpi-ip>:~/fall_detection_pipeline/
scp scaler.pkl     pi@<rpi-ip>:~/fall_detection_pipeline/
```

---

## Step 5 — Install Dependencies on RPi

SSH into your Raspberry Pi, then:

```bash
cd ~/fall_detection_pipeline

# Install Python packages
pip install pyserial numpy requests scikit-learn

# Install PyTorch (ARM CPU build for RPi)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> ⚠ PyTorch install can take 10–15 minutes on RPi. That's normal.

---

## Step 6 — Configure Serial Port

Check which port your IWR6843 uses:

```bash
ls /dev/ttyACM*
# or
ls /dev/ttyUSB*
```

Edit `config.py`:
```python
SERIAL_PORT_DATA = "/dev/ttyACM0"   # data port
SERIAL_PORT_CFG  = "/dev/ttyACM1"   # config port
```

---

## Step 7 — Run the Pipeline!

```bash
cd ~/fall_detection_pipeline
python rpi_sender.py
```

You should see:
```
============================================================
  IWR6843 → Supabase Fall Detection Pipeline
============================================================
  Supabase URL : https://abcdefg.supabase.co
  Serial port  : /dev/ttyACM0
  Inference    : ON  (./best_model.pth)
============================================================
[model] Loaded weights from ./best_model.pth (epoch 32, val_acc=97.63%)
[model] Loaded scaler from ./scaler.pkl
[radar] Config sent to /dev/ttyACM1
[radar] Listening on /dev/ttyACM0 @ 921600 baud …

[pipeline] Streaming … (Ctrl-C to stop)

  [14:23:01] frame=00040 | pred=Standing_walk               conf=94.2%  ✓ normal
  [14:23:03] frame=00045 | pred=stand_floor_transition       conf=87.1%  🚨 FALL DETECTED!
```

---

## Step 8 — View Live Data in Supabase

1. Supabase dashboard → **Table Editor** → `radar_predictions`
2. You'll see rows appearing in real-time
3. Or run SQL: `SELECT * FROM fall_events ORDER BY timestamp DESC LIMIT 10;`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied: /dev/ttyACM0` | Run: `sudo usermod -a -G dialout $USER` then log out/in |
| `CONFIG ERROR: YOUR_PROJECT_ID` | Fill in `config.py` with your real Supabase URL |
| `best_model.pth not found` | Copy the file from Colab to the RPi folder |
| No frames received from radar | Check serial port with `ls /dev/ttyACM*`, try both ports |
| Supabase 401 Unauthorized | Check your anon key in `config.py` |

---

## Architecture

```
IWR6843 Radar
     │ USB serial (921600 baud)
     ▼
Raspberry Pi
  radar_capture.py  → reads binary TLV frames
  feature_extract.py → extracts 12 features per frame
  model.py           → Transformer-CNN-LSTM inference
  rpi_sender.py      → batches into windows, sends to cloud
          │
          │ HTTPS POST (every ~2 seconds for 40-frame windows)
          ▼
 Supabase (free tier)
   radar_predictions table  ← stores all predictions
   fall_events view          ← fall rows only
   Realtime channel          ← live WebSocket for mobile app
```
