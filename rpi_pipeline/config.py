# ============================================================
#  config.py  —  Fill in your Supabase credentials here
# ============================================================
#
# HOW TO GET THESE VALUES:
#   1. Go to https://supabase.com and sign up (free, no card)
#   2. Create a new project
#   3. Go to Settings → API
#   4. Copy "Project URL" and "anon/public key" below
#
# ============================================================

SUPABASE_URL = "https://ozopcneaghprtmnfrokh.supabase.co"    # ← paste here
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im96b3BjbmVhZ2hwcnRtbmZyb2toIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM2NTE5NDMsImV4cCI6MjA4OTIyNzk0M30.K-Sh_vTb0h_7dIGE9pHoRXc4IbNTWaU5yZzjxYwdca8"              # ← paste here

# ============================================================
#  Radar / Serial Settings
# ============================================================
# Common values on Raspberry Pi:
#   /dev/ttyACM0   (most IWR6843 configs)
#   /dev/ttyUSB0
#   /dev/ttyACM1
SERIAL_PORT_DATA  = "/dev/ttyACM0"   # Data port of IWR6843
SERIAL_PORT_CFG   = "/dev/ttyACM1"   # Config port of IWR6843
SERIAL_BAUD       = 921600

# ============================================================
#  Pipeline Settings
# ============================================================
WINDOW_SIZE   = 40    # frames per inference window (matches training)
STRIDE        = 5     # frames to slide window by
SNR_THRESHOLD = 10.0  # minimum SNR to keep a point
FRAME_DT      = 0.055 # seconds between frames (~18 fps)
NUM_FEATURES  = 20    # feature vector size (matches multi_class_model_best.pth)

# ============================================================
#  Model Settings
# ============================================================
# Path to your trained model .pth file (only used if RUN_INFERENCE=True and
# INFERENCE_URL is empty, i.e. local inference fallback)
MODEL_PATH  = "./multi_class_model_best.pth"
SCALER_PATH = "./multi_class_scaler.pkl"

# Set to False to skip ML inference and only stream raw features
RUN_INFERENCE = True

# ============================================================
#  Cloud Inference (Hugging Face Space)
# ============================================================
# Paste your HF Space URL here after deploying.
# Format: https://<owner>-<space-name>.hf.space
# Leave blank ("") to fall back to local inference on the RPi.
INFERENCE_URL = "https://chiragshetty888-fall-detection-api.hf.space/predict"

# ============================================================
#  Class Labels (must match training order)
# ============================================================
CLASSES = [
    "Standing_walk",              # 0
    "Sitting_chair",              # 1
    "sitting_floor",              # 2
    "Stand_Sit_chair_transition", # 3
    "chair_floor_transition",     # 4
    "stand_floor_transition",     # 5
]
FALL_CLASS_IDS = {5}  # stand_floor_transition can indicate a fall
