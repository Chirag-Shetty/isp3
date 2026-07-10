import os

# ============================================================
#  Radar / Serial Settings
# ============================================================
# Common values on Raspberry Pi:
#   /dev/ttyACM0   (data port, IWR6843)
#   /dev/ttyACM1   (config port, IWR6843)
SERIAL_PORT_DATA = os.getenv("SERIAL_PORT_DATA", "/dev/ttyACM0")
SERIAL_PORT_CFG  = os.getenv("SERIAL_PORT_CFG",  "/dev/ttyACM1")
SERIAL_BAUD      = 921600

# ============================================================
#  Pipeline Settings
# ============================================================
WINDOW_SIZE   = 40    # frames per inference window (matches training)
STRIDE        = 5     # frames to slide window by
SNR_THRESHOLD = 10.0  # minimum SNR to keep a point
FRAME_DT      = 0.055 # seconds between frames (~18 fps)
NUM_FEATURES  = 20    # feature vector size (matches multi_class_model_best.pth)

# ============================================================
#  AWS EC2 API
# ============================================================
# Endpoint for per-frame feature streaming.
# Example: http://<ec2-public-ip>/frame
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "http://43.205.167.81/frame")
DEVICE_ID     = os.getenv("DEVICE_ID",     "rpi-1")
CLOUD_TIMEOUT = float(os.getenv("CLOUD_TIMEOUT", "1.5"))

# ============================================================
#  Class Labels (must match training order)
# ============================================================
CLASSES        = ["NO-FALL", "FALL"]
FALL_CLASS_IDS = {1}
