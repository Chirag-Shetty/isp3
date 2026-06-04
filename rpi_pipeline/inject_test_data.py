"""
inject_test_data.py
-------------------
Injects fake radar prediction rows directly into DynamoDB.
Use this to test the frontend without needing the RPi or EC2 API running.

Usage:
    pip install boto3
    python inject_test_data.py

Set your AWS credentials first (one of):
    - AWS env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    - Or run: aws configure
"""

import boto3
import random
import time
from datetime import datetime, timezone, timedelta

# ── Config — change these to match your setup ───────────────────────────────
TABLE_NAME = "radar_predictions"   # your DynamoDB table name
REGION     = "ap-south-1"          # your AWS region (Mumbai)
DEVICE_ID  = "rpi-1"
NUM_ROWS   = 20                     # how many fake rows to insert
# ────────────────────────────────────────────────────────────────────────────

CLASSES = ["NO-FALL", "FALL"]

def fake_row(i: int) -> dict:
    """Generate one realistic-looking prediction row."""
    ts = (datetime.now(timezone.utc) - timedelta(seconds=i * 3)).isoformat()
    is_fall = random.random() < 0.15          # ~15% fall rate
    class_id = 1 if is_fall else 0
    class_name = CLASSES[class_id]

    conf = random.uniform(0.70, 0.99)
    other = 1.0 - conf
    probs = [other, conf] if is_fall else [conf, other]

    return {
        "device_id":    DEVICE_ID,
        "timestamp":    ts,
        "class_id":     class_id,
        "class_name":   class_name,
        "confidence":   str(round(conf, 4)),   # DynamoDB stores as string for Decimal safety
        "is_fall":      is_fall,
        "probs":        [str(round(p, 4)) for p in probs],
        "frame_count":  (i + 1) * 40,
        "window_index": i + 1,
        "n_points":     random.randint(5, 30),
        "x_mean":       str(round(random.uniform(-1.0, 1.0), 4)),
        "y_mean":       str(round(random.uniform(0.5, 3.0), 4)),
        "z_mean":       str(round(random.uniform(0.0, 1.5), 4)),
        "height_range": str(round(random.uniform(0.1, 1.8), 4)),
        "source":       "inject_test",
    }


def main():
    print(f"Connecting to DynamoDB table '{TABLE_NAME}' in {REGION}...")
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(TABLE_NAME)

    # Check table exists
    try:
        status = table.table_status
        print(f"Table status: {status}")
    except Exception as e:
        print(f"ERROR: Could not access table '{TABLE_NAME}': {e}")
        print("Check your TABLE_NAME, REGION, and AWS credentials.")
        return

    print(f"\nInserting {NUM_ROWS} fake rows...")
    for i in range(NUM_ROWS):
        row = fake_row(i)
        table.put_item(Item=row)
        label = "⚠ FALL" if row["is_fall"] else "  safe"
        print(f"  [{i+1:02d}] {label}  conf={row['confidence']}  ts={row['timestamp'][11:19]}")
        time.sleep(0.05)

    print(f"\n✅ Done! {NUM_ROWS} rows inserted into '{TABLE_NAME}'.")
    print("Now open your frontend — data should appear.")


if __name__ == "__main__":
    main()
