"""
Wipes the S3 bucket and re-uploads only the correct dist files.
Run on EC2: python3 ~/wipe_and_upload.py
"""
import boto3
from pathlib import Path

BUCKET   = "radar-frontend-output"
REGION   = "ap-south-1"
# The new build landed at frontend-dist-new/dist/ due to SCP nesting
DIST_DIR = Path("/home/ubuntu/frontend-dist-new/dist")

MIME_MAP = {
    ".html": "text/html",
    ".js":   "application/javascript",
    ".css":  "text/css",
    ".svg":  "image/svg+xml",
    ".ico":  "image/x-icon",
    ".png":  "image/png",
    ".json": "application/json",
    ".txt":  "text/plain",
}

s3_client   = boto3.client("s3", region_name=REGION)
s3_resource = boto3.resource("s3", region_name=REGION)
bucket      = s3_resource.Bucket(BUCKET)

# ── 1. Delete every object in the bucket ──────────────────────────────────────
print("Deleting all existing objects...")
deleted = list(bucket.objects.all())
if deleted:
    bucket.delete_objects(Delete={"Objects": [{"Key": o.key} for o in deleted]})
    print(f"  Deleted {len(deleted)} objects.")
else:
    print("  Bucket already empty.")

# ── 2. Upload fresh dist ───────────────────────────────────────────────────────
files = [f for f in DIST_DIR.rglob("*") if f.is_file()]
print(f"\nUploading {len(files)} files to s3://{BUCKET}/")

for path in files:
    key  = path.relative_to(DIST_DIR).as_posix()
    ext  = path.suffix.lower()
    ct   = MIME_MAP.get(ext, "application/octet-stream")
    cc   = "no-cache" if ext == ".html" else "public, max-age=31536000, immutable"
    s3_client.upload_file(
        str(path), BUCKET, key,
        ExtraArgs={"ContentType": ct, "CacheControl": cc},
    )
    print(f"  ok  {key}")

print(f"\nDone! http://{BUCKET}.s3-website.{REGION}.amazonaws.com/")
