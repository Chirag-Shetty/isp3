import boto3
from pathlib import Path

BUCKET = "radar-frontend-output"
REGION = "ap-south-1"
DIST   = Path("/home/ubuntu/frontend-dist")

# SCP of a directory into an existing directory nests it — auto-detect
if (DIST / "dist").is_dir():
    DIST = DIST / "dist"

MIME   = {
    ".html": "text/html",
    ".js":   "application/javascript",
    ".css":  "text/css",
    ".svg":  "image/svg+xml",
    ".ico":  "image/x-icon",
    ".png":  "image/png",
    ".json": "application/json",
}

s3 = boto3.client("s3", region_name=REGION)

for p in DIST.rglob("*"):
    if not p.is_file():
        continue
    key = p.relative_to(DIST).as_posix()
    ct  = MIME.get(p.suffix, "application/octet-stream")
    cc  = "no-cache" if p.suffix == ".html" else "public, max-age=31536000, immutable"
    s3.upload_file(str(p), BUCKET, key, ExtraArgs={"ContentType": ct, "CacheControl": cc})
    print(f"  ok  {key}")

print("Done! All files uploaded.")
