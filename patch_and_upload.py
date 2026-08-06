"""
Uploads the new frontend dist to S3.
Run on EC2 where IAM credentials are available.
"""
import os
import mimetypes
import boto3
from pathlib import Path

BUCKET   = "radar-frontend-output"
REGION   = "ap-south-1"
DIST_DIR = Path("/home/ubuntu/frontend-dist-new")

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

def main():
    s3 = boto3.client("s3", region_name=REGION)

    files = [f for f in DIST_DIR.rglob("*") if f.is_file()]
    print(f"Uploading {len(files)} files to s3://{BUCKET}/")

    for path in files:
        key = path.relative_to(DIST_DIR).as_posix()
        ext = path.suffix.lower()
        content_type = MIME_MAP.get(ext, "application/octet-stream")
        cache = "no-cache" if ext == ".html" else "public, max-age=31536000, immutable"

        s3.upload_file(
            str(path), BUCKET, key,
            ExtraArgs={"ContentType": content_type, "CacheControl": cache},
        )
        print(f"  ok  {key}  ({content_type})")

    print(f"\nDone! http://{BUCKET}.s3-website.{REGION}.amazonaws.com/")

if __name__ == "__main__":
    main()
