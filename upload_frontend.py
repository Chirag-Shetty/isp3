"""
upload_frontend.py
------------------
Uploads the Vite dist/ folder to S3 for the radar frontend.
Run from the project root: python upload_frontend.py
"""
import os
import mimetypes
import boto3
from pathlib import Path

BUCKET   = "radar-frontend-output"
REGION   = "ap-south-1"
DIST_DIR = Path(__file__).parent / "frontend" / "dist"

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

    files = list(DIST_DIR.rglob("*"))
    files = [f for f in files if f.is_file()]

    print(f"Uploading {len(files)} files to s3://{BUCKET}/")

    for path in files:
        key = path.relative_to(DIST_DIR).as_posix()
        ext = path.suffix.lower()
        content_type = MIME_MAP.get(ext, "application/octet-stream")

        # HTML files: no cache so updates are instant
        # Assets (hashed filenames): cache 1 year
        cache = "no-cache" if ext == ".html" else "public, max-age=31536000, immutable"

        s3.upload_file(
            str(path), BUCKET, key,
            ExtraArgs={
                "ContentType":  content_type,
                "CacheControl": cache,
            }
        )
        print(f"  ✓ {key}  ({content_type})")

    print(f"\n✅ Done! Open: http://{BUCKET}.s3-website.{REGION}.amazonaws.com/")

if __name__ == "__main__":
    main()
