#!/usr/bin/env python3
"""Smoke-check the deployed service without printing its API key."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path


def request(base: str, key: str, path: str, *, method: str = "GET", data: bytes | None = None,
            content_type: str | None = None) -> tuple[int, dict]:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {key}"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            content = response.read()
            return response.status, json.loads(content) if content else {}
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read())
        except Exception:
            return error.code, {}


def multipart(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = "----attendance" + secrets.token_hex(16)
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode(), b"\r\n",
        ])
    mime = "image/png" if file_path.suffix.lower() == ".png" else "image/webp" if file_path.suffix.lower() == ".webp" else "image/jpeg"
    chunks.extend([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode(),
        f"Content-Type: {mime}\r\n\r\n".encode(), file_path.read_bytes(), b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Render service origin; no path suffix")
    parser.add_argument("--api-key-env", default="BIOMETRIC_API_KEY", help="Environment variable holding the secret")
    parser.add_argument("--collection", default="students")
    parser.add_argument("--image", type=Path, help="Optional image of a consenting test subject for detection/PAD smoke test")
    args = parser.parse_args()
    key = os.getenv(args.api_key_env, "")
    if len(key) < 32:
        print(f"FAIL: environment variable {args.api_key_env} must contain a secret of at least 32 characters", file=sys.stderr)
        return 2
    base = args.base_url.rstrip("/")

    try:
        with urllib.request.urlopen(base + "/healthz", timeout=20) as response:
            public_health = json.loads(response.read())
        if response.status != 200 or public_health.get("ready") is not True:
            raise RuntimeError("public health check is not ready")
        print("PASS: Render health check")

        status, health = request(base, key, "/v1/health")
        if status != 200 or health.get("status") != "ready" or health.get("auth_enabled") is not True:
            raise RuntimeError("authenticated provider health check failed")
        print("PASS: authenticated provider health")

        collection_path = "/v1/collections/" + args.collection
        status, collection = request(base, key, collection_path)
        if status == 404:
            body = json.dumps({"id": args.collection, "name": "Attendance students", "threshold": 0.4}).encode()
            status, collection = request(base, key, "/v1/collections", method="POST", data=body, content_type="application/json")
            if status not in (201, 409):
                raise RuntimeError("collection creation failed")
            status, collection = request(base, key, collection_path)
        if status != 200 or collection.get("collection", {}).get("id") != args.collection:
            raise RuntimeError("collection read-back failed")
        print("PASS: collection create/read")

        if args.image:
            body, content_type = multipart({"max_faces": "2", "collection_id": args.collection}, "image", args.image)
            status, detected = request(base, key, "/v1/detect", method="POST", data=body, content_type=content_type)
            faces = detected.get("faces") if isinstance(detected, dict) else None
            if status != 200 or not isinstance(faces, list) or len(faces) != 1:
                raise RuntimeError("image detection did not return exactly one face")
            liveness = faces[0].get("liveness", {})
            if liveness.get("status") != "ok" or liveness.get("is_live") is not True:
                raise RuntimeError("passive liveness was negative or unavailable; service correctly failed closed")
            landmarks = faces[0].get("landmarks")
            if not isinstance(landmarks, list) or len(landmarks) < 5:
                raise RuntimeError("five-point landmarks were not returned")
            print("PASS: one-face detect, explicit positive passive-liveness result, landmarks")
        else:
            print("SKIP: image/PAD smoke (provide --image from a consenting test subject)")
        return 0
    except Exception as exc:
        print("FAIL:", str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
