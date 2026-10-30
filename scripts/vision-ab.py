#!/usr/bin/env python3
"""Vision A/B on the dsv41 lane: does the vision path describe pixels, or confabulate?

Sends one synthetic image with a KNOWN answer to each served name and prints the reply.
Run on spark1. Read-only (generation only).
"""
import base64
import json
import sys
import urllib.request

# 64x64 PNG: left half red, right half green  (unambiguous ground truth)
def make_png():
    import struct
    import zlib

    W = H = 64
    rows = []
    for y in range(H):
        row = b"\x00"
        for x in range(W):
            row += b"\xd0\x20\x20" if x < W // 2 else b"\x20\xb0\x40"
        rows.append(row)
    raw = b"".join(rows)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


PNG_B64 = base64.b64encode(make_png()).decode()

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
MODELS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["vision", "GLM-5.3-Flash-EXL3"]

PROMPT = (
    "Look at this image. It has two solid color halves, left and right. "
    "Name the LEFT half color and the RIGHT half color. Answer with exactly: LEFT=<color> RIGHT=<color>"
)

for m in MODELS:
    body = json.dumps({
        "model": m,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG_B64}},
        ]}],
        "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    try:
        import time
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        msg = r["choices"][0]["message"]
        txt = (msg.get("content") or "").strip()
        print(f"[{m}] {time.time()-t0:.1f}s -> {txt[:200]!r}")
    except Exception as e:
        print(f"[{m}] ERROR: {e}")

print("\nGROUND TRUTH: LEFT=red RIGHT=green")
