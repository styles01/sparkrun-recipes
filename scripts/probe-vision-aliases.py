#!/usr/bin/env python3
"""Prove what each served alias on dsv41 actually does with an image."""
import base64
import json
import struct
import urllib.request
import zlib


def make_png():
    W = H = 64
    rows = []
    for y in range(H):
        r = b"\x00"
        for x in range(W):
            r += b"\xd0\x20\x20" if x < W // 2 else b"\x20\xb0\x40"
        rows.append(r)
    raw = b"".join(rows)

    def ch(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    h = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", h) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")


B64 = base64.b64encode(make_png()).decode()
PROMPT = ("This image has two solid color halves. Name them exactly. "
          "Answer in the form: LEFT=<color> RIGHT=<color>")

for model in ["vision", "GLM-5.3-Flash-EXL3", "dsv41", "DeepSeek-4.1-Flash-Ablit"]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + B64}},
        ]}],
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=180).read())
        txt = (r["choices"][0]["message"].get("content") or "").strip()
        print(f"{model:28} -> {txt[:100]!r}")
    except Exception as e:
        body_txt = getattr(e, "read", lambda: b"")()[:200]
        print(f"{model:28} -> ERROR {e} {body_txt!r}")

print("\nGROUND TRUTH: LEFT=red RIGHT=green")
