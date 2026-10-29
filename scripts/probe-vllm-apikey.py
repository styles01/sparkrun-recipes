#!/usr/bin/env python3
"""Prove vLLM's --api-key behaviour: does it validate keys, or just require one?
Run against a live vLLM lane. Read-only: sends tiny requests, changes nothing."""
import json, sys, urllib.request, urllib.error

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "dsv41"

def ask(key=None, label=""):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }).encode()
    h = {"Content-Type": "application/json"}
    if key is not None:
        h["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body, headers=h)
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())
        return f"ACCEPTED  {label}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} {label} :: {e.read().decode()[:120]}"
    except Exception as e:
        return f"ERR {label} :: {e}"

print("=== vLLM API-key validation probe ===")
print("1. no key at all          :", ask(None, "no header"))
print("2. clearly bogus key      :", ask("sk-totally-made-up-key-12345", "bogus"))
print("3. another bogus key      :", ask("hermes-profile:loca", "profile-style"))
print("4. empty bearer           :", ask("", "empty"))
