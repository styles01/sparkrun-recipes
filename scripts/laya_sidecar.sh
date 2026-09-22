#!/bin/bash
# Laya sidecar — persistent HTTP service on the Spark (:8299)
# Jev-compatible decision model (convaiinnovations/laya, typed-decisions variant)
# POST /decide  {"state": "...", "questions": {"q1": {"type":"choice","instructions":"...","criteria":{"a":"...","b":"..."}}}}
# -> {"answers": {...calibrated probabilities...}}
# Resident alongside EXL3 lane (3.6GB GPU). Safe: not an OOM-class load.
export HF_HOME=/home/jaita/models/hf
cd /home/jaita/models/hf/convaiinnovations/laya
exec /home/jaita/ComfyUI/venv/bin/python - <<'PYEOF'
import sys, os, json
sys.path.insert(0, "/home/jaita/models/hf/convaiinnovations/laya")
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from rl_agent_api import RLAgent
import torch

agent = RLAgent("/home/jaita/models/hf/convaiinnovations/laya/typed-decisions", device="cuda")

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status":"ok","model":"laya-typed-decisions","device":"cuda"}).encode()
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        if self.path != "/decide":
            self.send_response(404); self.end_headers(); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            res = agent.system_one(req["state"], req.get("questions", {}))
            body = json.dumps(res).encode()
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(500); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)

print("laya-sidecar listening on :8299", flush=True)
ThreadingHTTPServer(("0.0.0.0", 8299), H).serve_forever()
PYEOF