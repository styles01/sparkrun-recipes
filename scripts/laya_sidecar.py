#!/usr/bin/env python3
"""Laya System-1 gate sidecar — fp16, own venv, zero ComfyUI entanglement.
Jev-shaped: POST /decide {state, questions} -> calibrated typed answers.
Routes: GET /health, GET /metrics, POST /decide. stdlib-only HTTP."""
import sys, os, json, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, "/home/jaita/models/hf/convaiinnovations/laya")
os.environ.setdefault("HF_HOME", "/home/jaita/models/hf")
import torch
from rl_agent_api import RLAgent

CKPT = "/home/jaita/models/hf/convaiinnovations/laya/typed-decisions"
_t0 = time.time()
agent = RLAgent(CKPT, device="cuda")
# fp16 conversion (Phase 0 core change from the 3.3GB fp32 era)
try:
    agent.model = agent.model.half()
except Exception as e:
    print("fp16 convert failed:", e, file=sys.stderr)
_load_s = time.time() - _t0
_stats = {"calls": 0, "lat_ms_sum": 0.0, "lat_max": 0.0, "err": 0}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status":"ok","model":"laya-typed-decisions","precision":"fp16","device":"cuda","load_s":round(_load_s,1)})
        elif self.path == "/metrics":
            with torch.inference_mode():
                mem = torch.cuda.max_memory_allocated()/2**30
            n = max(_stats["calls"],1)
            self._send(200, {"calls":_stats["calls"],"p_avg_ms":round(_stats["lat_ms_sum"]/n,1),
                             "p_max_ms":round(_stats["lat_max"],1),"errors":_stats["err"],
                             "gpu_peak_gib":round(mem,2)})
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        if self.path != "/decide":
            self.send_response(404); self.end_headers(); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            t = time.time()
            with torch.inference_mode():
                res = agent.system_one(req["state"], req.get("questions", {}))
            dt = (time.time()-t)*1000
            _stats["calls"] += 1; _stats["lat_ms_sum"] += dt; _stats["lat_max"] = max(_stats["lat_max"], dt)
            self._send(200, res)
        except Exception as e:
            _stats["err"] += 1
            self._send(500, {"error": str(e)})

if __name__ == "__main__":
    print(f"laya-sidecar fp16 listening on :8299 (load {_load_s:.1f}s)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8299), H).serve_forever()
