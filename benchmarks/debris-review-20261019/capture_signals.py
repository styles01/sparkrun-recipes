import json, time, urllib.request, sys

BASE = "http://127.0.0.1:8000"
OUT = open("/tmp/capture_40k.jsonl", "w")

def get_metrics():
    r = urllib.request.urlopen(BASE + "/metrics", timeout=5)
    txt = r.read().decode()
    vals = {}
    for line in txt.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc{"):
            vals["kv"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:prompt_tokens_total{"):
            vals["prompt"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:generation_tokens_total{"):
            vals["gen"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_running{"):
            vals["run"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_waiting{"):
            vals["wait"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:iteration_tokens_total_sum{"):
            vals["iter"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:time_to_first_token_seconds_sum{"):
            vals["ttft_s"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:time_to_first_token_seconds_count{"):
            vals["ttft_c"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:prefix_cache_hits_total{"):
            vals["pfhits"] = float(line.rsplit(" ", 1)[1])
    return vals

def chat(prompt, max_tokens=8):
    body = json.dumps({"model": "GLM-5.3-Flash-EXL3",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "stream": False}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=120)

# ~40K-token prompt (repeated paragraph, ~4.1 chars/tok)
para = ("The quick brown fox jumps over the lazy dog near the riverbank while "
        "the sun sets slowly behind the distant mountains, painting the sky in "
        "shades of orange and pink that fade into the deep blue of the coming night. ")
prompt = para * 950  # ~40K tokens

phase = sys.argv[1] if len(sys.argv) > 1 else "single"

thread_holder = {}
def fire():
    t0 = time.time()
    r = chat(prompt)
    dt = time.time() - t0
    OUT.write(json.dumps({"event": "done", "wall": round(dt, 2)}) + "\n")

import threading
if phase == "single":
    th = threading.Thread(target=fire); th.start()
else:
    ths = [threading.Thread(target=fire) for _ in range(4)]
    [t.start() for t in ths]

t_end = time.time() + 34
i = 0
prev = {}
while time.time() < t_end:
    m = get_metrics()
    m["t"] = round(time.time(), 2)
    if prev:
        m["d_prompt"] = round(m.get("prompt", 0) - prev.get("prompt", 0), 1)
        m["d_kv"] = round(100 * (m.get("kv", 0) - prev.get("kv", 0)), 3)
        m["d_iter"] = round(m.get("iter", 0) - prev.get("iter", 0), 1)
    OUT.write(json.dumps(m) + "\n"); OUT.flush()
    prev = m
    time.sleep(1.0)
OUT.write(json.dumps({"event": "capture_end"}) + "\n")
OUT.close()
print("capture complete -> /tmp/capture_40k.jsonl")