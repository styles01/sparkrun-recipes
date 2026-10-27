import json, time, urllib.request, sys, threading

BASE = "http://127.0.0.1:8000"
OUT = open("/tmp/capture_decode.jsonl", "w")

def chat(prompt, max_tokens, out):
    body = json.dumps({"model": "GLM-5.3-Flash-EXL3",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "stream": False}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=240)
    data = json.loads(r.read())
    dt = time.time() - t0
    usage = data.get("usage", {})
    ct = usage.get("completion_tokens", 0)
    out.append((dt, ct))

outs = []
def run(max_tokens, n=1):
    outs.clear()
    ths = [threading.Thread(target=chat, args=("Count quietly from one to twenty in words.", max_tokens, outs)) for _ in range(n)]
    for t in ths: t.start()
    for t in ths: t.join()
    return outs

walls = run(300, 1)
print("1x decode:", walls, "tok/s", round(walls[0][1]/walls[0][0], 2))
time.sleep(3)
walls4 = run(300, 4)
for dt, ct in walls4:
    print("4x lane:", round(dt,2), "s", ct, "tok ->", round(ct/dt,2), "tok/s each; agg", round(sum(c for _,c in walls4)/max(d for d,_ in walls4),2))