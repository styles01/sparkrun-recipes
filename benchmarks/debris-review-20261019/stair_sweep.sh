#!/usr/bin/env bash
set -u
CACHE="$HOME/.cache/vllm/glm53_adaptive_k.json"
declare -a KS=(2 4 7)
for K in "${KS[@]}"; do
  echo "{\"mode\":\"ema\",\"set\":[$K],\"alpha\":0.25,\"margin\":1.0,\"min_steps\":0,\"saturate\":\"max\"}" > "$CACHE"
  sleep 3
  echo "=== pinned set=[$K] at $(date +%H:%M:%S)"
  TAG="k$K" python3 - <<'PYEOF'
import json, time, threading, urllib.request, statistics, sys
import os
BASE="http://127.0.0.1:8000"; MODEL="GLM-5.3-Flash-EXL3"; TAGV=os.environ.get("TAG","k?")
def metrics():
    try: return urllib.request.urlopen(BASE+"/metrics",timeout=5).read().decode()
    except Exception: return ""
def grab(m,n):
    v=0.0
    for L in m.splitlines():
        if L.startswith(n) and not L.startswith("#"): v+=float(L.rsplit(" ",1)[1])
    return v
def one(prompt,mt,box):
    payload=json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"stream":True,"stream_options":{"include_usage":True}}).encode()
    rq=urllib.request.Request(BASE+"/v1/chat/completions",payload,{"Content-Type":"application/json"})
    t0=time.time(); first=None; usage=None
    try:
        with urllib.request.urlopen(rq,timeout=550) as r:
            for line in r:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    try:
                        d=json.loads(line[6:])
                        if d.get("usage"): usage=d["usage"]
                        dl=((d.get("choices") or [{}])[0].get("delta") or {})
                        if dl.get("content") and first is None: first=time.time()-t0
                    except Exception: pass
    except Exception as e:
        box["err"]=str(e)[:80]; return
    dt=time.time()-t0-(first or 0)
    if usage: box.update(tps=usage["completion_tokens"]/dt if dt>0 else 0)
import os
def case(tag,prompt,conc,runs):
    m0=metrics(); boxes=[{} for _ in range(conc*runs*0+conc)]; allres=[]
    for rr in range(runs):
        th=threading.Thread(target=one,args=(prompt,256,boxes[rr%conc]))
        pass
    # simpler: each run = conc threads, fresh boxes
    for rr in range(runs):
        bx=[{} for _ in range(conc)]
        ts=[threading.Thread(target=one,args=(prompt,256,bx[i])) for i in range(conc)]
        for t in ts: t.start()
        for t in ts: t.join()
        allres.extend(b for b in bx if "tps" in b)
    m1=metrics()
    d=grab(m1,"vllm:spec_decode_num_draft_tokens_total")-grab(m0,"vllm:spec_decode_num_draft_tokens_total")
    a=grab(m1,"vllm:spec_decode_num_accepted_tokens_total")-grab(m0,"vllm:spec_decode_num_accepted_tokens_total")
    tps=[b["tps"] for b in allres]; med=statistics.median(tps) if tps else 0
    line=f"{TAGV}-{tag}: streams={len(tps)} median={med:.2f} tok/s min={min(tps):.2f} max={max(tps):.2f} acc%={a/max(d,1)*100:.1f}"
    print(line)
    open("/tmp/stair-sweep.jsonl","a").write(line+"\n")
    sys.stdout.flush()
SHORT="Write a plain essay about the history of lighthouses. "*4
LONG="Review this Python module for edge cases and output one concise note per issue:\n\n"+("    x = compute(i)  # branch\n\n"*2500)
case("short",SHORT,1,5)
case("long40k",LONG,1,5)
case("4x-short",SHORT,4,5)
case("4x-long40k",LONG,4,5)
PYEOF
done
echo "SWEEP-DONE"

