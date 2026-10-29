#!/usr/bin/env bash
# Muse decode probe: real generations via the consumer door, thinking off, fixed workload.
set -uo pipefail
H=http://192.168.2.14:8888
echo "=== $(date -u) muse decode probe ==="
echo "--- models ---"
curl -s -m10 "$H/v1/models" | head -c 400; echo
echo "--- probe A: thinking off, 200 tok free decode ---"
time curl -s -m300 "$H/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"Muse-Spark-Open","messages":[{"role":"user","content":"Write 200 words of plain prose about the sea."}],"max_tokens":260,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c "import sys,json;r=json.load(sys.stdin);c=r['choices'][0];print('finish',c.get('finish_reason'),'usage',r.get('usage'));print(repr(c['message'].get('content','')[:160]))"
echo "--- probe B: greedy fixed 100 tok ---"
time curl -s -m300 "$H/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"Muse-Spark-Open","messages":[{"role":"user","content":"Count slowly from 1 to 100 in words, one per line."}],"max_tokens":128,"temperature":0,"ignore_eos":true}' \
  | python3 -c "import sys,json;r=json.load(sys.stdin);print('usage',r.get('usage'),'finish',r['choices'][0].get('finish_reason'))"
echo "--- server-side rows ---"
docker logs muse-zig 2>&1 | grep "done chatcmpl" | tail -6
