#!/usr/bin/env bash
# SMOKE on cpu-basic (about $0.01): no GPU, no model. Proves what a GPU job would otherwise find
# out five dollars later: the image boots, the bucket mount and the code copy are readable, the
# stage scripts parse, the exllamav3 release wheel exists for this image's torch minor and python,
# the held-out texts are in the bucket, and the token reaches the Hub.
set -euo pipefail
cd /work/exl3/code
ls -la
bash -n common.sh prep.sh measure.sh convert.sh smoke.sh
python3 -m py_compile pack_rows.py qbench_project.py summarize.py spend.py
python3 - <<'PY'
import sys, torch, urllib.request
v = torch.__version__.split("+")[0].split(".")
tag = f"cp{sys.version_info[0]}{sys.version_info[1]}"
url = (f"https://github.com/turboderp-org/exllamav3/releases/download/v1.6.0/"
       f"exllamav3-1.6.0+cu128.torch{v[0]}.{v[1]}.0-{tag}-{tag}-linux_x86_64.whl")
print("torch", torch.__version__, "python", sys.version.split()[0], "tag", tag)
r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
print("wheel", r.status, url)
PY
ls -la /work/exl3/texts
pip install -q --no-input 'huggingface_hub[hf_xet]>=1.3' >/dev/null 2>&1 || pip install -q 'huggingface_hub[hf_xet]>=1.3'
hf auth whoami
hf buckets ls 0xBakeer/qwen38-engine-work/nvfp4-bf16/calib | grep -E "sens-(code|prose)"
echo SMOKE-OK
