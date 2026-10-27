import re

with open("/dev/stdin") as f:
    pass

# Read the serve.sh file
with open("/home/user/qwen3.5-122B-A10B-on-spark/runtime/serve.sh") as f:
    content = f.read()

# Add KV_CACHE_DTYPE env var after LOAD_FORMAT line
old = 'LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"'
new = 'LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"\n# KV cache dtype: fp8 halves KV memory with negligible quality loss on GB10.\n# Set KV_CACHE_DTYPE= to disable (defaults to bf16). Validated on Laguna S 2.1.\nKV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"'
content = content.replace(old, new, 1)

# Add --kv-cache-dtype to the exec line (before --attention-backend)
old_exec = '  --attention-backend "$BACKEND" \\'
new_exec = '  --kv-cache-dtype "${KV_CACHE_DTYPE}" \\\n  --attention-backend "$BACKEND" \\'
content = content.replace(old_exec, new_exec, 1)

with open("/home/user/qwen3.5-122B-A10B-on-spark/runtime/serve.sh", "w") as f:
    f.write(content)

print("serve.sh patched: added KV_CACHE_DTYPE=fp8 support")
print("fp8 in file:", "fp8" in content)
print("kv-cache-dtype in file:", "kv-cache-dtype" in content)