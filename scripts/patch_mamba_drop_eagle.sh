#!/bin/bash
# patch_mamba_drop_eagle.sh — Apply PR #48375 fix to vLLM MambaManager
# Fixes: MTP + prefix caching on hybrid Mamba models corrupts recurrent state
set -e
FILE="/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py"

if grep -q "drop_eagle_block and max_num_blocks" "$FILE" 2>/dev/null; then
    echo "[patch] MambaManager drop_eagle_block fix already applied"
    exit 0
fi

python3 << 'PYEOF'
path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py"
with open(path) as f:
    src = f.read()

old = """        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found."""

new = """        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        if drop_eagle_block and max_num_blocks > 0:
            # EAGLE/MTP: drop the final matched page. Its recurrent-state snapshot
            # may have been taken over draft tokens that verification later rejects,
            # so the state must be restored from the previous page boundary and
            # recomputed forward. FullAttentionManager (see find_longest_cache_hit
            # above) pops the last block; Mamba keeps only the rightmost real block
            # with the prefix null-padded, so a literal pop would delete the state
            # block itself -- lower the search ceiling by one page instead.
            max_num_blocks -= 1
        # Search from right to left and early stop when a match is found."""

if old in src:
    src = src.replace(old, new)
    with open(path, 'w') as f:
        f.write(src)
    print('[patch] MambaManager drop_eagle_block fix applied successfully')
elif 'drop_eagle_block and max_num_blocks' in src:
    print('[patch] Already patched')
else:
    print('[patch] WARNING: Could not find pattern to patch')
PYEOF

RESULT=$?
if [ $RESULT -ne 0 ]; then
    echo "[patch] Python script failed"
    exit 1
fi