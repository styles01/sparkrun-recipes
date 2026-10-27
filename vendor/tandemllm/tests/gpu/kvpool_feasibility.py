"""GB10 feasibility of the VMM-backed KV pool (engine/kvpool.py): granularity, a torch view over a
reserved range, commit/zero/trim, MemAvailable accounting, CUDA-graph replay across a later
commit, attention over a paged vs static buffer bit-identical. Small (<= 4 GiB committed)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from engine import kvpool as kp

def avail():
    return kp._meminfo_avail() / kp.GiB

torch.cuda.init(); torch.zeros(1, device="cuda")
drv = kp.Driver.get()
print("driver", drv is not None, "granularity", drv.granularity if drv else None)
pool = kp.KVPool(hard_floor=12 * kp.GiB, margin_rows=0)
shape = (16, 1, 4, 65536, 256)          # engine KV layout, 64k window: 2 GiB virtual
a0 = avail()
b = kp.PagedBuffer(shape, 3, torch.bfloat16, "cuda", owner="t", pool=pool)
print("reserved", b.nbytes / kp.GiB, "GiB virtual; committed", b.committed_bytes(), "avail", round(avail(), 2), "(was", round(a0, 2), ")")
print("tensor", b.tensor.shape, b.tensor.stride(), b.tensor.dtype, hex(b.tensor.data_ptr()))
t0 = time.perf_counter(); n = b.ensure(3000); t1 = time.perf_counter()
print("ensure 3000 rows:", n / kp.MiB, "MiB", f"{(t1-t0)*1e3:.1f} ms", "rows", b.rows, "avail", round(avail(), 2))
assert float(b.tensor[:, :, :, :b.rows].abs().sum()) == 0.0, "fresh pages not zero"
static = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
x = torch.randn(16, 1, 4, 3000, 256, device="cuda", dtype=torch.bfloat16)
b.tensor[:, :, :, :3000] = x; static[:, :, :, :3000] = x
q = torch.randn(1, 24, 1, 256, device="cuda", dtype=torch.bfloat16)
def attn(buf, L):
    k = buf[5, :, :, :L]; v = buf[6, :, :, :L]
    return torch.nn.functional.scaled_dot_product_attention(q, k.repeat_interleave(6, 1), v.repeat_interleave(6, 1))
print("attention paged==static:", torch.equal(attn(b.tensor, 3000), attn(static, 3000)))
# graph captured while 4096 rows committed, replayed after growing to 40000 rows
L_dev = torch.tensor([3000], device="cuda")
out = torch.empty(1, device="cuda")
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
def body():
    out.copy_(b.tensor[3, 0, 1, :40000].float().sum().reshape(1))
with torch.cuda.stream(s):
    b.ensure(40000); body()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    body()
b.tensor[3, 0, 1, 39000:40000] = 1.0
g.replay(); torch.cuda.synchronize(); print("graph replay sum after write (expect >= 1000*256):", float(out))
# trim then regrow: pages new and zero, graph still replays on the same addresses
rel = b.trim(4096); print("trim to 4096: released", rel / kp.MiB, "MiB rows", b.rows, "avail", round(avail(), 2))
b.ensure(40000)
g.replay(); torch.cuda.synchronize(); print("replay after regrow (expect 0 + rows<4096 part):", float(out), "ref", float(b.tensor[3,0,1,:40000].float().sum()))
# big commit timing + MemAvailable accounting
a1 = avail(); t0 = time.perf_counter(); n = b.ensure(65536); torch.cuda.synchronize(); t1 = time.perf_counter()
a2 = avail()
print(f"ensure full 65536: +{n/kp.GiB:.3f} GiB in {(t1-t0)*1e3:.0f} ms; MemAvailable {a1:.2f} -> {a2:.2f} (delta {a1-a2:.2f})")
print("torch allocated/reserved GiB", torch.cuda.memory_allocated()/kp.GiB, torch.cuda.memory_reserved()/kp.GiB)
t0 = time.perf_counter(); rel = b.trim(0); t1 = time.perf_counter(); a3 = avail()
print(f"trim 0: -{rel/kp.GiB:.3f} GiB in {(t1-t0)*1e3:.0f} ms; MemAvailable {a3:.2f}")
print("pool", pool.report())
b.close(); print("closed; avail", round(avail(), 2))
