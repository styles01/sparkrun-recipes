import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Where SAMPLE_NUCLEUS's row_draws spends its time at 16 / 32 rows (no model): CUDA time by kernel."""
import torch
from engine import sample as SM
from engine.sample import Sampler
from torch.profiler import profile, ProfilerActivity
SM.BATCH, SM.TOPP_K, SM.NUCLEUS = True, 1024, True
V = 248320
for n in (16, 32):
    x = torch.randn(n, V, device="cuda"); x[:, :40] += torch.linspace(16, 10, 40, device="cuda")
    s = Sampler(0.7, 0.95)
    for _ in range(5): s.row_draws(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(20): s.row_draws(x)
        torch.cuda.synchronize()
    rows = sorted(((e.key, (getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)) / 20e3) for e in p.key_averages()), key=lambda r: -r[1])
    print(f"[sprof] rows {n}: total {sum(r[1] for r in rows):.3f} ms; " + "; ".join(f"{k[:60]} {v:.3f}" for k, v in rows[:8]), flush=True)
