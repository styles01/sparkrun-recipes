"""The disk tier's writer against CUDA graph captures, on the GPU (a hold step; seconds, < 1 GB).

    python tools/capture_race_check.py [--captures 60] [--dir ~/.qwen38-spark-engine/race-check]

On 2026-10-02 at 21:09 the writer copied a session head's rows while the drafter
captured a graph: the copy failed with `cudaErrorStreamCaptureUnsupported`, the capture with it, and
the request ended in an error (engine/capture_lock.py). This runs the real `DiskTier` writer over
CUDA buffers shaped like the engine's (a fake engine: recurrent state, convolution tails, KV rows),
one job after another, while the main thread captures and replays small CUDA graphs the way
engine/drafters/draft_graph.py does, holding `capture_lock.LOCK` around each capture.

  * `--lock on` (what ships): no capture fails, no writer error, every job's chunks written.
  * `--lock off` (the control): the same with the lock taken away from both sides; the race is
    expected to show (capture or writer errors). It runs in a process of its own, because a
    failed capture can leave the CUDA context unusable.

Without `--lock`, both run, each in a subprocess, and the exit is 0 when `on` is clean; `off`
showing no error is reported as such (the overlap is timing-dependent), not a failure. A
temporary directory under `--dir` holds the files and is removed. Needs the GPU: in a hold.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def acquire(self, *a, **k):
        return True

    def release(self):
        pass


def one(lock: bool, captures: int, root: str, dev: str = "cuda") -> dict:
    import torch
    from engine import capture_lock, disk_cache
    from engine.cache import prefix_hashes
    if not lock:
        capture_lock.LOCK = _NoLock()             # the control: neither side holds it
    C, L, n_jobs = 1024, 4, 0
    # the engine's shapes, smaller: rows of 16 KiB a token (the served stack has 104 KiB)
    eng = types.SimpleNamespace(
        state=types.SimpleNamespace(S=torch.randn(8, 1, 8, 64, 64, device=dev),
                                    conv=torch.randn(8, 1, 1024, 4, device=dev)),
        kv=types.SimpleNamespace(k=torch.randn(4, 1, 2, 16384, 128, dtype=torch.bfloat16,
                                               device=dev),
                                 v=torch.randn(4, 1, 2, 16384, 128, dtype=torch.bfloat16,
                                               device=dev), fp8=False, length=0))
    d = tempfile.mkdtemp(prefix="race-", dir=root)
    errors, writer_errors = [], []
    try:
        t = disk_cache.DiskTier(d, chunk=C, fingerprint="race", min_tokens=C, head_anchors=4,
                                settings={"admit_after": 1, "budget_gb": 20},
                                on_event=lambda k, det: writer_errors.append(det)
                                if k == "error" else None).open()
        stop = threading.Event()

        def jobs():
            nonlocal n_jobs
            seed = 0
            while not stop.is_set():
                seed += 1
                g = torch.Generator().manual_seed(seed)
                keys = torch.randint(1, 50000, (L * C + 7,), generator=g).tolist()
                hashes = prefix_hashes(keys)
                # the engine thread's part (the plan, the anchors' staging copies, the seal):
                # in the engine it never overlaps that thread's own captures, so here it holds
                # the lock as one (the control takes it away from this side too)
                with capture_lock.LOCK:
                    job = t.plan(keys, hashes, 0, len(keys), len(keys), None, f"s{seed}",
                                 {"client": "opencode", "head_end": L * C})
                    if job is None:
                        continue
                    for line in sorted(job.lines):
                        t.capture(job, eng, line)
                    t.seal(job, eng, None, len(keys))
                t.flush(timeout=60)
                n_jobs += 1

        feeder = threading.Thread(target=jobs, daemon=True)
        feeder.start()
        x = torch.randn(256, 256, device=dev)
        w = torch.randn(256, 256, device=dev)
        if dev != "cuda":
            # the CPU smoke (preflight): the writer's jobs alone, no graph
            for i in range(captures):
                torch.relu(x @ w)
            stop.wait(2.0)
            captures = 0
        s = torch.cuda.Stream() if dev == "cuda" else None
        for i in range(captures):
            try:
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    torch.relu(x @ w)                 # the eager run before a capture
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with capture_lock.LOCK, torch.cuda.graph(g, stream=s):
                    y = torch.relu(x @ w) + i
                g.replay()
                torch.cuda.synchronize()
                if not torch.allclose(y, torch.relu(x @ w) + i):
                    errors.append(f"capture {i}: wrong replay")
            except Exception as exc:                                   # noqa: BLE001
                errors.append(f"capture {i}: {type(exc).__name__}: {str(exc)[:160]}")
                if len(errors) > 3:
                    break
        stop.set()
        feeder.join(timeout=120)
        rep = t.report()
        incomplete = [r["tokens"] for r in rep["top"] if not r["complete"]]
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return {"lock": lock, "captures": captures, "jobs": n_jobs, "capture_errors": errors,
            "writer_errors": [str(e.get("error"))[:160] for e in writer_errors],
            "incomplete": incomplete}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", choices=("on", "off"))
    ap.add_argument("--captures", type=int, default=60)
    ap.add_argument("--dir", default="~/.qwen38-spark-engine/race-check")
    ap.add_argument("--device", default="cuda", help="cpu: a smoke run of the writer's jobs alone "
                    "(the preflight's), no capture")
    a = ap.parse_args(argv)
    root = os.path.expanduser(a.dir)
    os.makedirs(root, mode=0o700, exist_ok=True)
    if a.lock is not None:
        print(json.dumps(one(a.lock == "on", a.captures, root, a.device)), flush=True)
        return 0
    got = {}
    for mode in ("on", "off"):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--lock", mode,
                            "--captures", str(a.captures), "--dir", root, "--device", a.device],
                           capture_output=True, text=True, timeout=600, cwd=ROOT)
        last = (p.stdout.strip().splitlines() or ["{}"])[-1]
        try:
            got[mode] = json.loads(last)
        except ValueError:
            got[mode] = {"crashed": p.returncode, "stderr": p.stderr[-300:]}
    on, off = got["on"], got["off"]
    clean = ("crashed" not in on and not on["capture_errors"] and not on["writer_errors"]
             and not on["incomplete"] and on["jobs"] > 0)
    print(f"[race] lock on: {on.get('captures')} captures, {on.get('jobs')} writer jobs, "
          f"capture errors {len(on.get('capture_errors', ['?']))}, writer errors "
          f"{len(on.get('writer_errors', ['?']))}, incomplete entries {on.get('incomplete')}"
          + (f" {on.get('capture_errors')[:1]} {on.get('writer_errors')[:1]}" if not clean else ""))
    seen = ("crashed" in off or off.get("capture_errors") or off.get("writer_errors")
            or off.get("incomplete"))
    print(f"[race] lock off (the control): "
          + ("the race shows: " + json.dumps({k: off.get(k) for k in
                                               ("crashed", "capture_errors", "writer_errors",
                                                "incomplete")})[:400]
             if seen else f"no error in {off.get('captures')} captures and {off.get('jobs')} "
                          f"jobs (timing-dependent; not a failure)"))
    print(f"[race] {'PASS' if clean else 'FAIL'}")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
