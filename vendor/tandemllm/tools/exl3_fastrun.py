"""The fast run of one weight configuration: prose and chat decode, prefill, tokens a round.

    python tools/exl3_fastrun.py --label exl3-6.0 --repo ~/exl3k/repo --env-file ops/serve.env \
        --env QWEN38_EXL3=~/exl3-quants/6.0 --port 8011 --json out.json
    python tools/exl3_fastrun.py --label nvfp4-live --repo ~/tandem-staircut --env-file ... --json ..

One server (the raw engine: no conductor, no prefix/session caches, the persistent suffix store OFF,
`--drop-idle`, the serve.env QWEN38_* knobs, exactly row3's server command), then, one request at a
time, streamed, temperature 0, thinking off:

  * prose: tools/bench_decode.py's `prose` prompt, `--reps` times, `--new` tokens
  * chat:  tools/bench_decode.py's `chat` prompt, likewise
  * prefill: held-out text cut to exactly `--prefill` tokens each (bench/heldout_*.txt, then
    bench/calib.txt), max_tokens 1: TTFT and prefill tok/s = prompt tokens / TTFT

decode tok/s = (completion - 1) / (e2e - ttft) on the client clock (row3's formula); tokens a round
and the first-miss histogram from the server's own `[req]` line (tools/rowlog.py). Every knob the
server ran with is recorded in the report.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import rowlog  # noqa: E402

HOME = Path(os.path.expanduser("~"))
PROSE = ("Write three paragraphs about why the memory system, and not the arithmetic units, "
         "sets the speed of a language model that generates one token at a time.")
# block_budget's code request (its answer is not in its prompt): the class the wide tree pays on
CODE = ("Write a Python function that parses an ISO-8601 duration such as P3DT4H5M6S into a "
        "number of seconds. Validate the input, raise ValueError with a useful message, and "
        "include three doctest examples.")
CHAT = ("I have a machine with 121 GB of unified memory and about 273 GB/s of bandwidth. "
        "Explain in plain language what that means for running a 27-billion-parameter model.")


NO_OVERLAY = "/nonexistent/qse-setup-env"


def env_from_file(path: str) -> dict[str, str]:
    """The QWEN38_* knobs (and PY/REPO-independent values) a serve.env sets, as start.sh exports
    them: sourced by bash so its ${HOME}/${REPO} expansions are the file's own. The setup guide's
    overlay (run/setup.env, which serve.env reads last) is left out: a benchmark row is the file's
    numbers, not whatever the box was last set up with (`setup_overlay` in the report says when
    one was there)."""
    out = subprocess.run(["bash", "-c", f"set -a; . {shlex.quote(path)} >/dev/null 2>&1; env -0"],
                         capture_output=True, check=True,
                         env=dict(os.environ, QSE_SETUP_ENV=NO_OVERLAY)).stdout.decode()
    env = {}
    for kv in out.split("\0"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k.startswith("QWEN38_"):
                env[k] = v
    return env


def server_cmd(a) -> list[str]:
    repo = Path(os.path.expanduser(a.repo))
    return [a.python, "-u", "server/app.py", "--host", a.host, "--port", str(a.port),
            "--served-model", a.served_model, "--max-len", str(a.max_len),
            "--drafter", "lenrouter", "--dflash2-path", "greedy", "--len-fixed", "0",
            "--tree", "--budget", "16", "--corpus", str(repo / "corpus"),
            "--dflash2-ckpt", str(repo / "train/ft-b8-v2"), "--dflash2-ckpt16", str(repo / "train/ft-b16"),
            "--nvfp4", a.nvfp4, "--fp8-head", a.head,
            *(["--no-session-cache", "--no-prefix-cache", "--cache-budget-gb", "0"] if not a.caches else []),
            "--verbose", "--suffix-store=", "--len-latch", "--drop-idle",
            "--reasoning-format", "tags", "--default-max-tokens", "32768",
            # the release's sampled verify (ops/start.sh SAMPLED_TREE=det): a sampled request walks
            # the greedy request's tree; "chain" = the q-aware chain every row before 2026-10-09
            # 11:40 ran
            *(["--sampled-tree=det"] if a.sampled_tree == "det" else [])] + a.server_arg


def health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


SAMPLING: dict = {}


def stream(port: int, model: str, content: str, max_tokens: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": content}], "stream": True,
            "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True}}
    body.update(SAMPLING)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, text, usage = None, [], None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = (ch.get("delta") or {})
                piece = d.get("content") or d.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(piece)
    e2e = time.time() - t0
    n = (usage or {}).get("completion_tokens") or 0
    return {"ttft_s": ttft, "e2e_s": e2e, "completion": n,
            "prompt": (usage or {}).get("prompt_tokens"), "text": "".join(text),
            "tok_s": (n - 1) / (e2e - ttft) if ttft is not None and n > 1 and e2e > ttft else None}


def prefill_text(repo: Path, tok, n: int) -> str:
    """Held-out text cut to exactly n tokens (the chat template adds its own few)."""
    src = ""
    for name in ("heldout_prose.txt", "heldout_code.txt", "calib.txt"):
        src += open(repo / "bench" / name, encoding="utf-8").read() + "\n\n"
    ids = tok(src, add_special_tokens=False)["input_ids"]
    while len(ids) < n:
        ids = ids + ids
    return tok.decode(ids[:n])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--env-file", default=None, help="a serve.env whose QWEN38_* knobs the server gets")
    p.add_argument("--env", action="append", default=[], metavar="K=V")
    p.add_argument("--python", default=str(HOME / "recipes/ling3-flash-dgx-spark/.venv/bin/python"))
    p.add_argument("--pythonpath", default=str(HOME / "pylibs"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--served-model", default="qwen38-spark-engine")
    p.add_argument("--max-len", type=int, default=262144)
    p.add_argument("--nvfp4", default=",".join(str(HOME / "nvfp4-bf16" / f) for f in
                                               ("mlp.safetensors", "gdn.safetensors", "attn.safetensors")))
    p.add_argument("--head", default=str(HOME / "nvfp4-bf16/head-fp8.safetensors"))
    p.add_argument("--caches", action="store_true")
    p.add_argument("--server-arg", action="append", default=[])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--new", type=int, default=512)
    p.add_argument("--prefill", default="8192,32768")
    p.add_argument("--workloads", default="prose,chat",
                   help="any of prose, chat, code, and long<N>k (the code task after N*1024 tokens of "
                        "held-out text: decode at long context, an opencode session's shape)")
    p.add_argument("--long-reps", type=int, default=2,
                   help="reps of a long<N>k workload (each pays its prefill unless --caches)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="decode workloads sampled at this temperature (prefill stays greedy)")
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--sampled-tree", default="det", choices=("det", "chain"),
                   help="a sampled request's verify: det (the release's) or the q-aware chain")
    p.add_argument("--prefill-salt", action="store_true",
                   help="start every prefill prompt with its own line, so no prefill reuses another's "
                        "rows from the caches (a list may repeat a length)")
    p.add_argument("--keep", action="store_true", help="leave the server up afterwards")
    p.add_argument("--no-server", action="store_true", help="measure a server already on --port")
    p.add_argument("--json", required=True)
    a = p.parse_args()
    repo = Path(os.path.expanduser(a.repo))
    env = dict(os.environ)
    knobs = env_from_file(os.path.expanduser(a.env_file)) if a.env_file else {}
    for kv in a.env:
        k, v = kv.split("=", 1)
        knobs[k] = os.path.expanduser(v)
    knobs.setdefault("QWEN38_PROBE_SNAPSHOTS", "off")
    env.update(knobs)
    env.update({"PYTHONPATH": f"{repo}:{a.pythonpath}", "TZ": "Europe/Berlin"})
    for k in ("QWEN38_DECIDER_MODELS", "QWEN38_NARRATOR_MODEL", "QWEN38_SHADOW_MODEL"):
        env[k] = ""
    log = Path(os.path.expanduser(a.json)).with_suffix(".server.log")
    cmd = server_cmd(a)
    proc = None
    overlay = repo / "run" / "setup.env"
    rep = {"label": a.label, "repo": str(repo), "server_cmd": cmd, "knobs": knobs,
           "setup_overlay": (f"{overlay} (ignored)" if a.env_file and overlay.exists() else None),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        if not a.no_server:
            if health(a.port):
                raise SystemExit(f"something already answers on :{a.port}")
            fh = open(log, "w")
            t0 = time.time()
            proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while not health(a.port):
                if proc.poll() is not None:
                    raise SystemExit(f"server exited {proc.returncode}; see {log}")
                if time.time() - t0 > 1200:
                    raise SystemExit("server not healthy in 1200 s")
                time.sleep(3)
            rep["load_s"] = round(time.time() - t0, 1)
            print(f"[fastrun] {a.label}: healthy after {rep['load_s']} s", flush=True)
        from transformers import AutoTokenizer
        from engine.config import load_config
        tok = AutoTokenizer.from_pretrained(load_config(None).path)

        def logpos():
            return log.stat().st_size if log.exists() else 0

        def req_lines(at: int) -> list:
            if not log.exists():
                return []
            with open(log, "rb") as f:
                f.seek(at)
                return rowlog.parse_requests(f.read().decode("utf-8", "replace"))

        stream(a.port, a.served_model, "Say hello.", 16)          # warm-up
        if a.temperature > 0:
            SAMPLING.update({"temperature": a.temperature}
                            | ({"top_p": a.top_p} if a.top_p is not None else {}))
        prompts = {"prose": PROSE, "chat": CHAT, "code": CODE}
        for name in a.workloads.split(","):
            if name.startswith("long"):
                n_ctx = int(name[4:].rstrip("k")) * 1024
                prompt = prefill_text(repo, tok, n_ctx) + "\n\n---\n\n" + CODE
                reps = a.long_reps
            else:
                prompt = prompts[name]
                reps = a.reps
            runs = []
            for i in range(reps):
                at = logpos()
                r = stream(a.port, a.served_model, prompt, a.new)
                time.sleep(0.3)
                lines = req_lines(at)
                if lines:
                    ln = lines[-1]
                    r["blocks"], r["committed"] = ln.get("blocks"), ln.get("committed")
                    r["tok_round"] = (ln["committed"] / ln["blocks"]) if ln.get("blocks") else None
                    r["ms_round"] = (ln["decode_ms"] / ln["blocks"]) if ln.get("blocks") else None
                    r["accept"] = ln.get("accept")
                runs.append(r)
                print(f"[fastrun] {name} {i}: {r['tok_s']:.2f} tok/s, {r['completion']} tokens, "
                      f"tok/round {r.get('tok_round')}, ttft {r['ttft_s']:.3f} s", flush=True)
            rates = [r["tok_s"] for r in runs if r["tok_s"]]
            srt = sorted(rates)
            tr = [r["tok_round"] for r in runs if r.get("tok_round")]
            mr = [r["ms_round"] for r in runs if r.get("ms_round")]
            rep[name] = {"runs": runs, "tok_s_median": statistics.median(rates) if rates else None,
                         "tok_s_mean": statistics.mean(rates) if rates else None,
                         "tok_s_min": srt[0] if srt else None, "tok_s_max": srt[-1] if srt else None,
                         "tok_round_median": statistics.median(tr) if tr else None,
                         "ms_round_median": statistics.median(mr) if mr else None,
                         "same_text": len({r["text"] for r in runs}) == 1}
        SAMPLING.clear()
        rep["prefill"] = {}
        for j, n in enumerate(int(x) for x in a.prefill.split(",") if x):
            text = prefill_text(repo, tok, n)
            if a.prefill_salt:
                text = f"[{a.label} {time.time():.6f} {j}]\n" + text
            r = stream(a.port, a.served_model, text, 1)
            r["prefill_tok_s"] = (r["prompt"] or n) / r["ttft_s"] if r["ttft_s"] else None
            r.pop("text", None)
            rep["prefill"][str(n) if not a.prefill_salt else f"{n}#{j}"] = r
            print(f"[fastrun] prefill {n}: prompt {r['prompt']} ttft {r['ttft_s']:.2f} s "
                  f"= {r['prefill_tok_s']:.0f} tok/s", flush=True)
    finally:
        if proc is not None and not a.keep:
            try:
                os.killpg(os.getpgid(proc.pid), 15)
                proc.wait(timeout=120)
            except Exception:  # noqa: BLE001
                os.killpg(os.getpgid(proc.pid), 9)
        rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        json.dump(rep, open(os.path.expanduser(a.json), "w"), indent=1, default=str)
    print(f"[fastrun] {a.label}: " + ", ".join(
        f"{w} {rep[w]['tok_s_median']:.2f} tok/s ({rep[w]['tok_round_median']} tok/round)"
        for w in a.workloads.split(",") if w in rep and rep[w].get("tok_s_median")))


if __name__ == "__main__":
    main()
