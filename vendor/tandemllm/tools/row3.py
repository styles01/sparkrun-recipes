"""The atlas row, N times a configuration, reported as a median and a spread.

Phase 9 compared two configurations by running the row once on each, read a 7 % difference in the
mean, and drew a conclusion that evaporated when the *reverted* configuration was run a third time
and read higher than the one that had just been blamed for the difference. Two runs of the same
engine differ by 10.4 % on the mean and 6.7 % on the median, because the row is fifty requests of
an ADAPTIVE policy: the latch decides each request's block width from timing taken inside that
request, so two runs make different decisions on the same prompts.

So a single row cannot decide anything smaller than about a tenth of itself, and the fix is not a
better formula -- it is more rows. This script runs the row `--runs` times a configuration,
recomputes every row from its own per-request records with the one formula

    tok/s = (completion_tokens - 1) / (e2e_s - ttft_s)

and reports, per configuration, the MEDIAN of the runs beside the spread of the runs, so that a
difference between two configurations can be read against the noise that produced it. A difference
smaller than the spread is not a result, and the script says so rather than leaving the arithmetic
to the reader.

**Use one server for all the runs of a configuration, which is the default.** Phase 9's three rows
of the release candidate came from three separate server processes and spread 10.4 % on the mean
and 6.7 % on the median; three rows of the same configuration through ONE process spread 5.9 % and
3.1 % (phase 10, 03:51). Most of the row's run-to-run noise is between processes rather than
between rows, so `--restart-each` -- which reproduces phase 9's procedure exactly -- roughly
doubles the noise the comparison has to beat. The one number it costs is `wall`: run 1 of a shared
server carries the Triton autotuning in its warm-ups and reads 20 % longer, while the fifty
measured requests do not, so compare on `mean` and `p50` and not on `wall`.

Two things it does that a shell loop does not. The atlas runner names its output file from a hash
of the configuration, so a second run of the same spec OVERWRITES the first: each run here gets its
own `--out` directory and its record is copied out under the run's own name. And the server is
started and torn down by PID rather than by pattern, on a port given on the command line, because
the release candidate serves on :8000 from another directory and must not be touched.

    # three rows of the shipped configuration, from the speed checkout, on :8011
    python tools/row3.py --label rc --runs 3 --port 8011

    # three rows of the two flags the row could not resolve in phase 9
    python tools/row3.py --label nodes-alias --runs 3 --port 8011 \
        --env QWEN38_DF2_TREE_MODE=nodes --env QWEN38_TREE_ALIAS_STATE=1

    # a server flag, which needs the equals form because its value begins with a dash
    python tools/row3.py --label drop-idle --runs 3 --port 8011 --server-arg=--drop-idle

    # and the comparison, off the two reports, without touching the board
    python tools/row3.py --compare results/row3/rc.json results/row3/nodes-alias.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools import rowlog  # noqa: E402

DEFAULT_PY = HOME / "recipes/ling3-flash-dgx-spark/.venv/bin/python"
DEFAULT_ATLAS = HOME / "inf-atlas/bench"
DEFAULT_SPEC = "packets/ownengine.serve-single-i256-o256-v1.json"
DEFAULT_NV = ",".join(
    str(HOME / "nvfp4" / n) for n in ("mlp-clip.safetensors", "gdn-clip.safetensors", "attn-clip.safetensors")
)
DEFAULT_HEAD = str(HOME / "nvfp4/head-fp8.safetensors")


# ---------------------------------------------------------------- the row's arithmetic


def row_stats(record: dict) -> dict:
    """Recompute one atlas run from its own per-request records.

    The first token is the prefill's, so it is subtracted, and the decode time is the request's
    wall clock past its first token. Warm-ups and failures are dropped. This is the formula every
    published row is computed with, and it reproduces phase 6's published
    32.13 / 23.26 / 99.09 / 768 ms / 358.851 s exactly off the stored run.
    """
    reqs = record["raw"]["payload"]["requests"]
    rates, ttfts = [], []
    for r in reqs:
        if r.get("warmup") or r.get("status") != "ok":
            continue
        n = r.get("completion_tokens") or 0
        decode_s = (r["e2e_ms"] - r["ttft_ms"]) / 1000.0
        if n < 2 or decode_s <= 0:
            continue
        rates.append((n - 1) / decode_s)
        ttfts.append(r["ttft_ms"])
    rates.sort()
    it = record["raw"]["payload"]["iterations"][record["raw"]["payload"].get("selected_iteration", 0)]
    return {
        "n": len(rates),
        "mean": statistics.fmean(rates),
        "p50": statistics.median(rates),
        "p90": rates[min(len(rates) - 1, int(round(0.9 * (len(rates) - 1))))],
        "max": rates[-1],
        "ttft_p50_ms": statistics.median(ttfts),
        "wall_s": it.get("duration_s"),
    }


def across(runs: list[dict], key: str) -> dict:
    """Median of the runs, and the spread that says what the median is worth."""
    xs = sorted(r[key] for r in runs if r.get(key) is not None)
    if not xs:
        return {}
    med = statistics.median(xs)
    return {
        "median": med,
        "min": xs[0],
        "max": xs[-1],
        "spread_pct": 100.0 * (xs[-1] - xs[0]) / med if med else 0.0,
    }


# The statistics `--compare` resolves, and which way is better. tok_blk and ms_blk are the two
# factors of the speed: a lossless kernel must not move the first, and a faster block is
# the second going down.
HIGHER_IS_BETTER = {"mean": True, "p50": True, "p90": True, "max": True, "ttft_p50_ms": False,
                    "wall_s": False, "tok_blk": True, "ms_blk": False}


def block_factors(record: dict, log_text: str) -> dict:
    """Tokens a block and ms a block of one run, from the `[req]` lines its server printed while
    the run was on the wire (`log_text` is that slice of the log).

    The runner sends its requests one at a time, warm-ups first, and the server prints one `[req]`
    line per request, so the lines and the record's requests are the same list in the same order
    when their counts agree; warm-ups and failures are then dropped exactly as `row_stats` drops
    them. When the counts disagree the first lines are dropped as warm-ups, and the report says
    how many lines it matched."""
    reqs = record["raw"]["payload"]["requests"]
    lines = rowlog.parse_requests(log_text)
    if len(lines) == len(reqs):
        keep = [ln for r, ln in zip(reqs, lines)
                if not r.get("warmup") and r.get("status") == "ok"]
    else:
        n_warm = sum(1 for r in reqs if r.get("warmup"))
        keep = lines[n_warm:]
    out = rowlog.factors(keep)
    if out:
        out["matched"] = f"{len(lines)} lines / {len(reqs)} requests"
    return out


# The code a report measured: the box trees are rsync copies without `.git`, so a report
# names its code by a content hash of these directories, which is the same for the same commit on
# the Mac and on any box directory.
CODE_DIRS = ("engine", "server", "tools", "ops")


def code_hash(repo: Path) -> str:
    """SHA-256 over the sorted (relative path, bytes) of every file under CODE_DIRS."""
    h = hashlib.sha256()
    files = sorted(p for d in CODE_DIRS for p in (Path(repo) / d).rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for p in files:
        h.update(str(p.relative_to(repo)).encode() + b"\0")
        h.update(p.read_bytes() + b"\0")
    return h.hexdigest()


def git_head(repo: Path) -> str | None:
    if not (Path(repo) / ".git").exists():
        return None
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                       text=True)
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


# ---------------------------------------------------------------- driving the board


def health_ok(port: int, path: str = "/v1/models") -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as fh:
            return fh.status == 200
    except Exception:
        return False


def _refuse_if_service_is_up() -> None:
    """The one-engine rule, mechanically.

    A second model process beside the :8000 service has wedged sshd twice (2026-09-18), at ~70 GB
    resident and with no large arena involved. The service is the box's job; a tool that needs an
    engine must run while it is stopped. Refuse loudly instead of starting a second one.
    """
    import urllib.request
    for port in (8000,):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as fh:
                if fh.status == 200:
                    raise SystemExit(
                        f"[guard] REFUSING to start an engine: the service answers on :{port}. "
                        "Stop it first (ops/stop.sh) and arm .watchdog.off, or the box runs two "
                        "engines and wedges sshd.")
        except SystemExit:
            raise
        except Exception:
            pass


def mem_available_gb(meminfo: str | None = None) -> float:
    """MemAvailable from /proc/meminfo, in GB. On this board the GPU allocates from the same pool."""
    text = meminfo if meminfo is not None else open("/proc/meminfo").read()
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 1e9
    return float("inf")


class MemGuard:
    """Kill the test server before the board runs out of memory, instead of after.

    2026-09-23 13:00: a 131k-token probe exhausted the unified memory and the box stopped answering
    until it was power-cycled. A process group killed at a floor is a lost measurement; a
    wedged board is a lost afternoon. Polls MemAvailable every `period` seconds and SIGKILLs the
    server's process group the first time it reads below `floor_gb`.
    """

    def __init__(self, proc: subprocess.Popen, floor_gb: float = 10.0, period: float = 0.2,
                 read=mem_available_gb):
        import threading
        self.proc, self.floor_gb, self.period, self.read = proc, floor_gb, period, read
        self.tripped = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "MemGuard":
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def check(self) -> bool:
        """One poll. True when it killed the server."""
        if self.proc.poll() is not None:
            return False
        gb = self.read()
        if gb >= self.floor_gb:
            return False
        self.tripped = gb
        print(f"[memguard] MemAvailable {gb:.1f} GB < {self.floor_gb:.1f}: killing the server "
              f"(pid {self.proc.pid}) before the board wedges", flush=True)
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            if self.check():
                return


def server_cmd(a) -> list[str]:
    """The exact server command a row runs against -- also what the report records.

    The caches are off so each of the fifty requests pays the same prefill, and since 2026-09-23 so
    is the persistent suffix store: every server on the board opens the same one, it keeps prompts
    and answers, and a row of fixed prompts decoded greedily reads its own previous answers back out
    of it. `--with-suffix-store` restores the earlier rows' setting,
    for reproducing them and for nothing else.
    """
    cmd = [
        str(a.python), "-u", "server/app.py",
        "--host", "127.0.0.1", "--port", str(a.port),
        "--max-len", str(a.max_len),
        "--drafter", "lenrouter", "--dflash2-path", "greedy",
        "--len-fixed", str(a.len_fixed),
        *(["--tree", "--budget", str(a.budget)] if getattr(a, "tree", True) else []),
        "--corpus", str(a.repo / "corpus"),
        "--dflash2-ckpt", str(a.repo / "train/ft-b8-v2"),
        "--dflash2-ckpt16", str(a.repo / "train/ft-b16"),
        "--nvfp4", a.nvfp4, "--fp8-head", a.head,
        # the long-context probe's --served-caches: the prefix and session caches on,
        # as ops/start.sh runs :8000, so a prefill takes the service's 1,024-row chunks
        *(["--cache-budget-gb", str(a.cache_gb)] if getattr(a, "served_caches", False) else
          ["--no-session-cache", "--no-prefix-cache", "--cache-budget-gb", "0"]),
        "--verbose",
    ]
    store = getattr(a, "store", "off")
    if getattr(a, "with_suffix_store", False):
        store = "live-rw"
    if store == "off":
        cmd.append("--suffix-store=")
    elif store == "live":
        cmd.append("--suffix-store-readonly")
    elif store == "clean":
        cmd += [f"--suffix-store={a.clean_store}", "--suffix-store-readonly"]
    cmd.append("--len-latch" if a.len_latch else "--no-len-latch")
    return cmd + list(a.server_arg)


def recorded_args(a) -> dict:
    """Every knob of the run, defaults included (a benchmark rule): a report that leaves
    one out cannot be told apart from a run that had it set. `p0-fx-s1..s3` were taken with
    `--len-fixed 16 --no-len-latch` and their reports said only `--drop-idle`."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(vars(a).items())
            if k != "compare"}


def exit_on_term() -> None:
    """SIGTERM / SIGHUP end this process through its `finally` blocks, not around them.

    A test server runs in a session of its own (so `stop_server` can signal its group), which is
    also why a signal to the TOOL's process group does not reach it. On 2026-09-25 03:44 hold.sh,
    signalled, stopped a longctx_probe's group as designed: Python's default SIGTERM action ended
    the probe without its `finally: stop_server(...)`, the :8011 engine lived on in its own session,
    hold.sh refused to restart :8000 beside it and released the box lock -- and the next hold
    (another agent's) found a second engine on the board. With this, the signal becomes SystemExit,
    the caller's `finally` stops the server by PID, and the hold's wait sees it gone."""
    def _exit(signum, _frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit)


def start_server(a, env_extra: dict[str, str], log: Path) -> subprocess.Popen:
    exit_on_term()
    _refuse_if_service_is_up()
    if health_ok(a.port):
        raise SystemExit(f"[row3] REFUSING: something already answers on :{a.port}")
    cmd = server_cmd(a)
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(a.pythonpath), "TZ": "Europe/Berlin"})
    env.update(env_extra)
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w")
    print(f"[row3] start :{a.port}  {' '.join(f'{k}={v}' for k, v in sorted(env_extra.items()))}")
    proc = subprocess.Popen(cmd, cwd=a.repo, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < a.start_timeout:
        if proc.poll() is not None:
            raise SystemExit(f"[row3] server exited with {proc.returncode}; see {log}")
        if health_ok(a.port):
            print(f"[row3] healthy after {time.time() - t0:.0f}s")
            if getattr(a, "mem_floor_gb", 10.0) > 0:
                proc.memguard = MemGuard(proc, getattr(a, "mem_floor_gb", 10.0)).start()
            return proc
        time.sleep(3)
    stop_server(proc)
    raise SystemExit(f"[row3] server never became healthy; see {log}")


def stop_server(proc: subprocess.Popen | None, grace: int = 60) -> None:
    """SIGTERM the process group we started, by PID. Never a pattern: the release candidate is
    another `server/app.py` on this board and a pattern kill would take the operator's service down."""
    if getattr(proc, "memguard", None) is not None:
        proc.memguard.stop()
    if proc is None or proc.poll() is not None:
        return
    print(f"[row3] stop pid {proc.pid}")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < grace:
        if proc.poll() is not None:
            return
        time.sleep(1)
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=30)


def refuse_conductor(port: int) -> None:
    """`--no-server` against a server with the conductor: it serves one model, `TandemLLM`, whose raw
    path is the `X-Tandem: raw` header, and the atlas runner sends no headers of its own. Such a
    row would measure the conductor, not the engine, so it is refused. A server row3 starts itself
    runs without `--conductor-config` and is the raw engine."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
            names = [m.get("id") for m in json.loads(r.read()).get("data") or []]
    except Exception as exc:                                       # noqa: BLE001
        raise SystemExit(f"[row3] --no-server: :{port}/v1/models did not answer ({exc})")
    if "TandemLLM" in names:
        raise SystemExit(f"[row3] REFUSING --no-server: :{port} serves TandemLLM (the conductor); "
                         "the atlas runner cannot send X-Tandem: raw. Let row3 start its own "
                         "server (no --no-server), which runs the raw engine.")


def run_row(a, out_dir: Path, tag: str, server_log: Path | None = None) -> dict:
    """One atlas row into a directory of its own, so the runner's hashed filename cannot collide
    with the previous run's. With `server_log`, the run's slice of it gives tokens/block and
    ms/block."""
    log_at = server_log.stat().st_size if server_log is not None and server_log.exists() else 0
    run_out = out_dir / f"raw-{tag}"
    if run_out.exists():
        shutil.rmtree(run_out)
    run_out.mkdir(parents=True)
    cmd = [
        str(a.atlas / ".venv/bin/atlas-bench"), "run",
        "--spec", a.spec,
        "--base-url", f"http://127.0.0.1:{a.port}/v1",
        "--out", str(run_out),
        "--login", a.login, "--dedicated", "--tokenizer", a.tokenizer,
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=a.atlas, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
        raise SystemExit(f"[row3] atlas-bench failed ({r.returncode}) for {tag}")
    # The runner does not write into `--out`: it writes into a registry-shaped tree underneath it,
    # `results/<engine>/<org>/<model>/<hardware>/<hash>--<workload>--<hash>.json`. So the record is
    # found by walking, and the walk is what makes a per-run directory worth having -- the leaf
    # name is a hash of the spec and a second run of the same spec overwrites the first.
    found = sorted(run_out.rglob("*.json"))
    found = [p for p in found if "serve-single" in p.name] or found
    if not found:
        raise SystemExit(f"[row3] no run record in {run_out}")
    record = json.loads(found[-1].read_text())
    st = row_stats(record)
    st["tag"] = tag
    st["record"] = str(found[-1])
    st["bench_wall_s"] = round(time.time() - t0, 1)
    if server_log is not None and server_log.exists():
        with open(server_log, "rb") as fh:
            fh.seek(log_at)
            st.update(block_factors(record, fh.read().decode("utf-8", "replace")))
    blk = (f"  {st['tok_blk']:.2f} tok/blk  {st['ms_blk']:.1f} ms/blk" if "tok_blk" in st else "")
    print(f"[row3] {tag:22s} mean {st['mean']:6.2f}  p50 {st['p50']:6.2f}  p90 {st['p90']:6.2f}  "
          f"max {st['max']:7.2f}  ttft p50 {st['ttft_p50_ms']:.0f} ms  wall {st['wall_s']:.1f} s{blk}",
          flush=True)
    return st


# ---------------------------------------------------------------- reporting


def report(label: str, runs: list[dict]) -> dict:
    keys = ("mean", "p50", "p90", "max", "ttft_p50_ms", "wall_s", "tok_blk", "ms_blk",
            "tok_blk_p50", "ms_blk_p50")
    summary = {k: across(runs, k) for k in keys}
    summary = {k: v for k, v in summary.items() if v}

    def cell(r, k, fmt):
        v = r.get(k) if isinstance(r, dict) else None
        return format(v, fmt) if v is not None else "n/a".rjust(int(fmt.split(".")[0]))

    print()
    print(f"=== {label}: {len(runs)} rows ===")
    print(f"{'run':<10}{'mean':>9}{'p50':>9}{'p90':>9}{'max':>9}{'ttft p50':>11}{'wall':>9}"
          f"{'tok/blk':>9}{'ms/blk':>9}")
    for r in runs:
        print(f"{r['tag']:<10}{r['mean']:>9.2f}{r['p50']:>9.2f}{r['p90']:>9.2f}{r['max']:>9.2f}"
              f"{r['ttft_p50_ms']:>10.0f}m{r['wall_s']:>8.1f}"
              f"{cell(r, 'tok_blk', '9.2f')}{cell(r, 'ms_blk', '9.1f')}")
    med = {k: v["median"] for k, v in summary.items()}
    spr = {k: v["spread_pct"] for k, v in summary.items()}
    print(f"{'MEDIAN':<10}{med['mean']:>9.2f}{med['p50']:>9.2f}{med['p90']:>9.2f}{med['max']:>9.2f}"
          f"{med['ttft_p50_ms']:>10.0f}m{med['wall_s']:>8.1f}"
          f"{cell(med, 'tok_blk', '9.2f')}{cell(med, 'ms_blk', '9.1f')}")
    print(f"{'spread %':<10}{spr['mean']:>9.1f}{spr['p50']:>9.1f}{spr['p90']:>9.1f}{spr['max']:>9.1f}"
          f"{spr['ttft_p50_ms']:>11.1f}{spr['wall_s']:>9.1f}"
          f"{cell(spr, 'tok_blk', '9.1f')}{cell(spr, 'ms_blk', '9.1f')}")
    return summary


def verdicts(base: dict, other: dict) -> list[dict]:
    """Per statistic: the medians, the delta, the noise, and RESOLVED / not resolved with its
    direction. A statistic either report lacks (an old report has no tok_blk) reads n/a."""
    out = []
    for k, up in HIGHER_IS_BETTER.items():
        b, o = base["summary"].get(k), other["summary"].get(k)
        if not b or not o:
            out.append({"stat": k, "verdict": "n/a"})
            continue
        delta = 100.0 * (o["median"] - b["median"]) / b["median"]
        noise = max(b["spread_pct"], o["spread_pct"])
        # The ranges are disjoint when every row of one configuration beat every row of the
        # other. With three rows a side that is the strongest thing this row can say, and it
        # is the claim phase 9 believed it had after one row each.
        disjoint = o["min"] > b["max"] or o["max"] < b["min"]
        resolved = abs(delta) > noise and disjoint
        better = (delta > 0) == up
        out.append({"stat": k, "base": b["median"], "other": o["median"], "delta": delta,
                    "noise": noise, "disjoint": disjoint, "resolved": resolved,
                    "worse": resolved and not better,
                    "verdict": ("RESOLVED " + ("better" if better else "WORSE")) if resolved
                    else "not resolved"})
    return out


def compare(paths: list[str]) -> None:
    """Read two or more reports and say, for each statistic, whether the difference between their
    medians is bigger than the spread that produced them."""
    reports = [json.loads(Path(p).read_text()) for p in paths]
    base = reports[0]
    print(f"baseline: {base['label']}  ({len(base['runs'])} rows)  code {base.get('code_sha256', 'n/a')[:16]}")
    for other in reports[1:]:
        hb, ho = base.get("code_sha256"), other.get("code_sha256")
        same = "n/a (a report without a code hash)" if not (hb and ho) else \
            ("same code" if hb == ho else "different code")
        print(f"\nagainst:  {other['label']}  ({len(other['runs'])} rows)  code {(ho or 'n/a')[:16]}"
              f"  -- {same}")
        print(f"{'stat':<12}{'baseline':>10}{'other':>10}{'delta %':>10}{'noise %':>10}{'disjoint':>10}   verdict")
        for v in verdicts(base, other):
            if v["verdict"] == "n/a":
                print(f"{v['stat']:<12}{'n/a':>10}{'n/a':>10}{'':>30}   n/a")
                continue
            print(f"{v['stat']:<12}{v['base']:>10.2f}{v['other']:>10.2f}{v['delta']:>+10.1f}"
                  f"{v['noise']:>10.1f}{('yes' if v['disjoint'] else 'no'):>10}   {v['verdict']}")
        print("\n  RESOLVED needs both: a difference bigger than the run-to-run noise, and run ranges\n"
              "  that do not overlap. 'not resolved' means the row did not decide it -- more rows or a\n"
              "  different instrument would. It does NOT mean the two configurations are equal.")


# ---------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", nargs="+", help="read saved reports and compare them; touches no board")
    p.add_argument("--label", default="row3")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--env", action="append", default=[], metavar="K=V",
                   help="environment for the server process, repeatable")
    p.add_argument("--server-arg", action="append", default=[],
                   help="extra server/app.py argument, repeatable. A value that begins with a dash "
                        "needs the equals form -- `--server-arg=--drop-idle`, not "
                        "`--server-arg --drop-idle`, which argparse reads as a missing value")
    p.add_argument("--restart-each", action="store_true",
                   help="a fresh server process per row, which is what phase 9 did and what roughly "
                        "doubles the spread: 10.4 %% on the mean across processes against 5.9 %% "
                        "within one. The default keeps one server, and then `wall` is not "
                        "comparable across runs because run 1 pays for the autotuning")
    p.add_argument("--no-server", action="store_true", help="bench a server that is already up on --port")
    p.add_argument("--repo", type=Path, default=REPO)
    p.add_argument("--python", type=Path, default=DEFAULT_PY)
    p.add_argument("--pythonpath", type=Path, default=HOME / "pylibs")
    p.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    p.add_argument("--spec", default=DEFAULT_SPEC)
    p.add_argument("--login", default="0xBakeer")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--len-fixed", type=int, default=0)
    p.add_argument("--len-latch", action="store_true", default=True)
    p.add_argument("--no-len-latch", dest="len_latch", action="store_false")
    p.add_argument("--nvfp4", default=DEFAULT_NV)
    p.add_argument("--head", default=DEFAULT_HEAD)
    p.add_argument("--start-timeout", type=int, default=600)
    p.add_argument("--out-dir", type=Path, default=REPO / "results/row3")
    p.add_argument("--store", default="off", choices=("off", "clean", "live"),
                   help="the persistent suffix store the server reads, never writes: off (the "
                        "engine on new text), clean (--clean-store: real traffic's store with every "
                        "atlas prompt removed, tools/store_audit.py), live (the box's store as it "
                        "is, which holds earlier rows' answers). ")
    p.add_argument("--clean-store", default=str(HOME / "qwen38-suffix-norow-0923"))
    p.add_argument("--mem-floor-gb", type=float, default=10.0,
                   help="kill the server if MemAvailable falls below this; 0 = off")
    p.add_argument("--with-suffix-store", action="store_true",
                   help="leave the persistent suffix store on, as every row before 2026-09-23 "
                        "did; it then contains the row's own previous answers")
    a = p.parse_args()

    if a.compare:
        compare(a.compare)
        return

    env_extra = dict(kv.split("=", 1) for kv in a.env)
    out_dir = a.out_dir / a.label
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, proc = [], None

    # A `finally` is not enough. This script is driven by a hold that may be cut short, and a
    # SIGTERM to Python runs no `finally` block at all: on 2026-09-18 at 07:46 a terminated run
    # left its :8001 server resident beside the restarted release candidate -- two full engines on
    # a bandwidth-bound board, which is the one thing the hold protocol exists to prevent. Turning
    # the signal into an exception puts the teardown back on the normal path.
    def _bail(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(_sig, _bail)
        except (ValueError, OSError):                 # not the main thread, or no such signal
            pass

    if a.no_server:
        refuse_conductor(a.port)
    try:
        log = None if a.no_server else out_dir / "server.log"
        if not a.no_server and not a.restart_each:
            proc = start_server(a, env_extra, log)
        for i in range(1, a.runs + 1):
            if a.restart_each and not a.no_server:
                log = out_dir / f"server-{i}.log"
                proc = start_server(a, env_extra, log)
            runs.append(run_row(a, out_dir, f"{i}", log))
            if a.restart_each and not a.no_server:
                stop_server(proc)
                proc = None
    except KeyboardInterrupt as exc:
        stop_server(proc)
        proc = None
        print(f"[row3] stopped by {exc}; {len(runs)} row(s) completed, server torn down")
        if not runs:
            raise SystemExit(130)
    finally:
        stop_server(proc)

    summary = report(a.label, runs)
    pooled: dict = {}
    for r in runs:
        rowlog.add_hist(pooled, r.get("accept") or {})
    doc = {
        "label": a.label,
        "env": env_extra,
        "server_arg": a.server_arg,
        "spec": a.spec,
        "restart_each": bool(a.restart_each),
        "suffix_store": "live-rw" if a.with_suffix_store else a.store,
        "args": recorded_args(a),
        "server_cmd": None if a.no_server else server_cmd(a),
        "code_sha256": code_hash(a.repo),
        "git_head": git_head(a.repo),
        "runs": runs,
        "summary": summary,
        # where in its blocks the row's drafts went wrong, all runs pooled
        "accept_curve": rowlog.curve(pooled) if pooled else [],
        "first_miss": rowlog.first_miss(pooled) if pooled else {},
        "taken": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = a.out_dir / f"{a.label}.json"
    path.write_text(json.dumps(doc, indent=2))
    print(f"\n[row3] {path}")


if __name__ == "__main__":
    main()
