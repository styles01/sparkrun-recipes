#!/home/jaita/yue2/.venv/bin/python
"""yue2-server.py — minimal HTTP job wrapper around the YuE2-3B CLI lane (spark2).

Task t_b9b2cfe0: Lara drives batch renders over HTTP; CLI stays single-job serial.
Task t_19b16e7e: expose the conditioning knobs the engine already supports, with
validated passthrough + honest QC echo (cfg_scale, sampling, truncation, seeds).

POST /render {"abc","lyrics","caption","seed","duration_s","id","batch","cot",
              "cfg_scale","abc_sampling","semantic_sampling","seed_randomize",
              "dry_run"}                        -> 202 {job_id} (200 if dry_run)
GET  /job/<id>   -> queued|running|complete|failed (+ wav path, seconds, timing,
                   truncated/seed/cfg/peak-clipping echo of what ACTUALLY ran)
GET  /health     -> lane state, queue depth, MemAvailable, last error, knob contract,
                   done_truncated + done_clipped
Design: subprocess per job (free VRAM between jobs; coexists with image :8030 +
TTS :8020 + decider :8715). MemAvailable gate before each job. WAV lands in
/home/jaita/songgeneration/output/yue2-batch-<N>/<id>/audio.wav (CLI writes
audio.flac; we convert with soundfile). duration_s -> semantic tokens @ 25 tok/s
(Sampling.min_tokens/max_tokens clamp). Concurrency = 1, enforced by queue.
Stdlib-only in the server itself; heavy imports happen inside job subprocesses.

KNOB CONTRACT (mirrors yue2.protocol; bad values fail 400 at POST, never 3 minutes
later inside a loaded pipeline):
  cfg_scale        float [0,20]. Default None -> engine guidance 1.0 (cot melody/full)
                   or 1.01 (cot off) = SINGLE AR BRANCH. Any value != 1 builds the
                   negative branch -> cfg_branches=2, ~2x AR time (sampling.py:109).
  abc_sampling /   dicts of {temperature[0..5], top_p(0..1], top_k>=1 int,
  semantic_sampling repetition_penalty>0, penalty_window 1..100 int, min_tokens,
                   max_tokens} (counts must be ints). Partial dicts inherit the
                   lane default for the keys you omit (protocol.resolve_sampling).
                   max_tokens <= 23040 so prefix+budget stays inside CONTEXT=24576.
                   Explicit min/max_tokens WIN over the duration_s clamp; duration_s
                   only fills the keys you left out.
  seed             int [0,2**63). Omitted + seed_randomize=false -> lane default
                   831001 (yue2.protocol.SongRequest.seed), i.e. DETERMINISTIC.
  seed_randomize   true + seed omitted -> server draws a fresh seed at enqueue and
                   records it, so the job stays replayable by re-firing that seed.
                   /job echoes seed + seed_source (explicit|randomized|lane_default).
  truncated        echoed as the engine reports it {abc: bool, semantic: bool}, plus
                   truncated_any and hit_token_cap (semantic hit max_tokens => the
                   song was CUT, not ended). max_tokens is a hard stop: the loop
                   breaks at the cap with no end token (sampling.py:105-126).
  peak QC          every completed render echoes peak (int16), peak_dBFS and
                   clipped_samples off the PCM_16 WAV, computed with the EXACT
                   definition qc_sweep.py uses (clipped = abs(v) >= 32760, within
                   8 LSB of full scale) so the batch owner sees ceiling-riding at
                   enqueue and /job numbers can never disagree with a spark2 sweep.
                   Loudness matching stays OUT of the lane (Mac delivery step,
                   loudnorm_takes.py; policy t_64b86f82) — this is reporting only.
"""
import json, os, re, secrets, shutil, subprocess, sys, threading, time, uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VENV_BIN   = "/home/jaita/yue2/.venv/bin"
MODEL_DIR  = "/home/jaita/models/yue2/YuE2-3B"
VAE_DIR    = "/home/jaita/models/yue2/YuE2-Vae"
OUT_ROOT   = Path("/home/jaita/songgeneration/output")
STATE_DIR  = Path("/home/jaita/yue2/server-state")
JOBS_FILE  = STATE_DIR / "jobs.json"
PORT       = int(os.environ.get("YUE2_PORT", "8045"))
BUDGET     = float(os.environ.get("YUE2_BUDGET", "14"))      # per-process VRAM fraction cap (GiB)
MEM_GATE   = float(os.environ.get("YUE2_MEM_GATE", "16"))    # GiB MemAvailable required to start a job
TOKENS_PER_SEC = 25.0                                        # 48000 Hz / 1920 downsampling
ID_RE      = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}")

# --- knob constants (kept in sync with yue2/protocol.py of the installed kit) ---
CONTEXT          = 24576                                     # protocol.CONTEXT
PREFIX_GUARD     = 1536                                      # room for instr+tags+lyrics+ABC
MAX_TOKENS_CAP   = CONTEXT - PREFIX_GUARD                    # 23040
DEFAULT_SEED     = 831001                                    # SongRequest.seed default
GUIDANCE_DEFAULT = {"off": 1.01, "melody": 1.0, "full": 1.0}  # SongRequest.guidance
# Lane baselines (GenerationConfig().semantic / .abc). Single source of truth for both
# the /health knob contract and the effective-window math below.
SEMANTIC_DEFAULT = {"temperature": 1.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2,
                    "penalty_window": 50, "min_tokens": 200, "max_tokens": 9000}
ABC_DEFAULT      = {"temperature": 0.7, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.005,
                    "penalty_window": 100, "min_tokens": 32, "max_tokens": 4096}
SAMPLING_COUNTS  = ("top_k", "penalty_window", "min_tokens", "max_tokens")
SAMPLING_KEYS    = frozenset(("temperature", "top_p", "top_k", "repetition_penalty",
                              "penalty_window", "min_tokens", "max_tokens"))

jobs: "OrderedDict[str, dict]" = OrderedDict()
lock   = threading.Lock()
cv     = threading.Condition(lock)
LAST_ERR = None

def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return -1.0

def save_jobs():
    tmp = JOBS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(jobs)))
    os.replace(tmp, JOBS_FILE)

def batch_n(requested):
    existing = {int(p.name.split("-")[-1]) for p in OUT_ROOT.glob("yue2-batch-*")
                if p.name.split("-")[-1].isdigit()}
    if requested is not None:
        if not (isinstance(requested, int) and 1 <= requested <= 9999):
            raise ValueError("batch must be an int 1..9999")
        return requested
    return max(existing) + 1 if existing else 1

def validate_sampling(name, value):
    """Mirror yue2.protocol.Sampling.__post_init__ so bad knobs fail at POST, not
    after a 3-minute model load. Returns the partial override dict (defaults for the
    omitted keys are applied by the engine via resolve_sampling)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object of Sampling overrides")
    unknown = sorted(set(value) - SAMPLING_KEYS)
    if unknown:
        raise ValueError(f"{name}: unknown keys {unknown}; allowed {sorted(SAMPLING_KEYS)}")
    out = {}
    for key in SAMPLING_COUNTS:
        if key not in value or value[key] is None:
            continue
        v = value[key]
        if type(v) is not int:
            raise ValueError(f"{name}.{key} must be an integer (got {v!r})")
        out[key] = v
    for key in ("temperature", "top_p", "repetition_penalty"):
        if key in value and value[key] is not None:
            v = value[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name}.{key} must be a number (got {v!r})")
            if not (v == v and abs(v) != float("inf")):
                raise ValueError(f"{name}.{key} must be finite")
            out[key] = float(v)
    t, p, r = out.get("temperature", 1.0), out.get("top_p", 0.95), out.get("repetition_penalty", 1.2)
    if not 0 <= t <= 5 or not 0 < p <= 1:
        raise ValueError(f"{name}: require temperature in [0,5] and top_p in (0,1]")
    if r <= 0:
        raise ValueError(f"{name}: repetition_penalty must be > 0")
    w = out.get("penalty_window", 50)
    if not 1 <= w <= 100:
        raise ValueError(f"{name}: penalty_window must be in [1,100]")
    k = out.get("top_k", 100)
    if k < 1:
        raise ValueError(f"{name}: top_k must be >= 1")
    mn, mx = out.get("min_tokens", 200), out.get("max_tokens", 9000)
    if mx > MAX_TOKENS_CAP:
        raise ValueError(f"{name}.max_tokens {mx} exceeds {MAX_TOKENS_CAP} "
                         f"(prefix + budget must stay inside CONTEXT {CONTEXT})")
    if not 0 <= mn <= mx or mx < 1:
        raise ValueError(f"{name}: require 0 <= min_tokens <= max_tokens")
    return out

def resolve_sampling_pair(abc_s, sem_s, duration_s):
    """duration_s derives the semantic token window (25 tok/s, 0.92/1.06 spread);
    an explicit min/max_tokens in semantic_sampling wins over that clamp. A window
    the two sources cannot agree on (explicit max below the duration-derived min) is
    a contradiction -> 400 with the numbers in it, never a silently ignored knob."""
    abc_s, sem_s = validate_sampling("abc_sampling", abc_s), validate_sampling("semantic_sampling", sem_s)
    sem = dict(sem_s or {})
    wanted_min, wanted_max = "min_tokens" in sem, "max_tokens" in sem
    parts = []
    if duration_s:
        tgt = duration_s * TOKENS_PER_SEC
        if "min_tokens" not in sem:
            sem["min_tokens"] = int(tgt * 0.92)
        if "max_tokens" not in sem:
            sem["max_tokens"] = int(tgt * 1.06)
    if sem:
        mn, mx = sem.get("min_tokens"), sem.get("max_tokens")
        if wanted_max and duration_s and mn is not None and mx < mn:
            raise ValueError(
                f"semantic_sampling.max_tokens {mx} contradicts duration_s {duration_s}s "
                f"(needs min_tokens {mn} = {duration_s}*{TOKENS_PER_SEC:.0f}*0.92); "
                f"drop duration_s or raise max_tokens")
        validate_sampling("semantic_sampling", sem)   # re-check the merged window
    for key, explicit in (("min_tokens", wanted_min), ("max_tokens", wanted_max)):
        if key in sem:
            parts.append(("explicit " if explicit else "duration_s ") + key)
    budget_from = ", ".join(parts) if parts else "engine_default"
    return abc_s, (sem or None), budget_from

def validate_cfg(value):
    """SongRequest.cfg_scale contract: finite float in [0,20]. JSON `true` is refused
    here on purpose — Python treats True == 1.0, so the engine would silently run a
    single CFG branch on a driver that meant to pass a scale."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("cfg_scale must be a number in [0,20]")
    v = float(value)
    if not (v == v and abs(v) != float("inf")) or not 0 <= v <= 20:
        raise ValueError("cfg_scale must be finite and in [0,20]")
    return v

def validate_seed(value):
    """SongRequest.seed contract: int in [0, 2**63). type() not isinstance(): True
    would pass as seed 1."""
    if value is None:
        return None
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("seed must be int in [0, 2**63)")
    return value

def resolve_seed(seed, randomize):
    if seed is not None:
        return validate_seed(seed), "explicit"
    if randomize:
        return secrets.randbelow(2**63), "randomized"
    return DEFAULT_SEED, "lane_default"

def flac_to_wav(flac, wav):
    """Engine FLAC -> PCM_16 WAV copy. Returns (seconds, qc) where qc is the
    ceiling-riding report echoed on /job. Peak/clipped are computed off the
    written PCM_16 WAV with the EXACT definition qc_sweep.py uses (clipped =
    abs(v) >= 32760) so lane numbers == sweep numbers. NOTE: reads the written
    WAV's byte length (not getnframes): audio is STEREO, and int16 array()
    keeps a ~140s render at ~27MB instead of a ~500MB Python-int tuple."""
    import math, soundfile as sf, wave
    from array import array
    audio, sr = sf.read(str(flac))
    sf.write(str(wav), audio, sr, subtype="PCM_16")
    with wave.open(str(wav), "rb") as f:
        data = f.readframes(f.getnframes())
    pcm = array("h")
    pcm.frombytes(data)
    if sys.byteorder != "little":
        pcm.byteswap()
    peak = max(max(pcm), -min(pcm)) if len(pcm) else 0
    qc = {"peak": peak,
          "peak_dBFS": round(20 * math.log10(peak / 32767.0), 2) if peak else None,
          "clipped_samples": sum(1 for v in pcm if abs(v) >= 32760)}
    return len(audio) / sr, qc

def run_job(job):
    global LAST_ERR
    jid, batch, spec = job["id"], job["batch"], job["spec"]
    outdir = OUT_ROOT / f"yue2-batch-{batch}" / jid
    reqfile = STATE_DIR / f"{jid}.request.json"
    logfile = STATE_DIR / f"{jid}.log"
    req = {"id": jid, "style": spec["caption"], "lyrics": spec["lyrics"],
           "abc": spec["abc"], "cot": spec["cot"], "seed": spec["seed"]}
    if spec.get("cfg_scale") is not None:
        req["cfg_scale"] = spec["cfg_scale"]
    if spec.get("abc_sampling"):
        req["abc_sampling"] = spec["abc_sampling"]
    if spec.get("semantic_sampling"):
        req["semantic_sampling"] = spec["semantic_sampling"]
    reqfile.write_text(json.dumps(req))
    cmd = [f"{VENV_BIN}/yue2", "generate", "--request", str(reqfile),
           "--model", MODEL_DIR, "--vae", VAE_DIR,
           "--device", "auto", "--budget", str(BUDGET), "--offline",
           "--output", str(outdir.parent), "--quiet"]
    with lock:
        jobs[jid].update(status="running", started=time.time())
        save_jobs()
    with open(logfile, "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd="/home/jaita/yue2")
    result = outdir / "result.json"
    if p.returncode != 0 or not result.exists():
        fail = outdir / "failure.json"
        reason = json.loads(fail.read_text())["reason"] if fail.exists() else \
                 f"exit {p.returncode}; see {logfile}"
        with lock:
            jobs[jid].update(status="failed", finished=time.time(), error=reason)
            LAST_ERR = f"{jid}: {reason[:200]}"
            save_jobs()
        return
    meta = json.loads(result.read_text())
    # config.json is where the pipeline records what it actually applied
    cfg_file = outdir / "config.json"
    applied = {}
    if cfg_file.exists():
        try:
            c = json.loads(cfg_file.read_text())
            applied = {"cfg_scale_applied": c.get("cfg_scale"),
                       "cfg_negative": c.get("cfg_negative"),
                       "config_overrides": c.get("overrides")}
        except Exception:
            applied = {}
    wav = outdir / "audio.wav"
    seconds, qc = flac_to_wav(outdir / "audio.flac", wav)
    trunc = meta.get("truncated") or {}
    sem_timing = (meta.get("timing") or {}).get("semantic") or {}
    max_tokens = (spec.get("semantic_sampling") or {}).get("max_tokens")
    sem_tokens = sem_timing.get("output_tokens")
    with lock:
        jobs[jid].update(status="complete", finished=time.time(),
                         wav=str(wav), flac=str(outdir / "audio.flac"),
                         seconds=round(seconds, 2),
                         **qc,                         # peak / peak_dBFS / clipped_samples
                         e2e_seconds=round(meta.get("timing", {}).get("e2e_seconds", 0), 1),
                         truncated=trunc,
                         truncated_any=bool(trunc.get("abc") or trunc.get("semantic")),
                         semantic_tokens=sem_tokens,
                         semantic_max_tokens=max_tokens,
                         hit_token_cap=bool(trunc.get("semantic") and max_tokens
                                            and sem_tokens is not None
                                            and sem_tokens >= max_tokens),
                         cfg_branches=sem_timing.get("cfg_branches"),
                         **applied)
        save_jobs()

def worker():
    while True:
        with cv:
            while not any(j["status"] == "queued" for j in jobs.values()):
                cv.wait()
            jid = next(k for k, j in jobs.items() if j["status"] == "queued")
            job = jobs[jid]
        avail = mem_avail_gib()
        if avail < MEM_GATE:
            with lock:
                jobs[jid]["note"] = f"waiting: MemAvailable {avail:.1f}G < {MEM_GATE}G gate"
                save_jobs()
            time.sleep(15)
            continue
        try:
            run_job(job)
        except BaseException as exc:            # never let the worker die
            with lock:
                jobs[jid].update(status="failed", finished=time.time(), error=str(exc))
                LAST_ERR = f"worker: {exc}"
                save_jobs()

KNOBS = {
    "cfg_scale": {"type": "float", "range": [0, 20], "default": None,
                  "default_guidance": GUIDANCE_DEFAULT,
                  "note": "any value != 1 enables the negative CFG branch: cfg_branches=2, ~2x AR time"},
    "abc_sampling": {"type": "object", "keys": sorted(SAMPLING_KEYS), "lane_default": ABC_DEFAULT},
    "semantic_sampling": {"type": "object", "keys": sorted(SAMPLING_KEYS),
                          "lane_default": SEMANTIC_DEFAULT,
                          "max_tokens_cap": MAX_TOKENS_CAP,
                          "note": "partial dict inherits lane defaults; explicit min/max_tokens win over duration_s; "
                                  "min_tokens >= max_tokens => every render truncates (end token can never fire)"},
    "seed": {"type": "int", "range": [0, 2**63], "default": DEFAULT_SEED,
             "note": "omitted + no seed_randomize == every render identical (engine default seed)"},
    "seed_randomize": {"type": "bool", "default": False,
                       "note": "with seed omitted: draw a fresh seed at enqueue, echoed on /job as seed_source=randomized"},
    "duration_s": {"type": "number", "range": [5, 240],
                   "note": f"-> semantic min/max_tokens at {TOKENS_PER_SEC:.0f} tok/s, spread 0.92/1.06; explicit sampling tokens win"},
    "dry_run": {"type": "bool", "default": False,
                "note": "validate + echo the resolved request/knobs without enqueueing or rendering"},
    "echo_on_job": ["truncated", "truncated_any", "hit_token_cap", "semantic_tokens",
                    "semantic_max_tokens", "seed", "seed_source", "cfg_branches",
                    "cfg_scale_applied", "config_overrides",
                    "peak", "peak_dBFS", "clipped_samples"],
}

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            with lock:
                st = [j["status"] for j in jobs.values()]
                trunc_n = sum(1 for j in jobs.values()
                              if j["status"] == "complete" and j.get("truncated_any"))
                clip_n = sum(1 for j in jobs.values()
                             if j["status"] == "complete" and j.get("clipped_samples"))
            self._send(200, {"ok": True, "lane": "yue2-3b-cli", "port": PORT,
                             "queue": st.count("queued"), "running": st.count("running"),
                             "done": st.count("complete"), "failed": st.count("failed"),
                             "done_truncated": trunc_n,
                             "done_clipped": clip_n,
                             "mem_available_gib": round(mem_avail_gib(), 1),
                             "concurrency": 1, "budget_gib": BUDGET,
                             "mem_gate_gib": MEM_GATE, "last_error": LAST_ERR,
                             "knobs": KNOBS})
        elif self.path.startswith("/job/"):
            jid = self.path.split("/job/", 1)[1].strip("/")
            with lock:
                j = jobs.get(jid)
            self._send(200 if j else 404, j or {"error": "no such job"})
        else:
            self._send(404, {"error": "use POST /render, GET /job/<id>, GET /health"})

    def do_POST(self):
        if self.path != "/render":
            return self._send(404, {"error": "POST /render only"})
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception as exc:
            return self._send(400, {"error": f"bad json: {exc}"})
        try:
            abc  = data.get("abc")
            lyr  = data.get("lyrics")
            cap  = data.get("caption")
            seed = data.get("seed")
            dur  = data.get("duration_s")
            cot  = data.get("cot", "full")
            cfg  = data.get("cfg_scale")
            rnd  = data.get("seed_randomize", False)
            dry  = bool(data.get("dry_run", False))
            if not (isinstance(abc, str) and abc.strip()):
                raise ValueError("abc required (nonempty ABC notation)")
            if not (isinstance(lyr, str) and lyr.strip()):
                raise ValueError("lyrics required")
            if not (isinstance(cap, str) and cap.strip()):
                raise ValueError("caption required (style tags)")
            if cot not in {"melody", "full"}:
                raise ValueError("cot must be melody or full (abc needs it)")
            if seed is not None:
                validate_seed(seed)
            if dur is not None and not (isinstance(dur, (int, float)) and not isinstance(dur, bool)
                                        and 5 <= dur <= 240):
                raise ValueError("duration_s must be 5..240")
            cfg = validate_cfg(cfg)
            if not isinstance(rnd, bool):
                raise ValueError("seed_randomize must be true/false")
            abc_s, sem_s, budget_from = resolve_sampling_pair(
                data.get("abc_sampling"), data.get("semantic_sampling"), dur)
            seed_resolved, seed_source = resolve_seed(seed, rnd)
            guidance = cfg if cfg is not None else GUIDANCE_DEFAULT[cot]
            jid = data.get("id") or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            if not ID_RE.fullmatch(jid):
                raise ValueError("id must match " + ID_RE.pattern)
            spec = {"abc": abc, "lyrics": lyr, "caption": cap,
                    "seed": seed_resolved, "duration_s": dur, "cot": cot,
                    "cfg_scale": cfg, "abc_sampling": abc_s, "semantic_sampling": sem_s,
                    "seed_source": seed_source}
            window = {**SEMANTIC_DEFAULT, **(sem_s or {})}   # effective semantic window
            resolved = {"id": jid, "style": cap, "cot": cot, "seed": seed_resolved,
                        "seed_source": seed_source, "lyrics_chars": len(lyr),
                        "abc_chars": len(abc), "cfg_scale": cfg,
                        "cfg_scale_effective": guidance,
                        "cfg_branches_expected": 1 if guidance == 1 else 2,
                        "abc_sampling_effective": abc_s, "semantic_sampling_effective": sem_s,
                        "semantic_window_effective": window,
                        "forced_truncation_expected": window["min_tokens"] >= window["max_tokens"],
                        "token_budget_from": budget_from, "tokens_per_sec": TOKENS_PER_SEC,
                        "duration_s": dur, "dry_run": dry}
            if dry:
                return self._send(200, resolved)
            with lock:
                if jid in jobs:
                    raise ValueError(f"duplicate id {jid}")
                n = batch_n(data.get("batch"))
                jobs[jid] = {"id": jid, "status": "queued", "batch": n, "spec": spec,
                             "queued": time.time(),
                             "seed_source": seed_source,
                             "cfg_scale_effective": guidance,
                             "cfg_branches_expected": 1 if guidance == 1 else 2,
                             "semantic_window_effective": window,
                             "forced_truncation_expected": window["min_tokens"] >= window["max_tokens"],
                             "output_dir": str(OUT_ROOT / f"yue2-batch-{n}" / jid)}
                save_jobs()
                cv.notify_all()
            self._send(202, {"job_id": jid, "status": "queued",
                             "output_dir": jobs[jid]["output_dir"],
                             "seed": seed_resolved, "seed_source": seed_source,
                             "cfg_scale_effective": guidance,
                             "cfg_branches_expected": 1 if guidance == 1 else 2,
                             "semantic_sampling_effective": sem_s,
                             "abc_sampling_effective": abc_s,
                             "semantic_window_effective": window,
                             "forced_truncation_expected": window["min_tokens"] >= window["max_tokens"],
                             "token_budget_from": budget_from})
        except ValueError as exc:
            print(f"REJECT {self.client_address[0]}: {exc}", flush=True)
            self._send(400, {"error": str(exc)})

def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if JOBS_FILE.exists():                    # survive restart: requeue in-flight, keep done
        for jid, j in json.loads(JOBS_FILE.read_text()).items():
            if j["status"] in ("complete", "failed"):
                jobs[jid] = j
            else:
                j["status"] = "queued"; j.pop("started", None)
                if not (OUT_ROOT / f"yue2-batch-{j['batch']}" / jid / "result.json").exists():
                    part = OUT_ROOT / f"yue2-batch-{j['batch']}" / jid   # TRASH-ONLY law: never rm
                    if part.exists():
                        part.rename(Path("/home/jaita/trash") / f"yue2-partial-{int(time.time())}-{jid}")
                jobs[jid] = j
    threading.Thread(target=worker, daemon=True).start()
    print(f"yue2-server on 0.0.0.0:{PORT} budget={BUDGET}GiB mem_gate={MEM_GATE}GiB "
          f"out_root={OUT_ROOT} knobs=cfg_scale+sampling+seed_randomize", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()

if __name__ == "__main__":
    main()
