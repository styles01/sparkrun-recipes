#!/usr/bin/env python3
"""OpenAI /v1 shim over Cruz native ExLlamaV3 (chat.py Generator path).

Same knobs as scripts/exl3_native/tuning/run-qwen38-exl3.sh.
Parses Qwen3 XML tool calls into OpenAI tool_calls (desk patch).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

EXL3_ROOT = Path(os.environ.get("EXL3_ROOT", os.path.expanduser("~/exllamav3")))
sys.path.insert(0, str(EXL3_ROOT))
# chat_templates lives in exllamav3/examples/ — add it regardless of cwd
_examples = EXL3_ROOT / "examples"
if str(_examples) not in sys.path:
    sys.path.insert(0, str(_examples))

import torch  # noqa: E402
import threading as _threading

# Cumulative counters for the SparkDash /health contract (server-side fix first:
# expose backend/busy/token totals so the dashboard's exl3 probe engages).
_HL_LOCK = _threading.Lock()
_HL = {
    # reference-count of in-flight run_generate calls (entered, incl. those
    # still queued on GEN_LOCK) — busy = inflight > 0 (red-team C2 fix)
    "inflight": 0,
    "prompt_tokens_total": 0,
    "completion_tokens_total": 0,
    # SparkDash vLLM-parity telemetry (audit 2026-09-21):
    "requests_completed_total": 0,
    "requests_failed_total": 0,
    "mtp_accepted_tokens_total": 0,
    "mtp_drafted_tokens_total": 0,  # accepted + rejected (tokens proposed)
    "is_prefilling": False,         # engine is prefilling RIGHT NOW (no first token yet)
    "ttft_seconds_sum": 0.0,        # engine time_prefill = time to first token
    "ttft_seconds_count": 0,
    "e2e_seconds_sum": 0.0,         # time_prefill + time_generate
    "e2e_seconds_count": 0,
    "itl_seconds_sum": 0.0,         # time_generate over new_tokens-1 gaps
    "itl_seconds_count": 0,
    "context_last": 0,              # prompt tokens of the most recent request
}

def _hl_job_done(rec: dict[str, Any], failed: bool = False) -> None:
    """Fold one finished request's engine stats into the /health counters."""
    new_tokens = int(rec.get("new_tokens") or 0)
    tpre = float(rec.get("time_prefill") or 0.0)
    tgen = float(rec.get("time_generate") or 0.0)
    n_prompt = int(rec.get("prompt_tokens") or 0)
    dacc = int(rec.get("accepted_draft_tokens") or 0)
    drej = int(rec.get("rejected_draft_tokens") or 0)
    with _HL_LOCK:
        # inflight is decremented by run_generate's finally; never here.
        _HL["requests_failed_total" if failed else "requests_completed_total"] += 1
        _HL["is_prefilling"] = False
        _HL["context_last"] = n_prompt
        # MTP/speculative: drafted = every token the draft head proposed
        _HL["mtp_accepted_tokens_total"] += dacc
        _HL["mtp_drafted_tokens_total"] += dacc + drej
        # TTFT = engine time_prefill (time to first token)
        if tpre > 0:
            _HL["ttft_seconds_sum"] += tpre
            _HL["ttft_seconds_count"] += 1
        # E2E = prefill + decode
        e2e = tpre + tgen
        if e2e > 0:
            _HL["e2e_seconds_sum"] += e2e
            _HL["e2e_seconds_count"] += 1
        # ITL mean over the new_tokens-1 inter-token gaps of pure decode time
        if tgen > 0 and new_tokens > 1:
            _HL["itl_seconds_sum"] += tgen / (new_tokens - 1)
            _HL["itl_seconds_count"] += 1


def _ctx_len() -> int:
    """Best-effort context length for the /health contract."""
    try:
        for attr in ("max_sequence_length", "max_seq_len", "context_length", "max_position_embeddings"):
            v = getattr(CONFIG, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    except Exception:
        pass
    return 0

from exllamav3 import Generator, Job, model_init  # noqa: E402
from chat_templates import prompt_formats  # noqa: E402

FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)
SERVED = os.environ.get("SERVED_NAME", "Qwen3.8-Flash-Next-EXL3")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8899"))

GEN = None
TOKENIZER = None
CONFIG = None
PROMPT_FORMAT = None
STOP_IDS: list[Any] = []
GEN_LOCK = threading.Lock()


def parse_qwen_xml(text: str) -> list[dict[str, Any]]:
    calls = []
    for m in FN_RE.finditer(text or ""):
        name = m.group(1).strip()
        body = m.group(2)
        args: dict[str, Any] = {}
        for p in PARAM_RE.finditer(body):
            args[p.group(1).strip()] = p.group(2).strip()
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
    return calls


def split_prose_and_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Return (prose, calls): prose = text with tool-call XML stripped.
    vLLM keeps the model's prose alongside tool_calls in one turn; the shim
    previously dropped it (content=None), hiding narration from clients.
    Accepts both bare <function=...> blocks and canonical <tool_call>-wrapped calls;
    wrapper residue is stripped so clients never see tool syntax."""
    calls = parse_qwen_xml(text)
    if not calls:
        return text, []
    stripped = text
    for m in FN_RE.finditer(text):
        stripped = stripped.replace(m.group(0), "")
    # Wrapper residue (opener/closer fragments from multi-call turns) never
    # belongs in client-visible prose.
    import re as _re
    stripped = _re.sub(r"</?tool_(?:call|response)>", "", stripped)
    prose = stripped.strip()
    return prose, calls


def messages_to_context(messages: list[dict[str, Any]]) -> tuple[str, list[tuple[str, str | None]]]:
    """Canonical Qwen tool-format history rendering (matches tokenizer chat_template):
    assistant calls: content + "\n\n<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n<tool_call>"
    tool results:    <tool_call>\nresult\n<tool_call>  inside a user turn, consecutive results merged.
    """
    system = ""
    context: list[tuple[str, str | None]] = []
    pending_user: str | None = None
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                (c.get("text") or "") if isinstance(c, dict) else str(c) for c in content
            )
        if role == "system":
            system = (system + "\n" + content).strip() if system else content.strip()
        elif role == "user":
            if pending_user is not None:
                context.append((pending_user, None))
            pending_user = content
        elif role == "assistant":
            # Re-render the model's own tool_calls in its canonical <tool_call>-wrapped XML.
            # Dropping them made the model amnesiac about its own calls: empty
            # assistant turns, identical re-calls, narrate-then-stop stalls.
            text = "<tool_call>\n\n<tool_call>\n\n" + (content or "")
            first = True
            for c in msg.get("tool_calls") or []:
                if not isinstance(c, dict):
                    continue
                fn = c.get("function") or {}
                name = fn.get("name") or "tool"
                raw = fn.get("arguments")
                if isinstance(raw, str):
                    try:
                        args = json.loads(raw)
                    except Exception:
                        args = {"args": raw} if raw else {}
                elif isinstance(raw, dict):
                    args = raw
                else:
                    args = {}
                opener = "<tool_call>\n<function=%s>\n" % name if first else "\n<tool_call>\n<function=%s>\n" % name
                if (content or "").strip():
                    opener = ("\n\n" + opener) if first else opener
                params = "".join(
                    "<parameter=%s>\n%s\n</parameter>\n"
                    % (k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                    for k, v in args.items()
                )
                text += opener + params + "</function>\n<tool_call>"
                first = False
            if pending_user is None:
                pending_user = ""
            context.append((pending_user, text))
            pending_user = None
        elif role == "tool":
            # Canonical result block: generic <tool_call> wrapper, NO function name,
            # consecutive results merged into one user turn.
            tool_txt = "<tool_call>\n%s\n<tool_call>" % content
            if context and context[-1][1] is not None:
                context.append((tool_txt, None))
            elif pending_user is not None:
                pending_user += "\n" + tool_txt
            elif context:
                context[-1] = (context[-1][0] + tool_txt, None)
            else:
                pending_user = tool_txt
    if pending_user is not None:
        context.append((pending_user, None))
    if not context:
        context = [("Hello", None)]
    return system, context


def sampler_from_body(body: dict[str, Any]):
    from exllamav3.generator.sampler import ComboSampler

    temp = float(body.get("temperature") if body.get("temperature") is not None else 0.0)
    top_p = float(body.get("top_p") if body.get("top_p") is not None else 1.0)
    top_k = int(body.get("top_k") if body.get("top_k") is not None else (1 if temp <= 0 else 0))
    return ComboSampler(
        rep_p=1.0,
        pres_p=0.0,
        freq_p=0.0,
        rep_sustain_range=1024,
        rep_decay_range=1024,
        temperature=max(temp, 0.0),
        min_p=0.0 if temp <= 0 else 0.08,
        top_k=top_k,
        top_p=top_p,
        temp_last=True,
        adaptive_target=1.0,
        adaptive_decay=0.9,
    )


@torch.inference_mode()
def run_generate(
    *,
    input_ids,
    max_new_tokens: int,
    sampler,
    stop_conditions: list[Any],
    on_chunk=None,
) -> dict[str, Any]:
    """Same thread + inference_mode as Cruz chat.py. One job at a time."""
    assert GEN is not None
    with _HL_LOCK:
        _HL["inflight"] += 1
    client_gone = False
    try:
        ident = uuid.uuid4().hex
        job = Job(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_conditions=stop_conditions,
            sampler=sampler,
            identifier=ident,
        )
        text = ""
        last: dict[str, Any] = {}
        first_tok_wall = None
        req_t0 = time.perf_counter()
        _nt_prev = 0
        try:
            with GEN_LOCK:
                GEN.enqueue(job)
                # prefill runs immediately after enqueue (engine thread): report prompt
                # tokens NOW so SparkDash's poll-diff shows the prefill spike at the
                # START of generation (not the end) — same fix vLLM's Prometheus path got.
                with _HL_LOCK:
                    try:
                        _n_prompt = int(input_ids.shape[-1])
                    except Exception:
                        _n_prompt = int(getattr(input_ids, "size", lambda d=-1: 0)(-1)) or len(input_ids)
                    _HL["prompt_tokens_total"] += _n_prompt
                    _HL["is_prefilling"] = True
                while GEN.num_remaining_jobs():
                    for r in GEN.iterate():
                        if r.get("identifier") != ident:
                            continue
                        chunk = r.get("text") or ""
                        if chunk:
                            if first_tok_wall is None:
                                # TTFT wall-clock fallback (incl. queue wait); engine
                                # time_prefill stays primary for ttft_seconds_sum.
                                first_tok_wall = time.perf_counter() - req_t0
                                with _HL_LOCK:
                                    _HL["is_prefilling"] = False
                            text += chunk
                            if on_chunk is not None:
                                try:
                                    on_chunk(chunk)
                                except Exception:
                                    # client hung up mid-stream (red-team M4):
                                    # stop pushing chunks but keep draining the
                                    # job so the engine stays clean and stats fold.
                                    client_gone = True
                                    on_chunk = None
                        # vLLM parity: completion counter climbs DURING decode so the
                        # dashboard's poll-diff shows the real-time decode rate (not a
                        # bulk spike when the job finishes). Each request adds only
                        # ITS OWN delta so concurrent/queued jobs cannot clobber
                        # each other's contribution (red-team C1).
                        try:
                            _nt_live = int(getattr(job, "new_tokens", -1))
                        except Exception:
                            _nt_live = -1
                        if _nt_live > _nt_prev:
                            with _HL_LOCK:
                                _HL["completion_tokens_total"] += _nt_live - _nt_prev
                            _nt_prev = _nt_live
                        if r.get("eos"):
                            last = r
        except Exception:
            # Engine-side failure: contain, count as failed, re-raise to the caller.
            _hl_job_done({
                "new_tokens": int(getattr(job, "new_tokens", 0) or 0) if client_gone is False else 0,
                "time_prefill": 0.0,
                "time_generate": 0.0,
                "prompt_tokens": 0,
                "accepted_draft_tokens": 0,
                "rejected_draft_tokens": 0,
            }, failed=True)
            raise
        dacc = int(last.get("accepted_draft_tokens") or 0)
        drej = int(last.get("rejected_draft_tokens") or 0)
        # Engine "time_prefill" = time to first token (TTFT); "time_generate" =
        # time to last token. Prefer engine numbers; fall back to wall clock.
        tpre = float(last.get("time_prefill") or 0.0)
        if tpre <= 0 and first_tok_wall is not None:
            tpre = first_tok_wall
        tgen = float(last.get("time_generate") or 0.0)
        new_tokens = int(last.get("new_tokens") or 0)
        n_prompt = int(last.get("prompt_tokens") or 0) or int(input_ids.shape[-1])
        _res = {
            "text": text,
            "prompt_tokens": n_prompt,
            "new_tokens": new_tokens,
            "eos_reason": last.get("eos_reason") or "stop",
            "accepted_draft_tokens": dacc,
            "rejected_draft_tokens": drej,
            "time_generate": tgen,
            "time_prefill": tpre,
            "decode_tok_s": (new_tokens / tgen) if tgen > 0 and new_tokens else 0.0,
            "draft_accept": (dacc / (dacc + drej)) if (dacc + drej) else None,
        }
        _hl_job_done(_res, failed=client_gone)
        # Fold any tail tokens the last iterate() loop missed (covers the gap
        # between the final poll and EOS; incremental so it composes).
        if new_tokens > _nt_prev:
            with _HL_LOCK:
                _HL["completion_tokens_total"] += new_tokens - _nt_prev
        return _res
    finally:
        with _HL_LOCK:
            _HL["inflight"] -= 1


def build_ids(system: str, context: list[tuple[str, str | None]], think: bool):
    assert TOKENIZER is not None and PROMPT_FORMAT is not None
    frm = PROMPT_FORMAT.format(system, context, think)
    add_bos = PROMPT_FORMAT.add_bos()
    return TOKENIZER.encode(frm, add_bos=add_bos, encode_special_tokens=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, obj: Any, extra_headers: list[tuple[str, str]] | None = None) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for k, v in extra_headers or []:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/v1/models", "/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": SERVED,
                            "object": "model",
                            "owned_by": "vcruz305-exllamav3",
                            "max_model_len": int(os.environ.get("CS", "262144")),
                        }
                    ],
                },
            )
            return
        if path in ("/health", "/v1/health"):
            with _HL_LOCK:
                hl = dict(_HL)
            # Live cache stats: KV residency + prefix hit rate (fail-open to null).
            # KV gauge = (used + cached) / max — pages holding reusable prefixes
            # are real cache occupancy even with no active job (red-team M3).
            cs = None
            try:
                cs = GEN.get_cache_stats()
            except Exception:
                cs = None
            if cs:
                max_tok = int(cs.get("max_tokens") or 0)
                used_tok = int(cs.get("used_tokens") or 0)
                cached_tok = int(cs.get("cached_tokens") or 0)
                hl["kv_cache_usage"] = (
                    round(min(1.0, (used_tok + cached_tok) / max_tok), 4) if max_tok > 0 else None
                )
                hr = cs.get("hit_rate")
                hl["prefix_cache_hit_rate"] = (
                    round(float(hr), 4) if isinstance(hr, (int, float)) else None
                )
                # Waiting = requests inside run_generate but not yet taken by the
                # engine (queued on GEN_LOCK) — engine pending_jobs alone misses
                # them (red-team m3).
                engine_jobs = int(cs.get("active_jobs") or 0) + int(cs.get("pending_jobs") or 0)
                hl["requests_waiting"] = max(0, int(hl["inflight"]) - engine_jobs)
            else:
                hl["kv_cache_usage"] = None
                hl["prefix_cache_hit_rate"] = None
                hl["requests_waiting"] = max(0, int(hl["inflight"]))
            # GPU memory utilization: real device census (weights + cache + ctx),
            # same semantics as vLLM's gauge / DS4's unified_device census.
            try:
                _free, _total = torch.cuda.mem_get_info(0)
                hl["gpu_memory_utilization"] = round(1.0 - (_free / _total), 4) if _total > 0 else None
            except Exception:
                try:
                    _mi = open("/proc/meminfo").read()
                    _mt = re.search(r"MemTotal:\s+(\d+) kB", _mi)
                    _ma = re.search(r"MemAvailable:\s+(\d+) kB", _mi)
                    hl["gpu_memory_utilization"] = (
                        round(1.0 - int(_ma.group(1)) / int(_mt.group(1)), 4)
                        if _mt and _ma and int(_mt.group(1)) > 0 else None
                    )
                except Exception:
                    hl["gpu_memory_utilization"] = None
            self._send(200, {
                "status": "ok",
                "engine": "exllamav3-native",
                # SparkDash exl3 contract (LlmProbe._healthLooksLikeExl3 + _applyExl3Health):
                "backend": "exl3",
                "busy": hl["inflight"] > 0,
                "is_prefilling": bool(hl.get("is_prefilling")),
                "context_length": _ctx_len(),
                "prompt_tokens_total": hl["prompt_tokens_total"],
                "completion_tokens_total": hl["completion_tokens_total"],
                # vLLM-parity telemetry (see runbook audit table):
                "requests_completed_total": hl["requests_completed_total"],
                "requests_failed_total": hl["requests_failed_total"],
                "requests_waiting": hl["requests_waiting"],
                "mtp_accepted_tokens_total": hl["mtp_accepted_tokens_total"],
                "mtp_drafted_tokens_total": hl["mtp_drafted_tokens_total"],
                "ttft_seconds_sum": hl["ttft_seconds_sum"],
                "ttft_seconds_count": hl["ttft_seconds_count"],
                "e2e_seconds_sum": hl["e2e_seconds_sum"],
                "e2e_seconds_count": hl["e2e_seconds_count"],
                "itl_seconds_sum": hl["itl_seconds_sum"],
                "itl_seconds_count": hl["itl_seconds_count"],
                "context_last": hl["context_last"],
                "kv_cache_usage": hl["kv_cache_usage"],
                "prefix_cache_hit_rate": hl["prefix_cache_hit_rate"],
                "gpu_memory_utilization": hl["gpu_memory_utilization"],
            })
            return
        self._send(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": {"message": "bad json", "type": "invalid_request_error"}})
            return
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        try:
            self._chat(body)
        except Exception as exc:  # noqa: BLE001
            # Red-team M5: count handler failures so requests_failed_total moves.
            with _HL_LOCK:
                _HL["requests_failed_total"] += 1
            try:
                self._send(500, {"error": {"message": str(exc), "type": "server_error"}})
            except Exception:
                pass

    def _chat(self, body: dict[str, Any]) -> None:
        messages = body.get("messages") or []
        tools = body.get("tools") or []
        think_kw = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
        effort = body.get("reasoning_effort")
        if effort is None:
            effort = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
        if think_kw is not None:
            think = bool(think_kw)
        elif effort is not None:
            # Hermes sends reasoning_effort, not enable_thinking. Honor it: only an
            # explicit low-effort request runs this thinking model non-thinking.
            think = str(effort).lower() not in ("none", "minimal", "off", "disabled")
        else:
            think = False
        system, context = messages_to_context(messages)
        if tools:
            # Canonical Qwen tool system block (matches the model chat_template):
            # schemas as JSON inside <tools>, calls wrapped in <tool_call>...<tool_call>.
            tool_json = "\n".join(
                json.dumps(t, ensure_ascii=False) for t in tools if isinstance(t, dict)
            )
            system = (
                (system + "\n\n") if system else ""
            ) + (
                "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
                + tool_json
                + "\n</tools>\n\n"
                "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                + "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                + "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
                + "that can span\nmultiple lines\n</parameter>\n</function>\n<tool_call>\n\n"
                + "<IMPORTANT>\n"
                + "Reminder:\n"
                + "- Function calls MUST follow the specified format: an inner <function=...></function> block "
                + "must be nested within <tool_call><tool_call> XML tags\n"
                + "- Required parameters MUST be specified\n"
                + "- You may provide optional reasoning for your function call in natural language BEFORE "
                + "the function call, but NOT after\n"
                + "If there is no function call available, answer the question like normal with your "
                + "current knowledge and do not tell the user about function calls\n"
                + "</IMPORTANT>"
            )
        ids = build_ids(system or PROMPT_FORMAT.default_system_prompt(think), context, think)
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
        max_new = max(1, min(max_new, 65536))
        sampler = sampler_from_body(body)
        stops = list(STOP_IDS)
        if body.get("ignore_eos"):
            stops = []
        stream = bool(body.get("stream"))
        t0 = time.perf_counter()
        if stream:
            self._stream_sse(ids, max_new, sampler, stops, bool(tools), t0, body)
            return
        rec = run_generate(
            input_ids=ids,
            max_new_tokens=max_new,
            sampler=sampler,
            stop_conditions=stops,
        )
        text = rec["text"]
        prose, calls = split_prose_and_calls(text) if tools else (text, [])
        finish = "tool_calls" if calls else ("length" if rec["eos_reason"] == "max_new_tokens" else "stop")
        msg: dict[str, Any] = {"role": "assistant", "content": prose if calls else text}
        if calls:
            msg["tool_calls"] = calls
        prompt_tokens = rec["prompt_tokens"] or int(ids.shape[-1])
        completion = rec["new_tokens"]
        print(
            f"gen decode={rec['decode_tok_s']:.1f} tok/s draft_accept={rec['draft_accept']} "
            f"new={completion} prefill_s={rec['time_prefill']:.2f}",
            flush=True,
        )
        self._send(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model") or SERVED,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion,
                    "total_tokens": prompt_tokens + completion,
                    "decode_tok_s": round(rec["decode_tok_s"], 2),
                    "draft_accept": rec["draft_accept"],
                },
            },
        )

    def _stream_sse(self, ids, max_new, sampler, stops, tools: bool, t0: float, body: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        held: list[str] = []

        def on_chunk(chunk: str) -> None:
            if tools:
                held.append(chunk)
                return
            obj = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(t0),
                "model": SERVED,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        rec = run_generate(
            input_ids=ids,
            max_new_tokens=max_new,
            sampler=sampler,
            stop_conditions=stops,
            on_chunk=on_chunk,
        )
        try:
            acc = rec["text"]
            calls = parse_qwen_xml(acc) if tools else []
            finish = "tool_calls" if calls else ("length" if rec["eos_reason"] == "max_new_tokens" else "stop")
            usage = {
                "prompt_tokens": rec["prompt_tokens"],
                "completion_tokens": rec["new_tokens"],
                "total_tokens": rec["prompt_tokens"] + rec["new_tokens"],
                "decode_tok_s": round(rec["decode_tok_s"], 2),
                "draft_accept": rec["draft_accept"],
            }
            if calls:
                prose, _ = split_prose_and_calls(acc)
                if prose:
                    pre = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": int(t0),
                        "model": SERVED,
                        "choices": [{"index": 0, "delta": {"content": prose}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(pre)}\n\n".encode())
                obj = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(t0),
                    "model": SERVED,
                    "choices": [{"index": 0, "delta": {"tool_calls": calls, "content": None}, "finish_reason": finish}],
                    "usage": usage,
                }
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            else:
                if tools and acc:
                    obj = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": int(t0),
                        "model": SERVED,
                        "choices": [{"index": 0, "delta": {"content": acc}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                obj = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(t0),
                    "model": SERVED,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": usage,
                }
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # client left mid-stream: run_generate already folded the stats as
            # failed — nothing to count here.
            pass
        print(
            f"stream decode={rec['decode_tok_s']:.1f} tok/s draft_accept={rec['draft_accept']} "
            f"new={rec['new_tokens']}",
            flush=True,
        )


def load_engine(ns: argparse.Namespace) -> None:
    global GEN, TOKENIZER, CONFIG, PROMPT_FORMAT, STOP_IDS
    PROMPT_FORMAT = prompt_formats["qwen35"]("User", "Assistant")
    print("loading native exllamav3", ns.model_dir, flush=True)
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(ns)
    CONFIG = config
    TOKENIZER = tokenizer
    print(
        f"mtp={bool(ns.mtp)} ndt={ns.num_draft_tokens} dds={ns.dynamic_draft} dc={ns.draft_confidence} "
        f"cq={ns.cache_quant} cs={ns.cache_size} draft_model={draft_model is not None} "
        f"batch={ns.autosplit_max_batch_size}",
        flush=True,
    )
    GEN = Generator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        draft_model=draft_model,
        draft_cache=draft_cache,
        num_draft_tokens=ns.num_draft_tokens,
        ngram_match_min=ns.ngram_match_min,
        dynamic_draft_tokens=ns.dynamic_draft,
        draft_confidence=ns.draft_confidence,
        cpu_cache_size=int(ns.cpu_cache_size * 1024**3),
        recurrent_cache_size=int(ns.recurrent_cache_size * 1024**3),
        max_chunk_size=2048,
    )
    stops = [sc for sc in PROMPT_FORMAT.stop_conditions(tokenizer) if sc]
    if config.eos_token_id_list and all(config.eos_token_id_list):
        stops += config.eos_token_id_list
    # qwen35 stop_conditions only list im_end; agents can emit im_start as the whole reply.
    try:
        sid = tokenizer.single_id("<|im_start|>")
        if sid is not None:
            stops.append(sid)
    except Exception:
        pass
    stops.append("<|im_start|>")
    STOP_IDS = stops
    print("native engine ready", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(
        parser,
        cache=True,
        add_sampling_args=True,
        add_draft_model_args=True,
        default_cache_size=262144,
        default_autosplit_max_batch_size=1,
    )
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    load_engine(args)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening {args.host}:{args.port}/v1", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
