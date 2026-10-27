"""A minimal OpenAI-compatible chat server for Flash-Next on the board (a test service, not server/app.py).

    PYTHONPATH=<repo>:~/fn-ref/site python tools/fn_serve.py --pack ~/fn2/packs/3.05 --port 8000

One engine, one sequence at a time (requests queue on a lock). Endpoints:

  GET  /health                 status, version ("flash-next-3.05+<git sha>"), speculation settings, the
                               last request's numbers
  GET  /v1/models              the one model ("flash-next-3.05"; any model name in a request is accepted)
  POST /v1/chat/completions    messages through the pack's chat template; `stream` (SSE chunks) or one
                               JSON answer; `max_tokens` (default 2048), `stop`, `temperature`, `top_p`,
                               `chat_template_kwargs.enable_thinking` (default false)

Greedy requests (temperature 0 or absent) decode with MTP speculation (`StairCut` over a 4-deep MTP
chain, the hot-vocabulary draft head): lossless, the text is plain greedy decoding's token for token
(tools/fn_spec_check.py). A request with temperature > 0 samples from the plain decode step (no
speculation). A conversation that extends the previous request's tokens reuses the state (only the new
tokens are prefilled); anything else starts over. Contexts past 2,051 tokens run the QSA selection on
the kernels (ENG-147); the prompt and the MTP layer's catch-up run on the prefill kernels (ENG-225,
about 1,400 tokens a second at 8k).
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.flashnext.engine import DecodeGraph, FlashNextEngine  # noqa: E402
from engine.models.flashnext.load import load  # noqa: E402
from engine.methods.staircut import StairCut  # noqa: E402
from engine.models.flashnext.serving import prices  # noqa: E402
from tools.fn_specloop import MTPGraphDrafter, SpecDecoder  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "flash-next-3.05"


class Recorder(MTPGraphDrafter):
    """The MTP drafter that also keeps every committed row's streams in the server's history (a later
    request that reuses the state primes the MTP layer from it)."""

    def __init__(self, srv, *a, **kw):
        super().__init__(*a, **kw)
        self.srv = srv

    def accepted(self, stack, nexts):
        super().accepted(stack, nexts)
        L = self.srv.st.length
        self.srv.hist[L - stack.shape[0]:L].copy_(stack)


class Server:
    def __init__(self, a):
        from transformers import AutoTokenizer
        t0 = time.time()
        idx, W, banks, tables, timing = load(a.pack, "cuda", mtp=True)
        eng = self.eng = FlashNextEngine(idx.cfg, W, tables, banks, device="cuda", max_len=a.max_len)
        self.tok = AutoTokenizer.from_pretrained(a.pack)
        self.eos = set(idx.cfg.eos_token_ids)
        if a.hot:
            eng.set_draft_head(np.load(os.path.join(ROOT, "ops", "fn-hot-vocab.npy"))[:a.hot])
        sc = eng.self_check(self.chat_ids([{"role": "user", "content": "Count from one to twenty in words."}], False),
                            log=None)
        self.spec_ok = bool(sc["ok"])
        self.st = eng.new_state(a.max_len)
        self.dg = DecodeGraph(eng, self.st)
        self.cut = StairCut(prices(a.pack), siblings=False)
        self.dr = Recorder(self, eng, a.depth, a.max_len, cut=self.cut)
        self.sd = SpecDecoder(eng, self.st, self.dr, graphs=True)
        c = idx.cfg
        self.hist = torch.zeros(a.max_len, c.hc_count * c.hidden_size, dtype=torch.bfloat16, device="cuda")
        self.ids: list[int] = []          # the tokens the trunk has run, in order (the state's length)
        self.lock = threading.Lock()
        self.busy = False
        self.max_len = a.max_len
        self.last: dict = {}
        # the commit: a checkout's own, or .fn-sha beside a copied tree
        sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        if not sha and os.path.exists(os.path.join(ROOT, ".fn-sha")):
            sha = open(os.path.join(ROOT, ".fn-sha")).read().strip()
        self.version = f"{MODEL}+{sha or 'unknown'}"
        # warm-up: the graphs a short greedy request uses (decode, verify, commit, MTP draft)
        self.generate(self.chat_ids([{"role": "user", "content": "Say hello in three languages."}], False),
                      64, 0.0, 1.0, lambda t: True)
        self.ids = []
        self.info = {"pack": a.pack, "load_s": round(time.time() - t0, 1), "load": timing, "depth": a.depth,
                     "draft_head": f"hot {a.hot}" if a.hot else "full", "max_len": a.max_len,
                     "self_check": sc, "warm_up": self.last}

    def chat_ids(self, messages, thinking: bool) -> list[int]:
        s = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking)
        return self.tok(s, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------------------------ the sequence
    @torch.no_grad()
    def prepare(self, ids: list[int]) -> tuple[torch.Tensor, int]:
        """Bring the state to `ids` (reusing it when it holds a prefix of them); the last row's logits."""
        eng, st, dr = self.eng, self.st, self.dr
        reuse = 0 < len(self.ids) < len(ids) and ids[:len(self.ids)] == self.ids
        if not reuse:
            eng.reset(st)
            self.ids = []
            dr.st.length = 0
        rest = ids[len(self.ids):]
        logits = eng.prefill(torch.tensor(rest), st, keep_stack=True)
        self.hist[len(self.ids):len(ids)].copy_(eng.prompt_stack)
        self.ids = list(ids)
        L, m = len(ids), dr.st.length
        if self.spec_ok and L - 1 > m:            # the MTP layer catches up over the new positions
            eng.mtp_prime(self.hist[m:L - 1], ids[m + 1:L], dr.st)
        return logits, reuse

    def generate(self, ids: list[int], max_tokens: int, temperature: float, top_p: float, emit) -> dict:
        eng, st = self.eng, self.st
        t0 = time.perf_counter()
        logits, reuse = self.prepare(ids)
        t1 = time.perf_counter()
        max_tokens = min(max_tokens, self.max_len - len(ids) - 40)
        stats = {"prompt_tokens": len(ids), "prefill_s": round(t1 - t0, 3), "reused": reuse}
        if temperature <= 0 and self.spec_ok:
            first = int(logits[0].argmax())
            self.dr.pending = (self.hist[len(ids) - 1:len(ids)], [first])
            self.sd.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in self.sd.stats.items()}
            out = self.sd.generate(first, max_tokens, self.eos, emit=emit)
            self.ids = (ids + self.sd.full)[:st.length]
            s = self.sd.stats
            stats.update(mode="greedy, MTP speculation (lossless)", rounds=s["rounds"],
                         tokens_a_round=round((len(out) - 1) / max(s["rounds"], 1), 2))
        else:
            g = torch.Generator(device="cuda").manual_seed(int(time.time() * 1e6) & 0xFFFFFFFF)

            def sample(lg):
                if temperature <= 0:
                    return int(lg.argmax())
                p = torch.softmax(lg.float() / temperature, -1)
                if top_p < 1.0:
                    v, i = torch.sort(p, descending=True)
                    keep = torch.cumsum(v, 0) - v < top_p
                    p = torch.zeros_like(p).scatter_(0, i[keep], v[keep])
                return int(torch.multinomial(p, 1, generator=g))
            out = [sample(logits[0])]
            go = emit([out[0]]) is not False
            while go and len(out) < max_tokens and out[-1] not in self.eos:
                lg = self.dg(out[-1])
                self.hist[st.length - 1:st.length].copy_(self.dg.stacks[eng.qsa_class(st.length - 1)])
                out.append(sample(lg[0]))
                go = emit([out[-1]]) is not False
            self.ids = (ids + out)[:st.length]
            stats.update(mode=f"sampled, temperature {temperature}, top_p {top_p} (plain decode)")
        dt = time.perf_counter() - t1
        stats.update(completion_tokens=len(out), decode_s=round(dt, 3),
                     tok_s=round((len(out) - 1) / dt, 2) if len(out) > 1 and dt > 0 else None)
        self.last = stats
        return stats


def build_app(srv: Server):
    # no postponed annotations in this module: FastAPI resolves `req: Request` by its object here
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL, "version": srv.version, "engine": "TandemLLM Flash-Next (tools/fn_serve.py)",
                "engine_busy": srv.busy, "speculation": {"on": srv.spec_ok, "drafter": "MTP chain + StairCut",
                                                          "lossless": "greedy requests"},
                **srv.info, "last_request": srv.last}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "tandemllm"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        kw = body.get("chat_template_kwargs") or {}
        thinking = bool(kw.get("enable_thinking", body.get("enable_thinking", False)))
        ids = srv.chat_ids(body.get("messages") or [], thinking)
        if len(ids) + 64 > srv.max_len:
            return JSONResponse({"error": {"message": f"prompt of {len(ids)} tokens exceeds max_len {srv.max_len}"}},
                                status_code=400)
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
        temperature = float(body.get("temperature") or 0.0)
        top_p = float(body.get("top_p") or 1.0)
        stop = body.get("stop") or []
        stop = [stop] if isinstance(stop, str) else list(stop)
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        import queue
        q: queue.Queue = queue.Queue()
        state = {"toks": [], "text": "", "sent": 0, "stopped": False}

        def emit(new):
            st = state
            new = [t for t in new if t not in srv.eos]
            st["toks"] += new
            text = srv.tok.decode(st["toks"], skip_special_tokens=True)
            cut = min([text.find(s) for s in stop if s and s in text] or [-1])
            if cut >= 0:
                text = text[:cut]
                st["stopped"] = True
            if text.endswith("�"):                        # an unfinished UTF-8 sequence
                return not st["stopped"]
            delta = text[st["sent"]:]
            st["sent"] = len(text)
            st["text"] = text
            if delta:
                q.put(delta)
            return not st["stopped"]

        def run():
            with srv.lock:
                srv.busy = True
                try:
                    stats = srv.generate(ids, max_tokens, temperature, top_p, emit)
                    q.put(("done", stats))
                except Exception as e:                   # noqa: BLE001
                    q.put(("error", repr(e)))
                finally:
                    srv.busy = False
        threading.Thread(target=run, daemon=True).start()

        def finish_reason(stats):
            return "stop" if state["stopped"] or stats.get("completion_tokens", 0) < max_tokens else "length"

        if body.get("stream"):
            def sse():
                head = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL}
                yield f"data: {json.dumps(head | {'choices': [{'index': 0, 'delta': {'role': 'assistant'}}]})}\n\n"
                while True:
                    item = q.get()
                    if isinstance(item, tuple):
                        kind, info = item
                        if kind == "error":
                            yield f"data: {json.dumps({'error': {'message': info}})}\n\n"
                        else:
                            yield ("data: " + json.dumps(head | {"choices": [{"index": 0, "delta": {},
                                   "finish_reason": finish_reason(info)}], "usage": {
                                   "prompt_tokens": info["prompt_tokens"], "completion_tokens": info["completion_tokens"],
                                   "total_tokens": info["prompt_tokens"] + info["completion_tokens"]},
                                   "x_flashnext": info}) + "\n\n")
                        yield "data: [DONE]\n\n"
                        return
                    yield f"data: {json.dumps(head | {'choices': [{'index': 0, 'delta': {'content': item}}]})}\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        import asyncio
        while True:
            item = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if isinstance(item, tuple):
                break
        kind, info = item
        if kind == "error":
            return JSONResponse({"error": {"message": info}}, status_code=500)
        return {"id": rid, "object": "chat.completion", "created": created, "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": state["text"]},
                             "finish_reason": finish_reason(info)}],
                "usage": {"prompt_tokens": info["prompt_tokens"], "completion_tokens": info["completion_tokens"],
                          "total_tokens": info["prompt_tokens"] + info["completion_tokens"]},
                "x_flashnext": info}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=40960)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--hot", type=int, default=32768)
    a = ap.parse_args()
    import uvicorn
    srv = Server(a)
    print(f"[fn-serve] {srv.version} ready on :{a.port} ({srv.info['load_s']} s)", flush=True)
    uvicorn.run(build_app(srv), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
