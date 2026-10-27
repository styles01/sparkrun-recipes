"""Record what the block drafter has to learn: the target's own hidden states, and what the
target then said.

The drafter is a conditional model of ONE distribution -- the target's greedy next token, given the
five residual streams the target leaves behind. Everything it needs can therefore be written down
once, offline, and read back without the target being resident. That is the whole reason a 27 B
target can be distilled into a 1.9 B drafter on a 121 GB board: training never runs the target.

WHAT A SAMPLE IS
----------------
One sequence of `N` tokens, and for every position `i` of it:

    fused[i]    the five tap tensors of `engine/model.py`, concatenated: [N, 5 * 5120] bf16.
                Tap invocation j is the residual stream ENTERING target layer j, so the five the
                drafter wants are invocations 5, 19, 33, 47, 61 -- see `drafters/dflash2.py`.
    label[i]    argmax of the target's logits at i, i.e. the token the target would emit next.
    top_ids[i]  the 64 most likely next tokens, and
    top_lp[i]   their log-probabilities. The block objective uses the argmax; the top-64 is what
                makes the soft term possible without storing a 248,320-wide row per position.

TWO KINDS OF SEQUENCE, AND WHY BOTH
-----------------------------------
`gen`     a prompt this tool built, followed by the target's OWN greedy continuation. This is the
          serving distribution exactly: the drafter at serve time conditions on hidden states of
          text the target itself produced. It is expensive -- 256 tokens at the engine's own decode
          rate -- so it is the smaller half.
`corpus`  a window of public text, teacher-forced in one prefill. The labels are still the target's
          argmax, not the corpus's next token, so the sample is still about the target; what is
          borrowed from the corpus is only the CONDITIONING text. At ~850 tok/s this is the bulk.

THE PROMPT SET
--------------
The bench row this program is measured against draws single-turn chat prompts of 256 tokens across
eleven topics, thinking off, temperature 0. This tool mirrors that SHAPE and nothing else: its
prompts are built here, from templates written in this file plus passages drawn from the public
corpus already on the board. The row's own prompts are never read, never tokenised and never
generated from. A drafter trained on the test set would report an acceptance nobody else could
reproduce, which is the same failure the ledger's 09:45 entry records for the lookup drafter's
corpus.

The held-out split is by sequence, declared in `manifest.json` and never crossed by the trainer:
`--holdout` of the `gen` sequences are marked `split: "heldout"` and are the only sequences the
acceptance gate is allowed to read, because only a self-generated sequence carries a greedy
continuation the simulation can walk.

Nothing personal is read. The corpus is public text (an encyclopedia dump and permissively licensed
Python), and the prompts are written in this file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.drafters.dflash2 import DFlash2Drafter  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import generate_spec  # noqa: E402

TAP_LAYERS = (5, 19, 33, 47, 61)

# ----------------------------------------------------------------------------- prompt templates
#
# Eleven topics, the same eleven the row's dataset card names, written here. Each entry is
# (topic, kind, template). `kind` says what the template needs: "plain" is self-contained,
# "en"/"de"/"code" ask for a passage from that part of the corpus.

TEMPLATES: list[tuple[str, str, str]] = [
    ("business", "plain",
     "A small hardware company sells one product and is deciding whether to add a second. "
     "Write a short memo for its founder that lays out the three questions that decide it, "
     "what evidence would answer each one, and what a wrong answer costs."),
    ("business", "plain",
     "Explain, for someone who has never run a team, what a quarterly plan is for, why most of "
     "them fail, and what a good one looks like. Use plain language and no jargon."),
    ("everyday", "plain",
     "My kitchen sink drains slowly and smells faintly of damp. Walk me through what to check, "
     "in order, from the cheapest thing to the most expensive, and tell me when to stop and "
     "call somebody."),
    ("everyday", "plain",
     "I want to start cycling to work, eleven kilometres each way, and I have not been on a bike "
     "in ten years. Tell me how to build up to it over six weeks and what will actually go wrong."),
    ("creative", "plain",
     "Write the opening of a short story about a lighthouse keeper who starts receiving letters "
     "addressed to somebody who died forty years ago. Third person, present tense, no dialogue "
     "in the first paragraph."),
    ("creative", "en",
     "Read this passage and then continue it for three paragraphs in the same register, keeping "
     "its subject and its tone:\n\n{passage}"),
    ("history", "en",
     "Summarise the following passage for a reader who knows nothing about the subject, then say "
     "what it leaves out:\n\n{passage}"),
    ("history", "plain",
     "Explain why the spread of printing changed what counted as knowledge in Europe, and name "
     "the two things about that change that are usually overstated."),
    ("science", "en",
     "Explain the following passage to a first-year undergraduate, then list the three terms in "
     "it that would trip them up and define each one:\n\n{passage}"),
    ("science", "plain",
     "Explain what memory bandwidth is, why it is a different thing from memory capacity, and "
     "why a program can be limited by one while having plenty of the other."),
    ("law", "plain",
     "In plain language, explain the difference between a warranty and a guarantee in a consumer "
     "contract, and give three examples where the distinction changes the outcome."),
    ("law", "plain",
     "A freelancer's contract says work is 'delivered on acceptance' but does not define "
     "acceptance. Explain the ways that can go wrong for each side and what a fair clause "
     "would say instead."),
    ("math", "plain",
     "Prove that the square root of two is irrational, then explain which step of the proof does "
     "the real work and what changes if you try the same argument on the square root of four."),
    ("math", "plain",
     "A process succeeds with probability p on each independent try. Derive the expected number of "
     "tries until the first success, then explain why the answer surprises people when p is small."),
    ("medicine", "plain",
     "Explain what a false positive rate is, using a screening test that is 99 % accurate on a "
     "condition one person in ten thousand has. Show the arithmetic and say what it means for "
     "somebody who just got a positive result."),
    ("medicine", "plain",
     "Explain in plain language how a fever works, why suppressing one is not automatically a "
     "good idea, and when it becomes one."),
    ("reasoning", "plain",
     "Three switches downstairs control three bulbs upstairs, and you may go upstairs once. "
     "Explain the solution, then explain what makes the puzzle work and how it breaks if the "
     "bulbs are LEDs."),
    ("reasoning", "plain",
     "Somebody claims a change made their program faster because the run took less time after "
     "they made it. List everything wrong with that inference and describe the smallest "
     "experiment that would settle it."),
    ("code", "code",
     "Explain what this code does, then say what its failure modes are and what you would change "
     "first:\n\n```python\n{passage}\n```"),
    ("code", "code",
     "Rewrite this code with type hints on every function and parameter, changing nothing else -- "
     "same names, same logic, same order. Output the complete result:\n\n```python\n{passage}\n```"),
    ("code", "plain",
     "Write a Python function that walks a directory tree and returns the ten largest files, "
     "with their sizes in human-readable units. Handle unreadable directories without crashing, "
     "and explain the choices you made."),
    ("multilingual", "de",
     "Fasse den folgenden Text auf Deutsch zusammen und erklaere anschliessend, welche Begriffe "
     "darin fuer Laien erklaerungsbeduerftig sind:\n\n{passage}"),
    ("multilingual", "de",
     "Lies den folgenden Abschnitt und schreibe ihn so um, dass ihn eine Schuelerin der neunten "
     "Klasse versteht. Behalte alle Fakten bei:\n\n{passage}"),
    ("multilingual", "plain",
     "Erklaere auf Deutsch, warum die Bandbreite des Speichers und nicht die Rechenleistung "
     "bestimmt, wie schnell ein Sprachmodell ein Wort nach dem anderen erzeugt. Schreibe drei "
     "Absaetze fuer interessierte Laien."),
]


# ----------------------------------------------------------------------------- corpus windows

class Corpus:
    """The token array `tools/build_corpus.py` wrote, sliced by source.

    `meta.json` lists the sources in the order they were concatenated, each with its token count,
    and one separator token per file was inserted between documents. Offsets follow from that.
    """

    def __init__(self, path: str, exclude: tuple[str, ...] = ()):
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.tokens = np.load(os.path.join(path, "tokens.npy"), mmap_mode="r")
        self.sep = int(self.meta["doc_sep"])
        self.ranges: dict[str, list[tuple[int, int]]] = {"en": [], "de": [], "code": []}
        at = 0
        for src in self.meta["sources"]:
            n = int(src["tokens"]) + int(src["files"])          # one separator per file
            lo, hi = at, at + n
            at = hi
            name = src["path"]
            if any(x in name for x in exclude):
                continue
            if "wt2" in name:
                self.ranges["en"].append((lo, hi))
            elif "de0" in name:
                self.ranges["de"].append((lo, hi))
            else:
                self.ranges["code"].append((lo, hi))

    def window(self, kind: str, n: int, rng: random.Random) -> list[int]:
        """`n` consecutive tokens of that source with no document separator in them."""
        spans = self.ranges[kind]
        for _ in range(64):
            lo, hi = spans[rng.randrange(len(spans))]
            if hi - lo < n + 2:
                continue
            at = rng.randrange(lo, hi - n - 1)
            w = np.asarray(self.tokens[at:at + n])
            if (w == self.sep).any():
                continue
            return [int(x) for x in w]
        raise RuntimeError(f"no clean window of {n} tokens in {kind}")


# ----------------------------------------------------------------------------- the tap collector

class Taps:
    """Collects the five wanted tap invocations of one `Qwen38Engine.forward`."""

    def __init__(self, n_layers: int):
        self.n = n_layers + 1
        self.want = {lid: j for j, lid in enumerate(TAP_LAYERS)}
        self.i = 0
        self.rows: list[torch.Tensor | None] = [None] * len(self.want)

    def __call__(self, h: torch.Tensor) -> None:
        if self.i == 0:
            self.rows = [None] * len(self.want)
        j = self.want.get(self.i)
        if j is not None:
            self.rows[j] = h.clone()
        self.i = (self.i + 1) % self.n

    def fused(self) -> torch.Tensor:
        if any(r is None for r in self.rows):
            raise RuntimeError("the forward did not deliver all five taps")
        return torch.cat(self.rows, dim=-1)            # [T, 5 * H]


# ----------------------------------------------------------------------------- recording

@torch.no_grad()
def record(eng: Qwen38Engine, ids: list[int], topk: int, chunk: int = 256) -> dict:
    """One teacher-forced pass over `ids`. Returns the fused taps and the target's top-k."""
    dev = eng.device
    taps = Taps(eng.cfg.num_hidden_layers)
    eng.tap = taps
    eng.reset()
    fused_parts, lab_parts, id_parts, lp_parts = [], [], [], []
    try:
        at = 0
        while at < len(ids):
            piece = ids[at:at + chunk]
            t = torch.tensor(piece, dtype=torch.long, device=dev)
            logits = eng.forward(t, start=at)
            fused_parts.append(taps.fused().to(torch.bfloat16).cpu())
            lp = torch.log_softmax(logits[0].float(), dim=-1)
            vals, idx = torch.topk(lp, topk, dim=-1)
            lab_parts.append(idx[:, 0].to(torch.int32).cpu())
            id_parts.append(idx.to(torch.int32).cpu())
            lp_parts.append(vals.to(torch.float16).cpu())
            at += len(piece)
    finally:
        eng.tap = None
    return {
        "ids": torch.tensor(ids, dtype=torch.int32),
        "fused": torch.cat(fused_parts),
        "label": torch.cat(lab_parts),
        "top_ids": torch.cat(id_parts),
        "top_lp": torch.cat(lp_parts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.expanduser("~/qwen38-spark-engine/train/data"))
    ap.add_argument("--corpus", default=os.path.expanduser("~/qwen38-spark-engine/corpus"))
    ap.add_argument("--nvfp4", default=None, help="the serving weight set; the drafter has to "
                                                  "learn the distribution it will actually face")
    ap.add_argument("--fp8-head", default=None)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=96, help="self-generated sequences")
    ap.add_argument("--gen-new", type=int, default=256, help="generated tokens each")
    ap.add_argument("--corpus-seqs", type=int, default=160, help="teacher-forced corpus windows")
    ap.add_argument("--corpus-len", type=int, default=512)
    ap.add_argument("--passage", type=int, default=160, help="tokens of passage inside a prompt")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--holdout", type=int, default=16, help="generated sequences kept for the gate")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--passage-only", action="store_true",
                    help="use only the templates that carry a corpus passage, so every prompt in "
                         "the run is distinct. A template with no variable part produces the same "
                         "prompt every time it comes round, and greedy decoding then produces the "
                         "same generation -- 96 sequences from 24 templates were 45 distinct "
                         "generations, and the copies landed on both sides of the split")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing directory and merge the manifests")
    ap.add_argument("--prefix", default="gen", help="name prefix, so an --append run does not "
                                                    "overwrite the first one's sequences")
    ap.add_argument("--budget-min", type=float, default=0.0,
                    help="stop generating after this many minutes (0 = no limit)")
    ap.add_argument("--teacher-force", default=None,
                    help="a JSONL of sequences whose text already exists (`name, topic, prompt_ids, "
                         "gen_ids`, optional `split`, `think`): each is pushed through the loaded "
                         "target once for its taps and ITS OWN argmax / top-k labels, and nothing is "
                         "generated. The text can come from another weight set; the labels never do")
    ap.add_argument("--traces", default=None,
                    help="a directory of lattice traces (`*.json` with prompt_ids + output_ids), "
                         "recorded the same way and marked held out: the gate's own positions, so "
                         "the trainer's in-loop eval reads the text the Spark replay reads")
    ap.add_argument("--skip-existing", action="store_true",
                    help="with --teacher-force / --traces: keep a sequence whose .pt is already "
                         "there (a resumed recording)")
    a = ap.parse_args()

    if a.teacher_force or a.traces:
        teacher_force_main(a)
        return

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    corpus = Corpus(a.corpus, exclude=("qwen38-spark-engine",))

    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.path)
    eos = cfg.eos_token_ids

    # ---- the prompts, built here ------------------------------------------------
    pool = [t for t in TEMPLATES if t[1] != "plain"] if a.passage_only else TEMPLATES
    prompts = []
    for i in range(a.gen):
        topic, kind, tpl = pool[i % len(pool)]
        if kind == "plain":
            text = tpl
        else:
            passage = tok.decode(corpus.window(kind, a.passage, rng), skip_special_tokens=True)
            text = tpl.format(passage=passage.strip())
        ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False),
                  return_tensors="pt").input_ids[0].tolist()
        prompts.append((f"{a.prefix}-{i:03d}", topic, ids))

    manifest = []
    t_start = time.perf_counter()

    # ---- self-generated sequences ----------------------------------------------
    drafter = DFlash2Drafter(eng, blocks=1, max_len=a.max_len)
    for name, topic, pids in prompts:
        if a.budget_min and (time.perf_counter() - t_start) / 60.0 > a.budget_min:
            print(f"[budget] stopping generation after {len(manifest)} sequences", flush=True)
            break
        t0 = time.perf_counter()
        out, st = generate_spec(eng, torch.tensor(pids, dtype=torch.long, device=a.device),
                                a.gen_new, drafter, 7, eos)
        gen_s = time.perf_counter() - t0
        drafter.detach()
        full = pids + [int(t) for t in out]
        rec = record(eng, full, a.topk)
        drafter.attach()
        rec.update({"name": name, "topic": topic, "kind": "gen", "gen_start": len(pids)})
        torch.save(rec, os.path.join(a.out, f"{name}.pt"))
        manifest.append({"name": name, "topic": topic, "kind": "gen", "n": len(full),
                         "gen_start": len(pids), "gen_tok_s": round(len(out) / gen_s, 2)})
        print(f"{name} {topic:12s} {len(full):5d} tok  gen {len(out)/gen_s:5.2f} tok/s", flush=True)
    drafter.detach()

    # ---- corpus windows ---------------------------------------------------------
    mix = ["en"] * 5 + ["de"] * 3 + ["code"] * 4
    for i in range(a.corpus_seqs):
        kind = mix[i % len(mix)]
        ids = corpus.window(kind, a.corpus_len, rng)
        rec = record(eng, ids, a.topk)
        name = f"corp{a.prefix[3:]}-{kind}-{i:03d}"
        rec.update({"name": name, "topic": kind, "kind": "corpus", "gen_start": 0})
        torch.save(rec, os.path.join(a.out, f"{name}.pt"))
        manifest.append({"name": name, "topic": kind, "kind": "corpus", "n": len(ids),
                         "gen_start": 0})
        if i % 20 == 0:
            print(f"{name} {len(ids)} tok", flush=True)

    # ---- the split --------------------------------------------------------------
    # Held out by TEMPLATE, one sequence each, the last one that used it. Neither the tail nor a
    # stride works here: the prompt list cycles through the templates in order, so the tail is a
    # run of consecutive templates and a stride whose step divides the cycle length lands on the
    # same four templates every time -- which is exactly what the first run of this tool did, and
    # it left the gate with no German and no free prose in it. One per template is the only rule
    # that cannot go wrong when the number of templates changes.
    gen_names = [m["name"] for m in manifest if m["kind"] == "gen"]
    last: dict[int, str] = {}
    for j, name in enumerate(gen_names):
        last[j % len(pool)] = name
    held = set(last.values())
    for m in manifest:
        m["split"] = "heldout" if m["name"] in held else "train"
    mpath = os.path.join(a.out, "manifest.json")
    if a.append and os.path.exists(mpath):
        with open(mpath) as f:
            old = json.load(f)["sequences"]
        have = {m["name"] for m in manifest}
        manifest = [m for m in old if m["name"] not in have] + manifest
    with open(mpath, "w") as f:
        json.dump({"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "seed": a.seed, "topk": a.topk,
                   "nvfp4": a.nvfp4, "fp8_head": a.fp8_head,
                   "tap_layers": list(TAP_LAYERS),
                   "sequences": manifest}, f, indent=2)
    n_tok = sum(m["n"] for m in manifest)
    print(f"\n{len(manifest)} sequences, {n_tok} positions, {len(held)} held out, "
          f"{(time.perf_counter() - t_start)/60:.1f} min")


def _tf_items(a) -> list[dict]:
    items: list[dict] = []
    if a.teacher_force:
        with open(a.teacher_force) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                items.append({"name": r["name"], "topic": r.get("topic") or r.get("klass", "prose"),
                              "split": r.get("split", "train"), "think": bool(r.get("think", False)),
                              "prompt_ids": [int(t) for t in r["prompt_ids"]],
                              "gen_ids": [int(t) for t in (r.get("gen_ids") or r["output_ids"])]})
    if a.traces:
        import glob
        for path in sorted(glob.glob(os.path.join(a.traces, "*.json"))):
            with open(path) as f:
                tr = json.load(f)
            items.append({"name": tr["name"], "topic": tr.get("klass", "prose"), "split": "heldout",
                          "think": bool(tr.get("thinking", False)),
                          "prompt_ids": [int(t) for t in tr["prompt_ids"]],
                          "gen_ids": [int(t) for t in tr["output_ids"]]})
    return items


def teacher_force_main(a) -> None:
    """Record taps and labels for text that already exists. One forward pass a sequence."""
    os.makedirs(a.out, exist_ok=True)
    items = _tf_items(a)
    if not items:
        raise SystemExit("nothing to record")
    names = [it["name"] for it in items]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate sequence names in the input")
    cfg = load_config(a.model)
    w = Weights(cfg.path, device=a.device, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
    eng = Qwen38Engine(cfg, w, max_len=a.max_len, device=a.device)
    mpath = os.path.join(a.out, "manifest.json")
    old: dict[str, dict] = {}
    if os.path.exists(mpath):
        with open(mpath) as f:
            old = {m["name"]: m for m in json.load(f)["sequences"]}
    manifest: dict[str, dict] = dict(old)
    t0 = time.perf_counter()
    done_pos = 0

    def flush() -> None:
        tmp = mpath + ".part"
        with open(tmp, "w") as f:
            json.dump({"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "topk": a.topk,
                       "target": os.environ.get("QWEN38_EXL3") or a.nvfp4 or cfg.path,
                       "teacher_forced": True, "tap_layers": list(TAP_LAYERS),
                       "sequences": list(manifest.values())}, f, indent=1)
        os.replace(tmp, mpath)

    for i, it in enumerate(items):
        dst = os.path.join(a.out, f"{it['name']}.pt")
        full = it["prompt_ids"] + it["gen_ids"]
        if len(full) > a.max_len:
            full = full[:a.max_len]
        if a.skip_existing and os.path.exists(dst) and it["name"] in manifest:
            continue
        rec = record(eng, full, a.topk)
        rec.update({"name": it["name"], "topic": it["topic"], "kind": "gen",
                    "gen_start": len(it["prompt_ids"])})
        torch.save(rec, dst + ".part")
        os.replace(dst + ".part", dst)
        # agreement of the recorded labels with the text: 1.0 on the target's own greedy text,
        # lower on another model's text (how off-policy the input is)
        gs = len(it["prompt_ids"])
        lab = rec["label"][gs - 1:len(full) - 1].long()
        nxt = torch.tensor(full[gs:], dtype=torch.long)
        agree = float((lab == nxt).float().mean()) if nxt.numel() else 0.0
        manifest[it["name"]] = {"name": it["name"], "topic": it["topic"], "kind": "gen",
                                "n": len(full), "gen_start": gs, "split": it["split"],
                                "think": it["think"], "label_agree": round(agree, 4)}
        done_pos += len(full)
        if i % 25 == 0 or i == len(items) - 1:
            dt = time.perf_counter() - t0
            print(f"[{i + 1:5d}/{len(items)}] {it['name']:24s} {len(full):5d} tok  agree {agree:.3f}  "
                  f"{done_pos / max(dt, 1e-9):7.1f} pos/s  {dt / 60:6.1f} min", flush=True)
            flush()
    flush()
    n_tok = sum(m["n"] for m in manifest.values())
    held = sum(1 for m in manifest.values() if m["split"] == "heldout")
    print(f"\n{len(manifest)} sequences, {n_tok} positions, {held} held out, "
          f"{(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
