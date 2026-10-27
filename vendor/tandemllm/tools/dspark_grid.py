"""Four ambiguous readings of the DSpark checkpoint, decided by acceptance.

The first run of `tools/dspark_probe.py` got slot 0 right and then wrote " the" five times, which
is what a block drafter does when the rows past the anchor are not being told apart. Four things
in the port could do that and only measurement separates them: which hidden state
`target_layer_ids` names, whether the YaRN rescaling belongs on the rotary at these positions, and
whether the bigram head helps or hurts when its previous tokens come from the drafter's own first
pass. So all of them are run, on the same prompt, and the accepted prefix decides.
"""
import argparse, itertools, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import load_config  # noqa: E402
from engine.loader import Weights  # noqa: E402
from engine.model import Qwen38Engine  # noqa: E402
from engine.spec import generate_spec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nvfp4"); ap.add_argument("--fp8-head"); ap.add_argument("--dspark")
ap.add_argument("--new", type=int, default=96)
ap.add_argument("--max-len", type=int, default=4096)
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402
cfg = load_config(None)
w = Weights(cfg.path, skip_mtp=True, nvfp4=a.nvfp4, fp8_head=a.fp8_head)
eng = Qwen38Engine(cfg, w, max_len=a.max_len)
tok = AutoTokenizer.from_pretrained(cfg.path)
ids = tok(tok.apply_chat_template(
    [{"role": "user", "content": "Write four paragraphs about the way a harbour town wakes up "
                                 "in winter."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False),
    return_tensors="pt").input_ids[0].cuda()
eos = cfg.eos_token_ids

print(f"{'tap':>7} {'rope':>6} {'markov':>7} {'acc/blk':>8} {'draft acc %':>12} {'tok/s':>7}  "
      f"first draft")
for tap, yarn, mk in itertools.product(("entry", "output"), (True, False), (True, False)):
    os.environ["QWEN38_DSPARK_NO_YARN"] = "0" if yarn else "1"
    from engine.drafters import dspark as D
    import importlib
    importlib.reload(D)
    d = D.DSparkDrafter(eng, a.dspark, markov=mk, max_len=a.max_len, tap=tap)
    d._build()
    d.attach()
    width = d.cfg.block_size - 1
    out, st = generate_spec(eng, ids, a.new, d, width, eos)
    out, st = generate_spec(eng, ids, a.new, d, width, eos)
    # one block's draft, for the eye
    eng.reset(); d.reset()
    with torch.no_grad():
        lg = eng.forward(ids, start=0, last_only=True)
        d.sync(ids.tolist(), eng.hidden_post_norm[0], 0)
    anchor = int(lg[0, -1].argmax())
    dr = d.propose(ids.tolist() + [anchor], width)
    d.detach()
    print(f"{tap:>7} {'yarn' if yarn else 'plain':>6} {str(mk):>7} {st.accept_len:8.2f} "
          f"{st.accept_rate * 100:11.1f}% {st.tok_s:7.2f}  {tok.decode(dr)!r}", flush=True)
