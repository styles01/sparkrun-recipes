# Portions of this file are from the Laya SDK 0.3.4 (`laya/common.py`, `laya/agent.py`),
# Copyright Convai Innovations, licensed under the Apache License 2.0
# (engine/decision/laya/LICENSE-APACHE-2.0; NOTICE at the repository root).
# Changes: split out of the SDK's runtime, `_to_internal` made a function, the collate trimmed to
# what inference needs (no training targets). The arithmetic and the token layout are unchanged:
# `tools/decider_equivalence.py` holds them to the SDK at 0.00.
"""How a Laya question becomes a row, and how a row's logits become an answer's numbers.

A row is `[CLS] <type> question: <instructions> [SEP] [MASK] option_0 [MASK] option_1 ... [SEP]
state [SEP]`: the question head, one `[MASK]` marker per option (the scorer reads the marker's
output), then the state, cut to the room the head leaves in `max_len`. Noul is always the two
options [false, true], so p[1] is the noul probability.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def to_internal(qdef: dict) -> dict:
    """A Jev question definition -> the SDK's internal `{t, ins, crit}` form."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def serialize_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    """Strings pass through; anything structured becomes compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: dict) -> list[str]:
    """Option texts in label-index order. Noul is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values
        return [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
                for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(false_crit) if false_crit not in (None, "")
                     else "no, the statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit not in (None, "")
                    else "yes, the statement holds"),
    ]


def build_sequence(tok, state, q: dict, max_len: int = 512, head_max_len: int = 192,
                   option_order: list[int] | None = None, truncate_left: bool = False):
    """The row's token ids and the positions of its option markers."""
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append([tok.mask_token_id]
                       + tok(" " + opts[i].replace(mask_tok, " "),
                             add_special_tokens=False)["input_ids"][:48])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def collate(rows: list[dict], pad_id: int) -> dict:
    """Rows -> padded tensors (on the CPU): input ids, attention mask, marker positions and mask,
    question types. A row's padding is attended by nobody and its markers past `k` are masked."""
    n, L = len(rows), max(len(r["ids"]) for r in rows)
    kmax = max(len(r["markers"]) for r in rows)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, : len(r["ids"])] = torch.tensor(r["ids"])
        att[i, : len(r["ids"])] = 1
        k = len(r["markers"])
        mpos[i, :k] = torch.tensor(r["markers"])
        mmask[i, :k] = True
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "qtype": torch.tensor([r["qtype"] for r in rows])}


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalised Shannon-entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def answer(q: dict, k: int, qtype: int, logits_row: np.ndarray, act_row: np.ndarray,
           temperature, temperature_by_options: dict) -> dict:
    """One row's logits -> the Jev answer object, temperature-scaled per type and option count."""
    t_scale = temperature_by_options.get(temp_bucket(qtype, k), temperature[qtype])
    z = logits_row[:k] / max(1e-3, float(t_scale))
    p = np.exp(z - z.max())
    p = p / p.sum()
    conf = round(confidence_from_probs(p, k), 4)
    ext = {"act_probability": round(float(act_row[0]), 4)}
    if q["t"] == "choice":
        keys = list(q["crit"].keys())
        return {"type": "choice", "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                "confidence": conf, "action": ext}
    if q["t"] == "score":
        return {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": conf, "action": ext}
    return {"type": "noul", "noul": round(float(p[1]), 4),
            "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4), "action": ext}
