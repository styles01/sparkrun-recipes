"""The Context panel's report (CONTRACT section 4, `ContextReport`; SRV-349, Chat 4).

Built from what the server says, never from a model name: the window from the model's card (or the
engine's own `max_len`), the split from chatd's counts and the last usage frame, the cache from the
engine's peek before a send (when the server has one) or from the last request's `cache_source`,
the live figures from the live row of the run, the memory from `/v1/dashboard/system`. A server
that says less gives `null`s and zeros, never invented figures. No message text goes in a report.
"""

from __future__ import annotations

FLOOR_BYTES = 15 * 1024 ** 3          # the engine's admission floor (docs/operations.md)
COMPACT_SHARE = 0.8


def estimate_tokens(text: str) -> int:
    """A count for text no tokenizer has seen yet: about 4 characters a token."""
    return (len(text) + 3) // 4 if text else 0


FILE_TOOLS = ("read_file", "write_file", "edit_file")


def file_tokens(branch: list[dict], start: int = 0, step_from: tuple[int, int] | None = None,
                fold_before: str | None = None, window: int | None = None) -> int:
    """The files in the window: what the file tools carry in the next request (the content a
    write sent, an edit's old and new text, what a read returned; a stub or a folded call where
    the request has one, chatd/render.py), estimated, from message `start` on (the first one a
    compaction kept); `step_from` (message index, step): that message's earlier steps were
    folded; `fold_before`: the newest compaction's time; `window`: the model's (the fold marks).
    Attachments count apart (their own token counts)."""
    import json

    from chatd import render
    sf = (step_from[0] - start, step_from[1]) if step_from else None
    folded = render.folded_calls(branch[start:], fold_before, sf, window)
    n = 0
    for i, m in enumerate(branch):
        if i < start or m.get("role") != "assistant":
            continue
        for p in m.get("parts") or []:
            if p.get("type") != "tool_call" or p.get("name") not in FILE_TOOLS:
                continue
            if step_from and i == step_from[0] and int(p.get("step") or 1) < step_from[1]:
                continue
            if p["name"] == "read_file":
                if not (p.get("output") or {}).get("is_error"):
                    n += estimate_tokens(render.result_for(p, fold_before))
                continue
            try:
                a = json.loads(render.call_arguments(p, p.get("id") in folded))
            except ValueError:
                continue
            if isinstance(a, dict):
                n += sum(estimate_tokens(str(a.get(k) or "")) for k in ("content", "old", "new"))
    return n


def report(*, model: dict, settings: dict, branch_tokens: int, files_tokens: int,
           stats: dict | None, run: dict | None, row: dict | None, facts: dict,
           peek: dict | None, harness: dict | None, split: dict | None = None,
           session: str | None = None, thinking: int | None = None) -> dict:
    """`model`: the chat's model as `service.model_info` gives it. `branch_tokens`: the branch as
    the next request carries it (the last request's prompt plus its answer, plus what came since).
    `stats`: the last finished answer's TurnStats; `run`/`row`: the run in flight and its live row.
    `split`: the last usage frame's `tandem.parts` (SRV-349): the system prompt's, the tool
    schemas' and the images' tokens as the engine counted them, which come out of the conversation's
    share.
    `session`: the session key the chat's requests carry, to find its rows in the engine's
    caches (`holding`). `thinking`: the turn's thinking that the next request carries (a
    compaction inside the run took its earlier steps'), when the service counted it; else the
    run's, or the last answer's."""
    window = int(model.get("max_len") or 0)
    skills = 0
    if row and row.get("hidden_tokens"):
        skills = int(row["hidden_tokens"])
    elif stats and stats.get("hidden_tokens"):
        skills = int(stats["hidden_tokens"])
    # the turn's thinking: the run's so far, its earlier calls' inside this call's prompt; after
    # the turn the last answer's, which the engine holds until the next message drops it (it
    # read 0 there, live 2026-10-16, beside "2,193 thinking" on the answer)
    th = (run or {}).get("thinking") or {}
    if thinking is None:
        thinking = int(th.get("used") or 0) if run else int((stats or {}).get("reasoning_tokens")
                                                            or 0)
    earlier = max(0, thinking - int(th.get("call") or 0)) if run else 0
    messages = max(0, branch_tokens - files_tokens)
    if row and row.get("prompt_tokens"):
        messages = max(0, int(row["prompt_tokens"]) - files_tokens - earlier)
    reserved = int(model.get("max_output") or 0)
    system = int((split or {}).get("system") or 0)
    tools = int((split or {}).get("tools") or 0)
    # the prompt's image rows as the engine counted them (VIS-205: a screenshot is 1,000 rows on Super VL)
    images = int((split or {}).get("images") or 0)
    # a relayed request's split is chatd's estimate (agent.py `estimated_split`): its skill
    # index sits inside the system prompt, counted apart
    est_skills = int((split or {}).get("skills") or 0) if (split or {}).get("estimated") else 0
    skills += est_skills
    messages = max(0, messages - system - tools - images - est_skills)
    parts = [
        {"kind": "system", "label": "system prompt", "tokens": system},
        {"kind": "tools", "label": "tool schemas", "tokens": tools},
        {"kind": "skills", "label": "skills and packages", "tokens": skills},
        {"kind": "files", "label": "files", "tokens": files_tokens},
        {"kind": "images", "label": "images", "tokens": images},
        {"kind": "messages", "label": "conversation", "tokens": messages},
        {"kind": "thinking", "label": "thinking, this turn" if run or not thinking else
         "thinking, last turn", "tokens": thinking},
        {"kind": "reserved", "label": "kept for the answer", "tokens": reserved},
    ]
    used = sum(p["tokens"] for p in parts if p["kind"] != "reserved")
    if window and reserved:
        # what a request asks for (agent.py `request_for`): the output limit, or the window's
        # room when less is left
        parts[-1]["tokens"] = reserved = max(0, min(reserved, window - used - 256))

    if row and row.get("prompt_tokens") is not None:
        cached = int(row.get("cached_tokens") or 0)
        cache = {"warm": cached, "cold": max(0, int(row["prompt_tokens"]) - cached),
                 "cold_eta_ms": None, "source": row.get("cache_source") or "none",
                 "slot": str(row["lane"]) if row.get("lane") is not None else None}
    elif peek:
        cache = {"warm": int(peek.get("cached_tokens") or 0),
                 "cold": int(peek.get("reread_tokens") or 0),
                 "cold_eta_ms": peek.get("reread_eta_ms"),
                 "source": peek.get("tier") or "none", "slot": None}
    elif stats:
        cached = int(stats.get("cached_tokens") or 0)
        cache = {"warm": cached, "cold": max(0, int(stats.get("prompt_tokens") or 0) - cached),
                 "cold_eta_ms": None, "source": stats.get("cache_source") or "unknown",
                 "slot": None}
    else:
        cache = {"warm": 0, "cold": used, "cold_eta_ms": None, "source": "none", "slot": None}

    mem = facts.get("memory") or {}
    ours = bool(facts.get("engine"))
    held = holding(facts, session, cache) if ours else {"kv_tokens": None, "kv_where": None,
                                                          "resident_bytes": None,
                                                          "resident_owner": None}
    return {
        "window": window, "used": used, "parts": parts,
        "compact_at": int(window * COMPACT_SHARE),
        "cache": cache,
        "harness": harness,
        "split": split,
        "memory": {"kv_bytes": None, "sandbox_disk_bytes": None,
                   "available_bytes": mem.get("unified_available_bytes"),
                   "floor_bytes": FLOOR_BYTES if ours else None, "sandbox_bytes": None,
                   **held},
        # the profile the engine serves (flashnext-3.05, exl3-4.5), else the chat's conductor profile
        "model": {"name": model.get("id") or "",
                  "profile": model.get("profile") or settings.get("profile"),
                  "weights": model.get("weights") or "", "drafter": model.get("drafter"),
                  "max_len": window, "effort": settings.get("effort") or "auto",
                  "thinking": bool(settings.get("thinking", True)),
                  "backend": model.get("backend") or "engine"},
    }


#: a cache source (the usage frame's `cache_source`) or a peek's tier -> the tier
TIERS = {"resident": "slot", "end": "slot", "side": "slot", "park": "park", "pin": "park",
         "ram_park": "park", "prefix": "ram_snapshot", "ram_snapshot": "ram_snapshot",
         "spill": "nvme_spill", "nvme_spill": "nvme_spill", "disk": "disk_tier",
         "disk_tier": "disk_tier"}


def _owns(res: dict, key: str, session: str | None) -> bool | None:
    """Is `session` the owner the engine's resident report names under `key` (`owner`,
    `park_owner`)? By the owner's tag when the engine sends one (engine/cache.py `owner_tag`),
    else by its shortened key; None when the slot has no owner."""
    import hashlib
    tag = res.get(key + "_tag")
    short = res.get(key)
    if tag is None and short is None:
        return None
    if not session:
        return False
    if tag is not None:
        return tag == hashlib.sha256(session.encode()).hexdigest()[:16]
    s = str(short)
    return session == s or (s.endswith("…") and session.startswith(s[:-1]))


def holding(facts: dict, session: str | None, cache: dict) -> dict:
    """What the engine holds of this chat (the Context panel's "Memory on the box"):
    `kv_tokens` the rows in the engine's slot (`kv_where` "slot") or parked beside it ("park");
    else `kv_where` the tier the next turn reads it from (`nvme_spill`, `disk_tier`,
    `ram_snapshot`), "other" when another chat has the slot and nothing of this one is cached,
    "none" when nothing is held. `resident_bytes` the resident prefix's snapshots of this chat,
    `resident_owner` "this", "other" or "none". Nulls when the engine does not report its
    caches (another server)."""
    res = (facts.get("caches") or {}).get("resident")
    if not isinstance(res, dict):
        return {"kv_tokens": None, "kv_where": None, "resident_bytes": None,
                "resident_owner": None}
    slot, park = _owns(res, "owner", session), _owns(res, "park_owner", session)
    if slot:
        return {"kv_tokens": int(res.get("valid") or 0), "kv_where": "slot",
                # its anchors, end and stash: the park (another chat) and the shared pinned head
                # are not this chat's
                "resident_bytes": max(0, int(res.get("bytes") or 0) - int(res.get("park_bytes")
                                                                          or 0)
                                      - int(res.get("pin_bytes") or 0)),
                "resident_owner": "this"}
    if park:
        return {"kv_tokens": int(res.get("park_rows") or 0), "kv_where": "park",
                "resident_bytes": int(res.get("park_bytes") or 0), "resident_owner": "this"}
    tier = TIERS.get(str(cache.get("source") or ""), None) if cache.get("warm") else None
    where = tier if tier in ("nvme_spill", "disk_tier", "ram_snapshot") else (
        "other" if slot is False else "none")
    return {"kv_tokens": None, "kv_where": where, "resident_bytes": None,
            "resident_owner": "other" if slot is False else "none"}
