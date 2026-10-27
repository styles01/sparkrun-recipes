"""The disk tier in the server (C1, 2026-10-02): where it lives, its settings, its journal, the tag
a request hands the prefill, and what the dashboard's Memory screen shows.

    STATE["disk_cache"] = start(STATE, a, journal)   # None: --disk-cache off, the fake engine
    refresh(STATE)                                   # a request starts: the settings' new version
    tag = request_tag(plan, client, prompt_ids, tok) # the client, the injected notes, the head
    view(STATE)                                      # GET /v1/conductor/memory/disk

`--disk-cache PATH` (serve.env `QSE_DISK_CACHE`, which ops/start.sh passes to the :8000 service
only, like the ledger) turns the tier on and names its directory; a row, gate or soak server never
reads or writes it. Everything else is the conductor's `settings` document, `memory.disk_cache`
(mode, budget_gb, ttl_days, admit_after, min_free_gb), versioned like the transcript settings and
applied at the start of the next request; with no conductor, the defaults. The free-space floor is
never under 50 GB, whatever the document says (engine/disk_cache.py FLOOR_BYTES).
"""

from __future__ import annotations

import hashlib
import os
import sys

from engine import disk_cache as dc

DEFAULT_ROOT = "~/.qwen38-spark-engine/cache/disk"
# what an operator acts on goes to the log as well as the journal; a hit is the request's own line
_LOGGED = ("write", "evict", "drop", "restore_failed", "error", "clear")
# the settings environment that cannot change a prefill's rows: left out of the fingerprint
_NOT_ARITHMETIC = ("QWEN38_DECIDER", "QWEN38_NARRATOR", "QWEN38_PROBE", "QWEN38_DRAFT_TEMP",
                   "QWEN38_CONDUCTOR_FIT")
# rc-0.8.0: nor can the shadow reader's (server/conductor/shadow.py: a second model's own states).
# The extra by path, which 0.7.2 hashed, keeps them, so its files can be adopted
_NOT_ROWS = _NOT_ARITHMETIC + ("QWEN38_SHADOW",)
#: the serve.env lines main added after 0.7.2, at their defaults: a 0.7.2 environment had none of
#: them, so 0.7.2 hashed this tree's extra by path without them (`legacy_fingerprints`). Live on the
#: box, these four and the tree's folder were the whole difference (tools/tier_fp_inputs.py)
ADDED_AFTER_072 = {"QWEN38_SHADOW_DIR": "", "QWEN38_SHADOW_DISK_GB": "20", "QWEN38_SHADOW_GB": "4",
                   "QWEN38_SHADOW_GRID": "512"}
# the arguments that name weight and drafter files
_ARG_FILES = ("model", "nvfp4", "fp8_head", "head", "dflash2_ckpt", "dflash2_ckpt16")
# the variables outside QWEN38_* that name weight files (NemotronH's GGUF beside its config folder)
_ROW_FILES = ("QSE_NH_GGUF",)
# a file past this size is identified by a sample of its bytes (`content_id`)
SAMPLE_OVER = 64 << 20
_IDS: dict = {}


def settings_of(settings: dict | None) -> dict:
    """`settings.memory.disk_cache` over the defaults (a version stored before C1 has none)."""
    got = ((settings or {}).get("memory") or {}).get("disk_cache") or {}
    return {**dc.SETTINGS, **{k: v for k, v in got.items() if k in dc.SETTINGS}}


def spill_of(settings: dict | None) -> dict:
    """`settings.memory.session_spill` (budget_gb, codec, ttl_hours, min_tokens): the session
    spill's settings that the document overrides, live from the next request (0.4.3-test). Keys
    the document does not set keep the command line's."""
    from engine import spill as sp
    got = ((settings or {}).get("memory") or {}).get("session_spill") or {}
    return {k: v for k, v in got.items()
            if k in ("budget_gb", "codec", "ttl_hours", "min_tokens") and v is not None
            and k in sp.SETTINGS}


def writes_of(settings: dict | None, a) -> dict:
    """`settings.memory.disk_writes.budget_gb` over `--disk-write-budget-gb` (None: not set)."""
    got = ((settings or {}).get("memory") or {}).get("disk_writes") or {}
    b = got.get("budget_gb")
    if b is None and a is not None:
        b = float(getattr(a, "disk_write_budget_gb", dc.WRITE_BUDGET_GB))
    return {"budget_gb": b}


def _settings_doc(state) -> tuple:
    cond = state.get("conductor")
    if cond is None:
        return None, {}
    try:
        return cond.store.peek("settings")
    except Exception:                                              # noqa: BLE001
        return None, {}


def _file_id(path: str) -> list:
    try:
        st = os.stat(path)
        return [path, st.st_size, int(st.st_mtime)]
    except OSError:
        return [path, None, None]


def content_id(path: str):
    """What is at `path`, not where it is: a file's size and sha256 (over the whole file up to
    SAMPLE_OVER; past that over its first and last MiB and 16 reads of 64 KiB spread between them),
    a folder's .safetensors, .json and .gguf files by name, "missing" for neither. Remembered by the
    file's real path, size, mtime and inode for the process."""
    p = os.path.expanduser(path)
    if os.path.isdir(p):
        return {"dir": [[f, content_id(os.path.join(p, f))] for f in sorted(os.listdir(p))
                        if f.endswith((".safetensors", ".json", ".gguf"))]}
    try:
        st = os.stat(p)
    except OSError:
        return "missing"
    key = (os.path.realpath(p), st.st_size, st.st_mtime_ns, st.st_ino)
    got = _IDS.get(key)
    if got is None:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            if st.st_size <= SAMPLE_OVER:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            else:
                spots = [(0, 1 << 20), (st.st_size - (1 << 20), 1 << 20)]
                spots += [(st.st_size * i // 17, 1 << 16) for i in range(1, 17)]
                for off, n in spots:
                    f.seek(off)
                    h.update(f.read(n))
        got = _IDS[key] = f"{st.st_size}:{h.hexdigest()}"
    return got


def _located(v: str):
    """An environment value that names files by where they are (a path, or paths with commas): what
    is there instead. Any other value as it is."""
    parts = v.split(",")
    if not v or not all(x.startswith(("/", "~")) for x in parts):
        return v
    ids = [content_id(x) for x in parts]
    return ids[0] if len(ids) == 1 else ids


def _libraries(out: dict) -> dict:
    out["python"] = sys.version.split()[0]
    for mod in ("torch", "triton"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                          # noqa: BLE001
            out[mod] = None
    return out


def fingerprint_extra(a) -> dict:
    """What a prefill's rows depend on besides the engine's code and the tokens, by content: the
    weight and drafter files, the arithmetic flags in the environment (a value that names files, by
    what is in them), the grid, the drafter and the libraries' versions. A path is only where a file
    lives: a tree in another folder, or a copied checkpoint, keeps the fingerprint (rc-0.8.0: the 27B
    moved from ~/tandem-live to ~/tandem-rc080 swept 29 files, 2.97 GB, for its folder's name)."""
    files = []
    for name in _ARG_FILES:
        v = getattr(a, name, None)
        files += [content_id(p) for p in (str(v).split(",") if v else [])]
    env = {k: _located(v) for k, v in sorted(os.environ.items())
           if (k.startswith("QWEN38_") and not k.startswith(_NOT_ROWS)) or k in _ROW_FILES}
    return _libraries({"files": files, "env": env, "chunk": getattr(a, "prefix_chunk", None),
                       "drafter": getattr(a, "drafter", None),
                       "narrow_arm": getattr(a, "narrow_arm", None)})


def fingerprint_extra_paths(a) -> dict:
    """The extra as 0.7.2 and the RC up to fp4 hashed it: the weight files by path, size and mtime,
    the environment's values as written. Kept to adopt their files (`legacy_fingerprints`)."""
    files = []
    for name in _ARG_FILES:
        v = getattr(a, name, None)
        for p in (str(v).split(",") if v else []):
            p = os.path.expanduser(p)
            if os.path.isdir(p):
                files += [_file_id(os.path.join(p, f)) for f in sorted(os.listdir(p))
                          if f.endswith((".safetensors", ".json"))]
            else:
                files.append(_file_id(p))
    env = {k: v for k, v in sorted(os.environ.items())
           if k.startswith("QWEN38_") and not k.startswith(_NOT_ARITHMETIC)}
    return _libraries({"files": files, "env": env, "chunk": getattr(a, "prefix_chunk", None),
                       "drafter": getattr(a, "drafter", None),
                       "narrow_arm": getattr(a, "narrow_arm", None)})


def _moved(extra: dict, here: str, old: str) -> dict:
    """`extra` as the same files had it under the tree's old folder `old`."""
    def at(v):
        return old + v[len(here):] if isinstance(v, str) and (v == here or v.startswith(here + "/")) else v
    out = dict(extra)
    out["files"] = [[at(f[0]), *f[1:]] for f in extra["files"]]
    out["env"] = {k: ",".join(at(x) for x in v.split(",")) for k, v in extra["env"].items()}
    return out


def _same_content(extra: dict, here: str, old: str) -> bool:
    """Whether every argument file of `extra` under `here` (the drafter checkpoints) holds what the
    same file under `old` holds: a moved tree's files, not others with the same size and mtime.
    False when one is missing at either place. The tables the environment names are not compared:
    the extra by path never read their content (fp4 and v3 do)."""
    for p in (f[0] for f in extra["files"]):
        if isinstance(p, str) and p.startswith(here + "/"):
            q = old + p[len(here):]
            if not (os.path.isfile(p) and os.path.isfile(q)) or content_id(p) != content_id(q):
                return False
    return True


def legacy_fingerprints(here: str, m, a, adopt_from=(), tree: str | None = None) -> list:
    """The fingerprints the files this start may adopt were written under, for the same rows: the
    extra by path on this tree (0.7.2's v2, the RC's fp3 before fp4), and as the trees in
    `adopt_from` had it (QSE_DISK_ADOPT_FROM: the folders a tree moved from), each also without the
    serve.env lines main added after 0.7.2 (`ADDED_AFTER_072`) where they hold their defaults. A
    folder counts only when its files hold what this tree's hold (`_same_content`).
    `here` is the code's tree; `tree` the folder the paths are under (`here` unless a test says)."""
    from engine import spill as spill_mod
    from engine.methods import manifest as mf
    tree = here if tree is None else tree
    base, out = fingerprint_extra_paths(a), []
    olds = [os.path.abspath(os.path.expanduser(x)) for x in adopt_from if x]
    for old in (tree, *[x for x in olds if x != tree and _same_content(base, tree, x)]):
        e = base if old == tree else _moved(base, tree, old)
        bare = dict(e, env={k: v for k, v in e["env"].items() if ADDED_AFTER_072.get(k) != v})
        for x in (e, bare):
            out += [dc.fingerprint(here, x, skip=spill_mod.NOT_ARITHMETIC),
                    mf.fingerprint(here, m, x, formula="fp3")]
            if mf.SAME_ROWS:
                # 0.8.2: the same on the bytes `SAME_ROWS` replaced
                out += [dc.fingerprint(here, x, skip=spill_mod.NOT_ARITHMETIC, same=mf.SAME_ROWS),
                        mf.fingerprint(here, m, x, formula="fp3", same=mf.SAME_ROWS)]
    if mf.SAME_ROWS:
        # and fp4 itself (its extra by content, as `start` computes it) on those bytes: the files
        # this tree's formula wrote before a `SAME_ROWS` change; ENG-341: on every mix of the
        # replaced bytes too, one of them the release's before (`same_variants`)
        x = fingerprint_extra(a)
        out.append(mf.fingerprint(here, m, x, same=mf.SAME_ROWS))
        out += [mf.fingerprint(here, m, x, same=s) for s in mf.same_variants(here, m)]
    return list(dict.fromkeys(out))


def _in_repository(path: str) -> str | None:
    from server.conductor.transcripts import in_repository
    return in_repository(path)


def start(state, a, journal=None):
    """The tier, opened, or None (`--disk-cache off`, the fake engine, no prefix cache, a drafter
    that cannot resume, or a directory inside a git work tree -- refused, never written)."""
    from engine import cache
    root = getattr(a, "disk_cache", "off") or "off"
    if root == "off" or getattr(a, "fake_engine", False):
        print("[disk-cache] off", flush=True)
        return None
    eng, drafter = state.get("engine"), state.get("drafter")
    chunk = int(state.get("prefix_chunk") or 0)
    if not state.get("prefix_cache") or chunk <= 0 or not cache.resident_capable(drafter):
        print("[disk-cache] off: it needs the prefix cache's grid and a drafter that resumes in "
              "place", flush=True)
        return None
    root = os.path.realpath(os.path.expanduser(root))
    repo = _in_repository(root)
    if repo is not None:
        print(f"[disk-cache] off: {root} is inside the git work tree {repo}", flush=True)
        return None
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from engine.methods import manifest as mf
    extra = fingerprint_extra(a)
    # fingerprint v3 (ENG-243): what the model's adapter says its rows depend on, by content (fp4)
    fp = mf.fingerprint(here, eng.manifest(), extra)
    # the first start under this formula adopts the files of the ones before it for the same rows
    # (0.7.2's v2, the RC's fp3; `legacy_fingerprints`); the record in the directory says so
    # from then on
    adopt = [x for x in (os.environ.get("QSE_DISK_ADOPT_FROM") or "").split(":") if x]
    also = dc.equivalents(root, fp, legacy_fingerprints(here, eng.manifest(), a, adopt),
                          formula=mf.adoption())
    state["disk_cache_fp_also"] = also
    tok = state.get("tok")

    def on_event(kind: str, detail: dict) -> None:
        if journal is not None:
            journal.note("memory", "disk_cache", decision=kind, detail=detail)
        if kind in _LOGGED:
            print(f"[disk-cache] {kind} " + " ".join(f"{k}={v}" for k, v in detail.items()),
                  flush=True)

    def preview(keys, L, kind, spans) -> str | None:
        """A short line of an opencode head (its system text) or a package (its note): text the
        operator and the client wrote, never the user's. Images' ids are not text."""
        if tok is None:
            return None
        vocab = len(tok)
        if kind == "package" and spans:
            s = spans[-1]
            ids = [int(x) for x in keys[s[0]:min(s[1], s[0] + 64)]]
        else:
            ids = [int(x) for x in keys[:96]]
        text = tok.decode([x for x in ids if 0 <= x < vocab], skip_special_tokens=True)
        text = " ".join(text.split())
        return text[:120] or None

    _, doc = _settings_doc(state)
    tier = dc.DiskTier(root, chunk=chunk, fingerprint=fp, settings=settings_of(doc),
                       head_anchors=int(getattr(a, "disk_cache_head_anchors", 4)),
                       min_tokens=int(getattr(a, "disk_cache_min_tokens", 4096)),
                       on_event=on_event, preview=preview, also=also)
    tier.plan_staging(eng, drafter)

    def on_guard(kind: str, detail: dict) -> None:
        if journal is not None:
            journal.note("memory", "disk_writes", decision=kind, detail=detail)
        print(f"[disk-guard] {kind} " + " ".join(f"{k}={v}" for k, v in detail.items()),
              flush=True)

    # The day's NVMe write budget (0.4.3-test): the tier's writes and the spill's count against it
    tier.guard = dc.WriteGuard(root, budget_gb=writes_of(doc, a)["budget_gb"], on_event=on_guard)
    tier.open()
    state["disk_cache_version"] = None
    refresh(state, tier)
    rep = tier.report()
    print(f"[disk-cache] {'on' if tier.on else 'off (settings)'}: {root}, {rep['entries']} "
          f"entries, {rep['bytes'] / dc.GB:.1f} of {rep['budget'] / dc.GB:g} GB, "
          f"{rep['free_bytes'] / dc.GB:.0f} GB free (floor {rep['floor'] / dc.GB:g}), "
          f"TTL {tier.cfg['ttl_days']} days, admit after {tier.cfg['admit_after']}; staging "
          f"{tier.staging_planned / dc.GB:.2f} GB when first used; fingerprint {fp[:12]}"
          + (f" (also reads {', '.join(x[:12] for x in also)})" if also else ""), flush=True)
    return tier


def start_spill(state, a, tier, journal=None):
    """The session spill (engine/spill.py) under the tier's directory, or None
    (`--session-spill off`, no disk tier, no resident prefix). Its fingerprint leaves out the
    engine files that keep state rather than compute it, so a deploy that changes only those
    still reads its sessions back."""
    from engine import spill as sp
    res = state.get("resident")
    al = state.get("end_align")
    if res is not None and al is not None:
        wire_end_align(None, al, res)          # the park's end anchor, with or without a spill
    if tier is None or res is None or getattr(a, "session_spill", "on") == "off":
        print("[spill] off", flush=True)
        return None
    from engine.methods import manifest as mf
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fp = mf.fingerprint(here, state.get("engine").manifest(), fingerprint_extra(a))
    # the tier's record (`dc.equivalents`): the same fingerprints, as the files are the same rows
    also = tuple(state.get("disk_cache_fp_also") or ()) if fp == getattr(tier, "fp", None) else ()

    def on_event(kind: str, detail: dict) -> None:
        if journal is not None:
            journal.note("memory", "session_spill", decision=kind, detail=detail)
        print(f"[spill] {kind} " + " ".join(f"{k}={v}" for k, v in detail.items()), flush=True)

    spill = sp.SessionSpill(os.path.join(tier.root, "sessions"), chunk=tier.chunk,
                            fingerprint=fp, tier=tier, on_event=on_event, also=also,
                            settings={"budget_gb": float(getattr(a, "session_spill_gb", 64.0)),
                                      "min_tokens": int(getattr(a, "session_spill_min_tokens",
                                                                16384)),
                                      "codec": str(getattr(a, "session_spill_codec", "zstd")),
                                      **spill_of(_settings_doc(state)[1])}).open()
    spill.guard = getattr(tier, "guard", None)
    if al is not None:
        wire_end_align(spill, al)
    res.spiller = spill
    rep = spill.report()
    print(f"[spill] on: {spill.root}, {rep['entries']} conversations, "
          f"{rep['bytes'] / dc.GB:.1f} of {rep['budget'] / dc.GB:g} GB, evictions from "
          f"{spill.min_tokens} tokens, codec {spill.codec or 'none'}; fingerprint {fp[:12]}",
          flush=True)
    return spill


def wire_end_align(spill, al, res=None) -> None:
    """The end alignment's part in the NVMe entries (server/endalign.py): its regions written with
    an entry and read back with it, and (SRV-269) the restore's offer aligning a copy the way the
    resume after the restore will, so a chat whose client leaves the reasoning out comes back
    from the disk instead of being read again. A parked conversation's end (`res`, engine/cache.py
    `ResidentPrefix._park`) the same way: its regions parked with it and adopted when it comes
    back, the park's offer aligning the copy, and the parked end written to the NVMe with them."""
    if spill is not None:
        spill.extra_of = lambda r: {"end_regions": [list(x) for x in al.regions(r)]}
        spill.on_adopt = lambda r, extra: al.adopt(r, extra.get("end_regions") or [])
        spill.offer_align = lambda toks, end, extra, keys: al.offers(
            toks, end, (extra or {}).get("end_regions") or [], keys)
        spill.park_extra_of = lambda r: {"end_regions": [list(x) for x in r.park_regions]}
    if res is not None:
        res.end_extra_of = al.regions
        res.end_adopt = lambda r, regions: al.adopt(r, regions or [])
        res.park_align = lambda toks, end, regions, keys: al.offers(toks, end, regions or [], keys)


def refresh(state, tier=None) -> list[str]:
    """A request starts: a new version of the settings document applies now."""
    tier = tier if tier is not None else state.get("disk_cache")
    if tier is None:
        return []
    v, doc = _settings_doc(state)
    if v is not None and v == state.get("disk_cache_version"):
        return []
    state["disk_cache_version"] = v
    changed = tier.apply(settings_of(doc))
    sp = state.get("spill")
    if sp is not None:
        before = dict(sp.cfg)
        sp.apply(spill_of(doc))
        changed += [f"spill.{k}={sp.cfg[k]}" for k in sp.cfg if sp.cfg[k] != before.get(k)]
    g = getattr(tier, "guard", None)
    if g is not None:
        b = writes_of(doc, None)["budget_gb"]
        if b is not None and float(b) != g.budget_gb:
            g.budget_gb = float(b)
            changed.append(f"disk_writes.budget_gb={b}")
    if changed:
        print(f"[disk-cache] settings v{v}: " + ", ".join(changed), flush=True)
    return changed


class SideReader:
    """A side slot's view of slot 0's disk tier (`QWEN38_SIDE_DISK=read`).

    A side slot has no resident prefix, no park and, until 0.4.0, no disk tier: a new chat
    sent there read its client's whole head again (Open WebUI's 6,687 tokens in 9.0 s, opencode's
    7,414 in 19.1 s against 10.7 s on slot 0, where the tier held the head). Through this view a
    side slot's prefill asks the tier as slot 0's does and reads a hit into ITS OWN buffers
    (`DiskTier.restore` streams the rows and the state into the engine it is given). What the
    tier does for slot 0 alone stays slot 0's: counting lines and planning writes (`plan`, so a
    side prefill writes nothing and admits nothing), the write-behind copies (`capture`, `seal`),
    and the fences, which protect rows of slot 0's buffers a write is still copying (a side
    slot's prefill never writes those rows, so it fences nothing). A restore and the writer share
    the tier's staging slots under its own locks, as two prefills on slot 0 would."""

    def __init__(self, tier):
        self.tier = tier
        self.stats = {"lookups": 0, "hits": 0, "tokens_restored": 0, "failed": 0}

    @property
    def on(self) -> bool:
        return bool(self.tier.on)

    def find(self, keys, hashes, max_len: int, min_len: int, have: int = 0):
        self.stats["lookups"] += 1
        return self.tier.find(keys, hashes, max_len, min_len, have)

    def restore(self, eng, drafter, hit) -> bool:
        ok = self.tier.restore(eng, drafter, hit)
        if ok:
            self.stats["hits"] += 1
            self.stats["tokens_restored"] += int(hit.L) - int(hit.have)
        else:
            self.stats["failed"] += 1
        return ok

    def fence(self, safe: int) -> None:                 # slot 0's rows only
        return None

    def plan(self, *a, **kw):                           # a side prefill writes nothing
        return None

    def capture(self, *a, **kw) -> None:
        return None

    def seal(self, *a, **kw) -> None:
        return None

    def guard_rows(self, n: int) -> int:
        return 0

    def _event(self, kind: str, **detail) -> None:
        self.tier._event(kind, side=True, **detail)

    def report(self) -> dict:
        return dict(self.stats)


def side_reader(state):
    """The disk tier a request on a side slot may read (`SideReader`), or None: no tier, the tier
    off, or `QWEN38_SIDE_DISK=off`. One reader for every side slot, kept in the state."""
    from engine.settings import SETTINGS as _S
    if (_S.get("SIDE_DISK") or "read").strip().lower() != "read":
        return None
    tier = dict.get(state, "disk_cache")
    if tier is None:
        return None
    r = dict.get(state, "disk_side")
    if r is None or r.tier is not tier:
        r = SideReader(tier)
        dict.__setitem__(state, "disk_side", r)
    return r


def request_tag(plan, client: str | None, prompt_ids: list[int], tok) -> dict:
    """What the tier is told about a request: its client, the notes the conductor spliced into
    its prompt (served positions, kinds, package names), and `head_end` -- where its last user
    message starts (the second-to-last `<|im_start|>`, the last one opening the answer), which is
    where the next session of the same client parts from it; `first_answer` -- where the
    conversation's first answer starts (the third `<|im_start|>`: the system turn, the first user
    message, the answer), past which two sessions share nothing; `side`, a client's side request
    (S1), which the tier writes nothing of."""
    spans = []
    for s in (getattr(plan, "spans", None) or []) if plan is not None else []:
        spans.append(tuple(s))
    head_end = first_answer = None
    try:
        im = tok.convert_tokens_to_ids("<|im_start|>")
        seen = 0
        for i in range(len(prompt_ids) - 1, -1, -1):
            if prompt_ids[i] == im:
                seen += 1
                if seen == 2:
                    head_end = i
                    break
        seen = 0
        for i, x in enumerate(prompt_ids):
            if x == im:
                seen += 1
                if seen == 3:
                    first_answer = i
                    break
    except Exception:                                              # noqa: BLE001
        head_end = first_answer = None
    # S1: a client's side request (a title or summary call) is read like any request and admits
    # nothing: its prompt is not a prefix the client sends again
    side = plan is not None and getattr(plan, "side", None) is not None
    return {"client": client, "spans": spans, "head_end": head_end, "first_answer": first_answer,
            "side": side}


def view(state) -> dict:
    """The Memory screen's Disk cache tab: whether the tier runs, its settings and their version,
    its numbers, the most-hit entries (previews only for opencode heads and packages) and the last
    events. Behind the admin token, like every memory route."""
    tier = state.get("disk_cache")
    v, doc = _settings_doc(state)
    args = state.get("args") or {}
    out = {"running": tier is not None, "settings_version": v, "settings": settings_of(doc),
           "flag": args.get("disk_cache", "off"), "floor_gb": dc.FLOOR_BYTES / dc.GB}
    if tier is None:
        root = args.get("disk_cache")
        out["root"] = None if root in (None, "off") else root
        free = None
        try:
            free = dc._statvfs_free(os.path.expanduser(out["root"] or "~"))
        except OSError:
            pass
        out["free_bytes"] = free
        return out
    out.update(tier.report(top=20, previews=True))
    return out


def clear(state) -> dict:
    tier = state.get("disk_cache")
    if tier is None:
        return {"cleared": False, "why": "the disk tier is not running on this server"}
    return {"cleared": True, **tier.clear()}
