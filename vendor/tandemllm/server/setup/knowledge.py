"""What Wren reads from (SRV-325): sections of the docs, the manual, the model registry and the
packages that describe TandemLLM itself, and a BM25 search over them.

The scorer is the conductor's (server/conductor/libsearch.py `TextIndex`, the BM25 the package
library's `SkillIndex` and the memory notes use: lower-case words, a plural s dropped, no
embeddings). The documents:

    docs/*.md                 one per heading section (#, ##, ###), long sections cut in parts
    manual/chapters/*.tex     the same, after a plain-text pass over the LaTeX
    models/*/model.yaml       one per model overview, weight profile, tool and field, in words
                              built from the registry (numbers with their units)
    PACKAGES                  the packages that describe TandemLLM to the model

A section's title counts three times. The index is built at the first search and again when a
source file's mtime moves or a file comes or goes, looked at once a second at most; a build over
the whole repository takes tens of milliseconds.
"""

from __future__ import annotations

import glob
import os
import re
import threading
import time

from server.conductor.libsearch import TextIndex
from server.setup import registry

ROOT = registry.ROOT
PACKAGES = ("packages/system/tandem-core/PACKAGE.md", "packages/inline/tandem-harness.md",
            "packages/inline/tandem-memory.md", "packages/skills/tandem-library/SKILL.md")
PART_CHARS = 1800
RECHECK_S = 1.0
WEIGHTS = {"title": 3, "text": 1}


def _parts(path: str, title: str, text: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []
    out, buf = [], []
    for para in re.split(r"\n\s*\n", text):
        if buf and sum(len(p) for p in buf) + len(para) > PART_CHARS:
            out.append("\n\n".join(buf))
            buf = []
        buf.append(para[:PART_CHARS * 2])
    if buf:
        out.append("\n\n".join(buf))
    return [{"path": path, "title": title + (f" ({i + 1})" if len(out) > 1 else ""), "text": t}
            for i, t in enumerate(out)]


def markdown_sections(path: str, text: str) -> list[dict]:
    """A markdown file as sections, one per heading (# to ###); fenced code stays in its section."""
    out, title, buf, fence = [], os.path.basename(path), [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            out += _parts(path, title, "\n".join(buf))
            title, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    return out + _parts(path, title, "\n".join(buf))


_TEX_DROP = re.compile(r"\\(label|ref|cite|index|footnotemark|vspace|hspace|newpage|clearpage|"
                       r"centering|small|footnotesize|toprule|midrule|bottomrule|noindent)"
                       r"(\[[^\]]*\])?(\{[^}]*\})?")


def tex_text(text: str) -> str:
    """LaTeX as plain text, roughly: comments and environments out, `\\cmd{x}` as x."""
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\begin\{[^}]*\}(\{[^}]*\}|\[[^\]]*\])*|\\end\{[^}]*\}", "", text)
    text = _TEX_DROP.sub("", text)
    for _ in range(3):
        text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"\\\\", "\n", text)
    text = text.replace("~", " ").replace("&", " | ").replace("\\%", "%").replace("\\_", "_")
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    return re.sub(r"[{}]", "", text)


def _braced(s: str, i: int) -> str:
    """The text of the brace group that opens at s[i]."""
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j]
    return s[i + 1:]


def tex_sections(path: str, text: str) -> list[dict]:
    out, title, buf = [], os.path.basename(path), []
    for line in text.splitlines():
        m = re.match(r"\s*\\(chapter|section|subsection)\*?(\[[^\]]*\])?\{", line)
        if m:
            out += _parts(path, tex_text(title).strip(), tex_text("\n".join(buf)))
            title, buf = _braced(line, m.end() - 1), []
        else:
            buf.append(line)
    return out + _parts(path, tex_text(title).strip(), tex_text("\n".join(buf)))


def _cell(m: dict, pid: str, mode: str, task: str):
    for r in m["measurements"]["rows"]:
        if r["profile"] == pid:
            c = (r["cells"].get(mode) or {}).get(task)
            return c["tok_s"] if c else None
    return None


def _pack_words(m: dict, p: dict) -> str:
    """OPS-101: a pack's largest window on the device and its n-gram table (where its profile has them); the
    27B's profiles have neither and say nothing more."""
    out = ""
    dev = m["devices"][0]["label"] if m.get("devices") else "this device"
    if "max_window" in p:
        out += (f" On the {dev} it does not fit at any window: its weights alone leave too little memory."
                if p["max_window"] is None else
                f" The longest conversation it holds on the {dev}: {p['max_window']:,} tokens (its largest window).")
    t = p.get("table")
    if t:
        out += (f" Its n-gram table ({t['gib']:g} GiB) stays warm in the page cache, which counts as used memory."
                if t["policy"] == "warm" else
                f" Its n-gram table ({t['gib']:g} GiB) is read past the page cache through a {t['cache_gib']:g} GiB row cache.")
    return out


def model_sections(m: dict) -> list[dict]:
    """The registry entry in words: what Wren quotes when it explains a choice on the page."""
    path = f"models/{m['id']}/model.yaml"
    out = [{"path": path, "title": f"{m['name']}: the model",
            "text": f"{m['name']} ({m['params']} parameters). {m['pitch']} {m['architecture']} "
                    f"License: {m['license']}"},
           {"path": path, "title": f"{m['name']}: how the speeds were measured",
            "text": f"Build {m['measurements']['build']}, {m['measurements']['date']}. "
                    f"{m['measurements']['method']}"}]
    for p in m["profiles"]:
        speed = ", ".join(f"greedy {t} {v} tok/s" for t in ("prose", "chat", "code")
                          if (v := _cell(m, p["id"], "greedy_512", t)) is not None)
        q = p["quality"]
        out.append({"path": path, "title": f"Weight profile {p['id']} ({p['label']})",
                    "text": f"{p['label']}: a {p['disk_gb']:g} GB download, {p['weights_gib']:g} GiB "
                            f"of weights in memory and {p['drafters_gib']:g} GiB for the draft "
                            f"models. Against the original BF16 model: KL {q['kl_vs_bf16']}, the "
                            f"same top token {q['top1_pct']} % of the time, perplexity {q['ppl']}. "
                            f"Speed: {speed or 'not measured on this build'}. "
                            f"{'Recommended. ' if p['recommended'] else ''}"
                            f"{', '.join(p['badges'])}. {p['note'] or ''}{_pack_words(m, p)}"})
    for c in m["components"]:
        deps = ", ".join(c["depends_on"])
        mem = f"{c['memory_gib']:g} GiB of memory" + (
            f", up to {c['memory_max_gib']:g} GiB" if c["memory_max_gib"] else "")
        out.append({"path": path, "title": f"Tool: {c['label']} ({c['id']})",
                    "text": f"{c['label']}: {c['what']} It takes {mem}"
                            + (f" and {c['disk_gb']:g} GB of disk" if c["disk_gb"] else "")
                            + f". {'On' if c['default'] else 'Off'} in the recommended setup"
                            + ("; it cannot be switched off" if c["required"] else "")
                            + (f". It needs: {deps}" if deps else "") + f". {c['note'] or ''}"})
    for g in ("methods", "advanced"):
        for f in m[g]:
            rng = (f" from {f['min']:g} to {f['max']:g}" if f.get("min") is not None else "")
            opts = (" Options: " + "; ".join(o["label"] for o in f["options"]) + "."
                    if f.get("options") else "")
            out.append({"path": path, "title": f"Setting: {f['label']} ({f['id']})",
                        "text": f"{f['label']}: {f['what']}{opts} Default {f['default']}"
                                f"{rng}; recommended {f['recommended']}."})
    p = m["parallelism"]
    out.append({"path": path, "title": "Parallel requests and memory",
                "text": f"Up to {registry.ENGINE_MAX_SLOTS} requests at once (slots; measured up to "
                        f"{p['slots']['max']} on this model, as many as the memory holds), each with a context "
                        f"window of up to {p['window']['max']:,} tokens. A token of window costs "
                        f"{p['kv_bytes_per_token']:,} bytes ({p['kv_bytes_per_token'] / 1024:g} "
                        f"KiB). With the shared KV pool, memory is taken only as conversations grow. "
                        f"The engine keeps {p['floor_gb']['default']:g} GiB free (the floor) and a "
                        f"new request waits when memory runs short."})
    return out


class Knowledge:
    def __init__(self, root: str | None = None, packages=PACKAGES, clock=time.monotonic):
        self.root = root or ROOT
        self.packages = packages
        self.clock = clock
        self.docs: list[dict] = []
        self.text = TextIndex()
        self._sig = None
        self._checked = -1e9
        self._lock = threading.Lock()
        self.builds = 0
        self.build_ms = 0.0

    def _files(self) -> list[str]:
        r = self.root
        files = sorted(glob.glob(os.path.join(r, "docs", "*.md")))
        files += sorted(glob.glob(os.path.join(r, "manual", "chapters", "*.tex")))
        files += sorted(glob.glob(os.path.join(r, "models", "*", "model.yaml")))
        files += [os.path.join(r, p) for p in self.packages if os.path.isfile(os.path.join(r, p))]
        return files

    def ensure(self) -> None:
        now = self.clock()
        if self._sig is not None and now - self._checked < RECHECK_S:
            return
        with self._lock:
            self._checked = now
            files = self._files()
            sig = tuple((f, os.path.getmtime(f)) for f in files)
            if sig == self._sig:
                return
            self._build(files)
            self._sig = sig

    def _build(self, files: list[str]) -> None:
        t0 = time.perf_counter()
        docs = []
        models, _ = registry.load_all(os.path.join(self.root, "models"))
        for f in files:
            rel = os.path.relpath(f, self.root)
            if f.endswith("model.yaml"):
                m = models.get(os.path.basename(os.path.dirname(f)))
                docs += model_sections(m) if m else []
                continue
            try:
                with open(f, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                continue
            docs += tex_sections(rel, text) if f.endswith(".tex") else markdown_sections(rel, text)
        self.text.build(docs, WEIGHTS)
        self.docs = docs
        self.builds += 1
        self.build_ms = round((time.perf_counter() - t0) * 1e3, 2)

    def search(self, query: str, k: int = 4, max_chars: int = 6000) -> list[dict]:
        """The `k` best sections for `query`, each {path, title, text, score}, their texts
        together at most `max_chars` (the last one cut)."""
        self.ensure()
        scores = self.text.scores(query)
        best = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])[:k]
        out, left = [], max_chars
        for i in best:
            if left <= 200:
                break
            d = self.docs[i]
            text = d["text"][:left]
            left -= len(text)
            out.append({"path": d["path"], "title": d["title"], "text": text,
                        "score": round(scores[i], 3)})
        return out
