"""OPS-101 (second part): the Flash-Next docs and manual say what the code and the registry say, no GPU.

  * the manual has a Flash-Next chapter and includes it (manual/chapters/flashnext.tex,
    manual/tandemllm-manual.tex);
  * the pack table in docs/quantisation.md and in the chapter (Table "The Flash-Next packs") matches
    models/qwen3.8-flash-next/model.yaml and ops/profiles/flashnext-<bpw>.env row by row: the branch,
    the disk size, the weights, the n-gram table and its policy, the largest window (or that the pack
    does not load yet, or fits none);
  * docs/architecture.md marks the layering pieces not built yet (ENG-251, 253, 255, 259) as planned;
  * docs/adding-a-model.md names every adapter verb: the ones engine/models/adapter.py lists and the
    ones the 27B's adapter module installs on its engine class.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def _profiles() -> list[dict]:
    import yaml
    with open(os.path.join(ROOT, "models", "qwen3.8-flash-next", "model.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["profiles"]


def _env(profile_id: str) -> dict:
    out = {}
    for line in _read("ops", "profiles", f"{profile_id}.env").splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _window(p: dict) -> str:
    """The largest-window cell a profile's row must carry."""
    if "does not load yet" in (p.get("badges") or []):
        return "does not load yet"
    if p.get("max_window") is None:
        return "does not fit"
    return f"{p['max_window']:,}"


def _cells(rows: list[str], profile_id: str, sep: str) -> list[str]:
    split = [[c.strip() for c in r.strip().rstrip("\\").strip().strip("|").split(sep)] for r in rows]
    hit = [c for c in split if re.search(rf"(?<![\w.-]){re.escape(profile_id)}(?![\w.])", c[0])]
    assert len(hit) == 1, (profile_id, hit)
    return hit[0]


def test_the_chapter_is_in_the_manual():
    chapter = _read("manual", "chapters", "flashnext.tex")
    assert "\\chapter{Qwen3.8-Flash-Next}\\label{ch:flashnext}" in chapter
    main = _read("manual", "tandemllm-manual.tex")
    assert "\\input{chapters/flashnext}" in main
    # in the part on how the engine works, after the chapter on adding a model
    assert main.index("\\input{chapters/adding-model}") < main.index("\\input{chapters/flashnext}") \
        < main.index("\\part{The conductor}")
    return "chapters/flashnext.tex, input after adding-model"


def test_the_pack_tables_match_the_registry_and_profiles():
    profiles = _profiles()
    assert len(profiles) == 5, [p["id"] for p in profiles]
    md_rows = [ln for ln in _read("docs", "quantisation.md").splitlines() if ln.startswith("| `flashnext-")]
    tex_rows = [ln for ln in _read("manual", "chapters", "flashnext.tex").splitlines()
                if ln.startswith("\\code{flashnext-")]
    for p in profiles:
        pid, env, table = p["id"], _env(p["id"]), p["table"]
        assert env["QSE_WEIGHTS_REVISION"] == p["hf_revision"], (pid, env["QSE_WEIGHTS_REVISION"])
        assert env["QSE_FN_TABLE"] == table["policy"], (pid, env["QSE_FN_TABLE"])
        want_table = f"{table['gib']} GiB, {table['policy']}"
        # docs: profile | branch | disk | weights | table | window
        c = _cells(md_rows, pid, "|")
        assert ("(recommended)" in c[0]) == bool(p.get("recommended")), (pid, c[0])
        assert c[1] == f"`{p['hf_revision']}`", (pid, c[1])
        assert c[2] == f"{p['disk_gb']} GB", (pid, c[2])
        assert c[3].startswith(f"{p['weights_gib']} GiB"), (pid, c[3])
        assert c[4] == want_table, (pid, c[4])
        assert _window(p) in c[5], (pid, c[5])
        # manual: profile & disk & weights & table & window
        t = [x.replace("~", " ") for x in _cells(tex_rows, pid, "&")]
        assert t[1] == f"{p['disk_gb']} GB", (pid, t[1])
        assert t[2] == f"{p['weights_gib']} GiB", (pid, t[2])
        assert t[3] == want_table, (pid, t[3])
        assert _window(p) in t[4], (pid, t[4])
        assert ("recommended" in t[4]) == bool(p.get("recommended")), (pid, t[4])
    return f"{len(profiles)} profiles in docs/quantisation.md and manual/chapters/flashnext.tex"


def test_architecture_marks_the_unbuilt_layers_planned():
    lines = _read("docs", "architecture.md").splitlines()
    for key in ("ENG-251", "ENG-253", "ENG-255", "ENG-259"):
        rows = [ln for ln in lines if ln.startswith("|") and key in ln]
        assert rows, f"{key} has no row in the layers table"
        assert all("planned" in r for r in rows), (key, rows)
    return "ENG-251, 253, 255, 259 planned"


def _adapter_verbs() -> set[str]:
    doc = _read("engine", "models", "adapter.py")
    m = re.search(r"the verbs the server's loop and the caches\s+call \(([^)]*)\)", doc)
    assert m, "engine/models/adapter.py no longer lists the loop's verbs"
    verbs = {v.strip() for v in re.split(r",\s*", m.group(1).replace("\n", " ")) if v.strip()}
    # and the three it names beside the list: `caps`, `cost()`, `manifest()`
    for v in ("caps", "cost", "manifest"):
        assert re.search(rf"`{v}(\(\))?`", doc), f"engine/models/adapter.py no longer names `{v}`"
        verbs.add(v)
    installed = re.findall(r"^Qwen38Engine\.(\w+) = ", _read("engine", "models", "qwen3_5", "adapter.py"), re.M)
    assert installed, "engine/models/qwen3_5/adapter.py installs no verbs"
    return verbs | set(installed)


def test_adding_a_model_names_every_adapter_verb():
    page = _read("docs", "adding-a-model.md")
    spans = re.findall(r"`([^`]+)`", page)
    verbs = _adapter_verbs()
    # a verb as a code span of its own: `name`, `name()`, `name(n)`, `eng.name`, `name.attr`
    missing = sorted(v for v in verbs
                     if not any(re.fullmatch(rf"(eng\.)?{re.escape(v)}(\([^)]*\))?(\.\w+)?", sp) for sp in spans))
    assert not missing, f"docs/adding-a-model.md names no {missing}"
    return f"{len(verbs)} verbs: {', '.join(sorted(verbs))}"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<55} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<55} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
