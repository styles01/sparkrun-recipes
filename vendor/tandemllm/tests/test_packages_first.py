"""The first packages (OPS-39; QA OPS-40): the seven folders are consistent with the shipped
documents, within their caps, their eval sets load and are free of private text, and the tool fit
keeps the two tool-bound packages out of a chat without their tools.

  1  tools/packages_first.py's check: no finding on the repository (manifests, seed.json, INDEX.md,
     the router, the texts' sizes on the server's estimate, the eval sets, the rubrics, privacy,
     the shipped documents validate)                                   test_the_folders_check_out
  2  the counts the ticket asks for: 24 creative-coding, 20 charts, 20 voxel-worlds (one of them
     the six-feature sequence), 12 explainer-video prompts; the system set's small sets; every
     prompt's checks are known check ids                              test_the_eval_sets
  3  on the engine's tokenizer (TANDEM_TOKENIZER=<dir with tokenizer.json>, else skipped): every
     text that goes in is within its cap, the library bodies within the house 350
                                                                      test_the_caps_on_the_engine_tokenizer
  4  the shipped router: tandem-core on every class but summarize; render-tools on every class
     that builds; the four first packages where the manifests say; nothing over 16
                                                                      test_the_router_lists_them

Run: python tests/test_packages_first.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.conductor import schema  # noqa: E402
from tools import packages_first as P  # noqa: E402

COUNTS = {"tandem-core": 8, "render-tools": 6, "eval-judge": 6, "creative-coding": 24, "charts": 20,
          "voxel-worlds": 20, "explainer-video": 12}
KNOWN_CHECKS = {
    "one_file", "runs", "no_console_errors", "fps_min", "stable_60s", "audio_nonsilent", "prompt_terms",
    "fps_baseline_held", "seed_stable", "todo_file",
    "file_exists", "length_range", "loudness_range", "true_peak", "no_black_second", "sources_named",
    "schematic_labelled", "narration_present", "no_number_scenes",
    "tool_args_typed", "no_think_tags", "one_answer", "file_lines_max_200", "plain_final_answer",
    "render_called", "renders_max_3", "audio_probe_called", "fix_after_error", "allowlist_only",
    "json_only", "all_items_scored", "scores_in_range", "stable_twice", "expected_band"}


def test_the_folders_check_out():
    findings, rows = P.check(None)
    assert not findings, "\n".join(findings)
    assert [r["name"] for r in rows] == list(P.SYSTEM + P.FIRST)
    for r in rows:
        assert r["measured"] == "not measured", r      # no run exists; nothing is marked measured
    return f"{len(rows)} packages"


def test_the_eval_sets():
    for name, n in COUNTS.items():
        m = P.manifest(name)
        rows = P._jsonl(os.path.join(P.folder(name), m["eval"]["prompts"]))
        assert len(rows) == n == m["eval"]["count"], (name, len(rows))
        for r in rows:
            assert set(r["checks"]) <= KNOWN_CHECKS, (name, r["id"], set(r["checks"]) - KNOWN_CHECKS)
        assert set(m["eval"]["checks"]) <= KNOWN_CHECKS, name
    seq = [r for r in P._jsonl(os.path.join(P.folder("voxel-worlds"), "eval/prompts.jsonl"))
           if r.get("features") == 6]
    assert len(seq) == 1 and "fps_baseline_held" in seq[0]["checks"] and seq[0]["id"] == "vw-20"
    charts = P._jsonl(os.path.join(P.folder("charts"), "eval/prompts.jsonl"))
    assert all("prompt_terms" in r["checks"] and len(r["terms"]) >= 3 for r in charts)
    videos = P._jsonl(os.path.join(P.folder("explainer-video"), "eval/prompts.jsonl"))
    assert all(r["asked_seconds"] in (30, 40, 45, 50, 60) for r in videos)
    assert sum(1 for r in videos if r["asked_seconds"] == 30) == 5
    return f"{sum(COUNTS.values())} prompts"


def test_the_caps_on_the_engine_tokenizer():
    d = os.environ.get("TANDEM_TOKENIZER")
    if not d or not os.path.isfile(os.path.join(os.path.expanduser(d), "tokenizer.json")):
        print("    (TANDEM_TOKENIZER not set: skipped)")
        return "skipped"
    findings, rows = P.check(d)
    assert not findings, "\n".join(findings)
    for r in rows:
        assert r["tokenizer"] == "engine" and r["tokens"] <= r["cap"], r
        if r["library"]:
            assert r["tokens"] <= P.HOUSE_BODY, r
    return ", ".join(f"{r['name']} {r['tokens']}/{r['cap']}" for r in rows)


def test_the_router_lists_them():
    classes = {c["id"]: c["policy"]["packages"] for c in schema.SHIPPED_ROUTER["classes"]}
    for cid, pk in classes.items():
        assert len(pk) <= 48, (cid, len(pk))
        # summarize never; chat only with the project's switch (schema.CHAT_CORE, PKG3)
        if cid == "summarize" or (cid == "chat" and not schema.CHAT_CORE):
            assert "tandem-core" not in pk, (cid, pk)
        else:
            assert pk and pk[0] == "tandem-core", (cid, pk)
    for cid in ("code_edit", "debug", "design_ui", "drawing", "three_d"):
        assert "render-tools" in classes[cid], cid
    # the prose classes: tandem-core (not chat), then the everyday library (OPS-79)
    for cid in ("writing", "plan", "research", "chat", "summarize"):
        core = {"chat": schema.CHAT_CORE, "summarize": []}.get(cid, ["tandem-core"])
        want = core + ["everyday-core"] + schema.EVERYDAY_ALL       # PKG3: the whole set
        assert classes[cid] == want, (cid, classes[cid])
    assert "creative-coding" in classes["drawing"] and "creative-coding" in classes["three_d"]
    assert "charts" in classes["drawing"] and "charts" in classes["design_ui"]
    assert "voxel-worlds" in classes["three_d"] and "explainer-video" in classes["drawing"]
    assert "eval-judge" not in {p for pk in classes.values() for p in pk}, "the judge is never routed"
    p = schema.SHIPPED_PACKAGES["packages"]
    assert p["render-tools"]["requires_tools"] == ["render"]
    assert p["explainer-video"]["requires_tools"] == ["render_video"] and \
        p["explainer-video"]["fallback"] == "explainer-video-script"
    assert p["tandem-core"]["general"] and p["tandem-core"]["requires_tools"] == []
    with open(os.path.join(ROOT, "docs", "contract", "conductor-v1", "packages.config.schema.json")) as f:
        enum = json.load(f)["properties"]["packages"]["additionalProperties"]["properties"][
            "requires_tools"]["items"]["enum"]
    assert enum == ["shell", "write", "render", "render_video"], enum
    return "router ok"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"  {name:50s} ok   {fn() or ''}", flush=True)
            passed += 1
    print(f"{passed} passed")
