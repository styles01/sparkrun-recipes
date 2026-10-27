"""SRV-667: a recorded Chat-tab conversation replayed through chatd in seconds (tools/chat_trace.py).

A synthetic trace of the owner's reef run's shape (2026-10-20): a long write, the model sending
an earlier write's stub as content, a bash whose exit code is not 0 after a warning on stderr, a
plan-mode turn and its Approve. The replay must hold every check of `chat_trace.check`; with the
call cap at 2 the chat ends idle. With CHAT_TRACES set (a folder of chat snapshots, the box's
~/ornith/replay-r*.json and ~/chat-analysis/chat.json) every one of them is replayed too.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import chat_trace as T  # noqa: E402

from chatd import render  # noqa: E402

PAGE = "".join(f"export const reef{i} = Math.sin({i} * 0.1);\n" for i in range(120))


def _call(k, step, name, args, out, code=None, status="done"):
    return {"type": "tool_call", "id": f"c{k}", "name": name, "step": step, "status": status,
            "input": json.dumps(args), "output": {"content": out, "exit_code": code,
                                                  "is_error": status == "failed"}}


def _trace() -> dict:
    stub = render.wrote_stub("/workspace/reef/js/effects.js", PAGE)
    return {"settings": {"effort": "auto", "thinking": True, "permission_mode": "yolo",
                         "tools": ["bash", "read_file", "write_file", "edit_file", "glob",
                                   "todo_write"], "max_steps": 100},
            "messages": [
                {"role": "user", "parts": [{"type": "text", "text": "Build the reef"}]},
                {"role": "assistant", "parts": [
                    {"type": "thinking", "text": "write the effects", "step": 1},
                    _call(1, 1, "write_file", {"path": "/workspace/reef/js/effects.js",
                                               "content": PAGE}, "wrote"),
                    {"type": "text", "text": "Again:", "step": 2},
                    _call(2, 2, "write_file", {"path": "/workspace/reef/js/effects.js",
                                               "content": stub}, "refused", status="failed"),
                    _call(3, 3, "bash", {"command": "cat > app.js <<'EOF'\nx\nEOF\nnode --check app.js"},
                          "(node:25) Warning: To load an ES module ...\nSyntaxError\n[exit code 1]", 1,
                          status="failed"),
                    _call(4, 4, "bash", {"command": "ls"}, "app.js\n", 0),
                    {"type": "text", "text": "Done.", "step": 5}]},
                {"role": "user", "parts": [{"type": "text", "text": "Split it. Rewrite the plan."},
                                           {"type": "plan", "text": "Plan mode is on"}]},
                {"role": "assistant", "parts": [
                    _call(5, 1, "read_file", {"path": "/workspace/reef/js/effects.js"}, ""),
                    {"type": "text", "text": "1. core.js\n2. main.js", "step": 2}]},
                {"role": "user", "parts": [{"type": "text", "text": T.APPROVED}]},
                {"role": "assistant", "parts": [
                    _call(6, 1, "edit_file", {"path": "/workspace/reef/js/effects.js",
                                              "old": "reef0 =", "new": "reef0 ="}, "edited"),
                    {"type": "text", "text": "Built.", "step": 2}]}]}


def test_a_trace_replays_and_every_check_holds():
    res = T.replay(_trace())
    bad = T.check(res)
    assert not bad, bad
    calls = {p["id"]: p for p in res["calls"]}
    assert len(res["requests"]) >= 7, len(res["requests"])
    stubbed = next(p for p in res["calls"] if p["name"] == "write_file" and p["status"] == "failed")
    assert "is the text that write sent, correct as it is" in stubbed["output"]["content"]
    bash = [p for p in res["calls"] if p["name"] == "bash"]
    assert [p["status"] for p in bash] == ["failed", "done"], [p["status"] for p in bash]
    assert res["statuses"][-1] == "idle", res["statuses"]
    assert calls                                                   # (ids are chatd's own)


def test_the_call_cap_leaves_the_chat_idle():
    res = T.replay(_trace(), cap=2)
    assert res["snapshot"]["messages"][-1]["status"] == "interrupted"
    assert not T.check(res, cap=2), T.check(res, cap=2)
    assert res["snapshot"]["status"] == "idle"


def test_the_recorded_traces():
    folder = os.environ.get("CHAT_TRACES")
    if not folder:
        return "skipped: CHAT_TRACES not set"
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        res = T.replay(T.load(path))
        bad = T.check(res)
        assert not bad, (path, bad[:5])
        out.append(f"{os.path.basename(path)} {len(res['requests'])} requests")
    return ", ".join(out)


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
