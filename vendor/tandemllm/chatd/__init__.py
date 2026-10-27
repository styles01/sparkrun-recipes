"""chatd: the Chat tab's service. A chat's run lives here, not in the browser.

    python -m chatd --port 8016 --data ~/.tandemllm/chat --engine http://127.0.0.1:8000

WHAT IT IS (VIS-E9, SRV-343/345/349; the plan is the Memo note "Chat tab: plan and architecture
(2026-10-11)", the wire contract is dashboard/src/chat/CONTRACT.md). A torch-free process with the
standard library only, the rule `proxy/` follows. It holds the stream to the engine, so an answer
keeps going when every browser closes; it writes every change of a conversation to an event log in
SQLite with a sequence number per conversation, so a device that opens the chat later loads a
snapshot and tails the log from the snapshot's `seq` and sees the answer at the current token.

    chatd/store.py     the SQLite file (WAL), its schema and forward-only migrations, the backups
    chatd/events.py    the subscribers of a conversation's stream and of the global one
    chatd/engine.py    the model API client: the streamed call, the model list and card, the live tail
    chatd/render.py    the request for a model call, byte-stable for the same conversation
    chatd/runs.py      one run: the call, its parts as they stream, stop, retry, continue
    chatd/context.py   the Context panel's report (ContextReport)
    chatd/service.py   conversations, the message tree, the queue, the routes' logic
    chatd/http.py      the routes, SSE, the forward key

WHO CAN CALL IT. Only the engine server's forward (server/chat_forward.py), which checks the
dashboard's auth first and adds `X-Tandem-Forward: <key>`; chatd answers 403 to anything else. The
key is QSE_CHAT_FORWARD_KEY, else the file `<data>/forward.key` that chatd makes on its first start
(mode 600) and the engine server reads.

WHAT IT NEVER DOES. It never imports the engine and never logs message text: its log lines carry
ids, counts and times (the rule SRV-30 set for the engine). The agent loop, the tools, the
per-chat sandbox, terminals, MCP servers and skills are phase 2 (chatd/agent.py, sandbox.py,
terminals.py, mcp.py, skills.py; docs/chat.md).
"""

__version__ = "0.1.0"
