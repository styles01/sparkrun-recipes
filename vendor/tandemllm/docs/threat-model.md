# Threat model

This page is for security researchers and for reviewers of a change. It says where untrusted input enters TandemLLM and which parts matter most. [SECURITY.md](../SECURITY.md) says how to report.

## What runs and where input enters

TandemLLM is a personal AI environment that runs on one machine on a home or office LAN. It has five parts:

| part | what it does | code |
|-|-|-|
| engine server, port 8000 | OpenAI-compatible API on the standard library's `ThreadingHTTPServer`, plus the dashboard API, the conductor, the memory routes and the setup API | `server/`, `engine/` |
| proxy | the same API in front of other OpenAI-compatible backends and cloud APIs | `proxy/` |
| chatd, 127.0.0.1:8016 | the Chat tab: conversations, agent runs, tools, MCP, uploads; the engine server forwards `/v1/chat-app/*` to it | `chatd/`, `server/chat_forward.py` |
| chat sandbox | one Docker or Podman container per chat, `--network none`; its only way out is chatd's egress proxy | `chatd/sandbox.py`, `chatd/egress.py` |
| dashboard | the web UI, served by the engine server | `dashboard/` |

The supported deployment, from [server.md](server.md#the-admin-rule) and `server/auth.py` (AU1): the server listens on the LAN, sometimes behind nginx or a tunnel. The admin password (`QSE_ADMIN_TOKEN`, at least 32 characters) is optional and off by default. With no password, every route answers anyone who reaches the port. With one, the admin routes and every chat route need the bearer token or the `qse_dash` session cookie. One exemption, trusted-local, lets a request in without a token: its TCP peer is loopback and it carries neither `X-Forwarded-For` nor `X-Real-IP`. The chat API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) is open in both modes. chatd trusts only requests that carry the forward key (`X-Tandem-Forward`).

Treat each of these as attacker-controlled:

- every HTTP request that reaches port 8000, chatd's front on 4180, or the proxy: method, path, headers, body, chunked framing, WebSocket upgrades;
- chat messages and their attachments: text files, images (decoded with Pillow in `server/images.py`), PDFs and HTML that the dashboard shows in a pane;
- answers from MCP servers, `web_search` results and pages that `web_fetch` reads;
- the model's output. It picks the tool calls, and those run bash, write files and start servers inside the sandbox. A prompt injection in a fetched page can steer it;
- whatever runs inside a sandbox: it owns `/workspace` and its home, and can swap a file for a symlink or a FIFO while chatd reads it;
- weight files and their headers downloaded from the Hugging Face Hub: safetensors, EXL3 packs, GGUF; and the community index of weight packs (`server/setup/community.py`);
- config files the setup API and the dashboard write: profiles, `model.yaml`, conductor config versions.

## Parts that matter most

In this order:

1. Auth in `server/auth.py`, `server/chat_forward.py` and `chatd/http.py`: any way past the admin rule with a password set, a forged trusted-local, a forged forward key or `X-Tandem-*` header, a session cookie that outlives its token, a CSRF on `/v1/setup/*` (the `same_origin_json` check in `server/setup_api.py`).
2. The sandbox boundary: a container setting that lets a chat reach the host, and egress bypasses in `chatd/egress.py` such as DNS rebinding, IPv4-mapped IPv6, redirects or CONNECT tricks that reach the LAN, the box or port 8000. `ops/sandbox/relay.py`, which runs inside the container, is part of this.
3. Path handling on the host for a chat's files: `chatd/workspace.py` (`SandboxDir`, O_NOFOLLOW walks), uploads (`chatd/attach.py`), snapshots, the files pane, the Open WebUI tool runner (`chatd/owui.py`). Traversal, symlink races and writes outside `<data>/workspaces/<id>/` are in scope.
4. Script from a sandbox page that runs on the dashboard's origin. Previews must stay in an opaque-origin iframe; the CSPs live in `chatd/panes.py` and `chatd/preview.py`, and the port proxy in chatd.
5. HTTP handling we wrote: the chunked reader and keep-alive pool in `proxy/http1.py`, the byte forward and WebSocket splice in `server/chat_forward.py`, the egress proxy's request parsing. Request smuggling, header injection and unbounded reads count.
6. Parsers for untrusted files: GGUF headers (`engine/models/nemotron_h/gguf.py`), safetensors and EXL3 headers (`engine/loader.py`, `engine/models/*/load.py`), images (`server/images.py`). Integer overflow in sizes and offsets, allocations of many GB and out-of-range reads count.
7. The setup API: it downloads, applies and removes weights and runs install steps. A crafted request or community pack that deletes files outside the weight folders, or runs a command, is in scope.

These matter less and stay in scope: the dashboard's own XSS from server data, the conductor and memory routes, the metrics route.

Out of scope: `bench/`, the scripts under `tools/` (measurement, quantising, training; nothing serves them), `docs/`, `manual/`, the GPU kernels in `engine/`, and `tests/`.

## Where to test

Every CPU test is a standalone script: `python tests/test_<name>.py`, exit 0 is a pass. They need the Python packages of install.sh's lock, with a CPU build of torch, and no GPU or Docker. `tests/gpu/` needs a DGX Spark, and `tests/test_sandbox_box.py` needs Docker with the sandbox image.

| area | run it | tests |
|-|-|-|
| server, auth, admin rule | `python server/app.py --fake-engine --port 8000 --served-model qwen38-spark-engine --usage-ledger /tmp/l.sqlite3`; set `QSE_ADMIN_TOKEN` to turn the rule on | `test_auth.py`, `test_fake_engine.py`, `test_server_model_card_auth.py` |
| chat forward and chatd | `python -m chatd --port 8016 --data /tmp/chat`; the key is in `/tmp/chat/forward.key` | `test_chat_forward.py`, `test_chatd.py`, `test_chatd_agent.py` |
| sandbox and egress without Docker | `FakeBoxes` in `tests/chatd_util.py`, `RelayBoxes` in `tests/test_chatd_agent.py` | `test_chatd_sandbox_scope.py`, `test_chatd_sandbox_lifecycle.py`, `test_sandbox_relay_fd.py`, `test_chatd_websearch.py` |
| workspace paths and uploads | chatd as above | `test_chatd_symlinks.py`, `test_chatd_attach.py`, `test_chatd_preview.py` |
| proxy and relay | `python -m proxy --help`; `tests/fake_backend.py` is a backend | `test_proxy_relay.py`, `test_proxy_http1_tls_fd.py`, `test_proxy_model_card_auth.py` |
| setup API | the fake engine above | `test_setup_api.py`, `test_setup_remove.py`, `test_setup_apply.py` |
| loaders and images | no server needed | `test_nemotron_h_gguf.py`, `test_exl3_standalone.py`, `test_images.py` |
| dashboard | `cd dashboard && npm test` | `dashboard/tests/*.test.ts` |

## Not in scope

- Numerical differences in the GPU kernels and quantised weights, and the quality or safety of model output.
- Third-party weights and their contents, beyond how our loaders parse them.
- Rate limits and resource use of the chat API for a client the deployment lets in.
- The installer's need for network and `sudo` on a fresh machine.
