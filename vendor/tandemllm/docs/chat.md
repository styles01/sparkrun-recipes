# The Chat tab and chatd

The dashboard's Chat tab (`#/chat`) is a chat with history whose runs live on the server: close the
browser in the middle of an answer, open the chat on a phone, and it shows the answer at the token
the engine is writing now and keeps streaming. The wire contract is
`dashboard/src/chat/CONTRACT.md`; this page is how it runs on a box.

## What runs where

```
browser  --https-->  engine server :8000  --/v1/chat-app/* (auth checked, byte forward)-->  chatd 127.0.0.1:8016
                                                                                           |  ~/.tandemllm/chat/chat.sqlite3
                                                                                           |  POST /v1/chat/completions --> :8000
```

- **chatd** (`python -m chatd`, the `chatd/` package): a standard-library Python process. It holds
  the stream to the engine, so a run does not end when a browser closes; it writes every change of
  a conversation to an event log with a sequence number, and a device attaches with a snapshot and
  then tails the log from the snapshot's `seq` (SSE, `Last-Event-ID` on a reconnect, `event: reset`
  for a device more than 2,000 events behind).
- **The forward** (`server/chat_forward.py`): the engine server (and the standalone proxy,
  `tandem-dash`) passes `/v1/chat-app/*` to chatd after the dashboard's auth check. chatd answers
  only requests that carry the forward key.
- **The model** is reached as any client reaches it, through `POST /v1/chat/completions` with
  `x-session-id: chat-<conversation id>`, so the engine's park, spill and live rows follow the chat.
  What a model can do comes from `/v1/models` (the `tandem` card, `models/<id>/card.json`), never
  from a model name.

chatd never imports the engine and an engine restart does not touch it: a run that loses the
engine retries with backoff for 10 minutes, then ends `interrupted` with its text kept, and
Continue resumes it. Stopping chatd ends its open runs `interrupted` the same way. A call the
engine refuses for memory (503, "needs ... GiB with every cache given back") is retried once after
its Retry-After, then the chat is compacted and the smaller prompt sent; from then on the chat
compacts at 80 % of the refused size, not only of the window. A 400 that says the request is over
the model's context window (a tool result grew it past the guard's estimate) compacts the chat
and sends the call once more when the compaction made it smaller; when it cut nothing, or the
smaller call is refused too, the run fails with the engine's message.

One step's tool results share the room the model has left (#18): half of what is left under the
compaction line, 2,000 tokens at least, never past the window less twice the answer's room, and
never more than 64,000 characters a result (12,000 in Agents mode). `read_file` cuts at whole lines
and says where to go on ("[truncated at 64,000 characters: lines 1-30 of 317 shown, 639,237 more
bytes; read on with offset=31]"); a line longer than one read goes on with its `column` argument.
A file over 4 MiB is read 4 MiB at a time from the line `offset` names, numbered as in the file,
and the marker names the bytes after the page ("[lines 80000-80002 shown of a 6,749,998-byte
file, 749,845 more bytes after them; read on with offset=80003]"), so no part of it is out of
reach (SRV-651).
`grep` takes a folder or one file: a file named by its path is searched whole, and a folder's
files over 2 MiB are named as not searched instead of counted as no match (SRV-649).
Other results keep their first and last quarter and the whole output under `.tandem/outputs/`. A
request's estimate carries a margin for the text the engine has not counted yet (15 % of it): at
the edge our engine counts the request exactly (`POST /v1/tokenize`) and a request over the window
fails before it is sent, with the count; a relayed model gets a `max_tokens` that leaves the
margin free.

File texts stay out of the request: a finished `write_file`'s content, and an `edit_file`'s old or
new text over 2,000 characters, go to the model as a stub (path, lines, bytes, sha256 and the
`read_file` call that shows it) from the next request on; the store and the thread keep the text.
A worker's (Agents mode) stay whole until a compaction, which stubs them: the engine holds the
rows of the call, and a stub inside its last answer made the next call read that answer again.
A `read_file` or `bash` result over 4,000 characters stays whole until a compaction, which stubs
it (a read names its `read_file` call; a command keeps its first and last lines), so a request only
ever changes at its tail between compactions and the engine's prefix cache holds the rest. A
compaction lands under half the limit that set it off: a long run keeps fewer than its last three
steps until it does. The `[compact]` log line and the Context panel give the chat's request size
before and after, and the summary call's size apart.

## Starting and stopping

chatd starts with the engine and stops with `tandem stop` (OPS-155). `tandem start` and
`ops/start.sh` (the `@reboot` line, the watchdog, the Setup Guide's apply job) start it once the
engine answers `/health`; `tandem stop` stops it first, so open runs end `interrupted`. `tandem
restart`, `ops/stop.sh` and every engine restart (the watchdog's, a hold's, a Setup Guide switch)
leave it running. Setup mode and a hold's restart (`HOLD_RESTART=1`) do not start it. When the
engine answers and chatd does not, the watchdog starts chatd, at most 3 times an hour, and never
restarts the engine for it. A chatd that cannot start is a warning; the engine starts either way.

```
~/TandemLLM/bin/tandem chat status            # pid, interpreter, data folder, URL, /health
~/TandemLLM/bin/tandem chat start | stop | restart
~/TandemLLM/bin/tandem chat logs [n]          # src/logs/chatd.log
```

`tandem status` has one chatd line. `QSE_CHATD=off` in `run/local.env` keeps the engine's start
from starting chatd. The script behind these commands is `~/TandemLLM/src/ops/chatd.sh` (`bash
ops/chatd.sh start|stop|restart|status` in a checkout); `start` exits 1 with the log's last lines
when chatd does not answer `/health`. It runs chatd under the first of: `CHATD_PY`, serve.env's
`PY`, the install's venv (`~/TandemLLM/venv`), `./venv`, `$VIRTUAL_ENV`, `python3` (said so, with a
warning below 3.11); chatd needs only the standard library, and the venv's `cryptography` for the
MCP servers' stored secrets. chatd and the engine read one secrets file, `QSE_SECRETS`
(`~/TandemLLM/state/secrets.env` in an install), which `ops/make-secrets.sh` writes; with no key
there chatd makes `forward.key` in its data folder, which the engine reads too. chatd's address is
`QSE_CHAT_URL` (`http://127.0.0.1:8016`), its data `QSE_CHAT_DATA` (`~/.tandemllm/chat`).

Under the `--systemd` unit `systemctl --user stop tandemllm` stops chatd too, and `systemctl
restart` restarts it (runs end `interrupted`); `tandem restart` keeps runs. `ops/chatd.service` is
a unit for chatd alone, for the installer's layout; with it set `QSE_CHATD=off`.

The sandbox is optional: without a container runtime the chat answers and uses the file tools;
`ops/sandbox/install.sh` adds the rest.

## When the Chat tab says the chat service is not running

The tab probes `GET /v1/chat-app/capabilities` and never falls back to an in-browser mock (VIS-179).
Any other answer than the chat API's is a panel with the cause, the server's message, the command
and Check again; it checks again by itself every 5 s, growing to 30 s, and turns live without a
reload.

| the tab says | why | on the box |
|---|---|---|
| The chat service is not running. | 502 `chat_unavailable`: chatd does not answer | `tandem chat start` |
| chatd is running but does not accept this server's key. | 403 `forbidden` from chatd | `tandem chat restart && tandem restart` |
| This engine build has no Chat tab server. | 404 | run the installer again |
| Cannot reach the engine server. | no answer | `tandem status` |

A live tab that loses chatd shows one banner, "The chat service stopped", and recovers by itself.
`?chat=mock` (and the Vite dev server) loads the in-browser mock, with a "mock data" badge; it is
for dashboard development only.

At most 2 runs talk to the
engine at once (`QSE_CHAT_MAX_RUNS`); the others wait as `queued`. A run that waits for a
permission prompt gives its slot back until the answer comes, and takes one again for its next call. A run on a
cloud model takes no slot: its calls run at the provider, not on the box.

**An engine without the forward.** An engine server from an older build (or another engine
altogether) has no `/v1/chat-app` route. `bash ops/chatd.sh start --front 0.0.0.0:4180` also serves
the built dashboard on :4180 from chatd's own process, forwarding `/v1/chat-app/*` to chatd and
everything else to the engine (`chatd/front.py`). The front has no admin password, so it belongs on
the trusted network only.

## Auth

The dashboard's one switch (the admin password, `QSE_ADMIN_TOKEN`). With none set, every chat route
answers anyone on the trusted network and the tab shows no sign-in. With one set, every chat route
needs the dashboard session cookie or the admin bearer token (or trusted-local); the probe
(`GET /v1/chat-app/capabilities`) still answers and says `signed_in: false`, and the tab shows the
setup guide's sign-in card. A write with `Sec-Fetch-Site: cross-site` is refused.

## Data

- `~/.tandemllm/chat/chat.sqlite3` (WAL, mode 600): conversations, the message tree, runs, the event
  log, the queue, attachments' metadata. Blobs under `blobs/<sha[:2]>/<sha>`.
- A `VACUUM INTO` copy once a day in `backups/`, 7 kept. A deleted chat stays in the trash for 7
  days; text deltas in the event log are dropped after 7 days (the messages keep their text).
- chatd's log has ids, counts and times, never message text.

## The sandbox (phase 2)

Each chat's tools run in its own container, `tandem-chat-<id>` from the image `tandem-sandbox:1`
(build it once: `bash ops/sandbox/build.sh`). chatd speaks the Docker Engine API over a unix socket,
`QSE_SANDBOX_SOCKET` (default `/var/run/docker.sock`); Podman serves the same API, and
`ops/sandbox/install.sh` (root once) sets up rootless Podman under the user `tandem-sbx` with its
nftables rules. Every container is locked down whatever the runtime: uid 1000, no capabilities, a
read-only root, `--network none`, 1 GiB of memory, 4 CPUs, 512 pids, no GPU. Its internet goes
through chatd's per-chat proxy, which refuses the LAN and this box (`sandbox_network` per chat: on
or off).

A chat's tool call gets its sandbox: the sandbox has priority over what the engine holds idle. The
engine keeps itself at its own floors (the KV pool's hard floor and the request admission's floor),
so a sandbox only has to leave room for a prefill's workspace: 2.7 GiB measured for a 203,052-token
prefill of the 27B EXL3 6.0 on the Spark, 3.5 GiB at the full window. A sandbox starts when
MemAvailable minus its limit stays at 4 GiB or more. When it does not, chatd asks the engine for
the difference (`POST /v1/cache/release`): the engine gives back its free allocator blocks, its
caches, the small models and idle conversations (to the NVMe when the spill is on), never what a
running request uses. A call waits for its sandbox at most 60 s; then it fails with the reason and
the numbers, the model reads them as the tool's result and the chat shows them on the call. The
guard asks the engine first and then stops the least recently used idle sandbox under 3 GiB, and
kills them all under 2 GiB; an idle sandbox stops after 30 minutes. The files stay.

A process the kernel kills at the memory limit (the cgroup's `oom_kill` count) fails its call with
"out of memory (4 GiB sandbox limit)". The default was 1 GiB until 0.11.3; headless Chromium drawing
a Three.js app in software (SwiftShader) went past it, so render_page and the verify phase failed on
the apps they check. A container made under another limit takes the current one at its next start.
render_page and the laya kit start Chromium with `--disable-dev-shm-usage` (a container's /dev/shm
is 64 MB) and one renderer. The check browser has WebGL2 (SwiftShader) and no WebGPU, and render_page
says so in each result: with `--enable-unsafe-webgpu --enable-features=Vulkan
--use-vulkan=swiftshader --use-webgpu-adapter=swiftshader` headless Chromium 155 gets an adapter
that draws offscreen but presents nothing to a canvas, so a page's WebGPU path is not checked. What fits in 1 GiB, measured on the Spark (peak of the
container's memory, page cache included): `pip install` of pillow, numpy, imageio-ffmpeg and
piper-tts 549 MiB; `npm i` of typescript, eslint, @playwright/test, react and webpack 531 MiB; a
Vite build 254 MiB; render_page of a 6,000-element page at 1280x800 503 MiB, at 2400x2400 442 MiB,
a WebGL page at 1920x1080 422 MiB; piper text to speech, 55 s of audio, 366 MiB; 60 s of 1280x720
video drawn with PIL and encoded with libx264 415 MiB; the same at 1920x1080, preset medium, 1,010
MiB; pandas with 2 million rows and a matplotlib plot 391 MiB; g++ -O2, four files at once, 784
MiB. What does not: libx264 at 1920x1080 with preset slow and its default threads, 20 g++ at once,
a 1.5 GB numpy array. The sandbox has 4 CPUs while `nproc` counts the whole box, so a tool that
starts a thread per core (x264, make -j, xargs -P) takes more memory than it needs: with
`-threads 4` the 1080p slow encode peaks at 652 MiB. Knobs:

- `QSE_SANDBOX_MEM_GB` (4): each sandbox's memory limit
- `QSE_SANDBOX_FLOOR_GB` (4): MemAvailable a new sandbox must leave after its limit
- `QSE_SANDBOX_WAIT_S` (60): the longest a call waits for its sandbox
- `QSE_SANDBOX_MAX_RUNNING` (4) and `QSE_SANDBOX_TOTAL_GB` (the two multiplied, 16): how many run at
  once, and their limits' sum
- `QSE_SANDBOX_IMAGE` (`tandem-sandbox:1`)

Set them in `run/local.env` (chatd reads serve.env and local.env), then `tandem chat restart`. Without a runtime, starting a sandbox answers 503
`sandbox_unavailable`, bash, render_page and the terminals are not offered, and the file tools work
on the chat's workspace folder.

The unit tests (`tests/test_chatd*.py`, `tests/test_chat_forward.py`) never reach a container
runtime: `tests/chatd_util.py` points `QSE_SANDBOX_SOCKET` at a path that does not exist, and a test
that needs a sandbox installs a fake one. On the Spark, Docker and the sandbox image are there, and
before this the tests found them: `bash` was offered, the tools' order changed and the capabilities
said the sandbox was installed. Only `tests/test_sandbox_box.py` (with `QSE_SANDBOX_BOX=1`) uses
the real runtime.

A deleted chat goes to the trash for 7 days, and its sandbox stops at once. `POST /conversations/{id}/restore` brings the chat back, and its sandbox starts again on its next tool call. When the 7 days end, or on `DELETE /conversations/{id}?forever=1`, chatd removes the container and the chat's folders under `workspaces/`, `homes/`, `egress/` and `snapshots/`. When chatd starts, it removes every container whose chat is gone or in the trash, and logs each one.

MCP servers: stdio servers run inside the chat's sandbox, HTTP servers are called from chatd
through the same egress rules; their secrets are sealed in the store and never come back in an
answer.

chatd reads and writes the chat's workspace and the sandbox's home on the host (the file tools, the
files pane, uploads, snapshots and undo, the Open WebUI runner's files), and the sandbox can change
both at any moment. So no such access follows a symlink (`chatd/workspace.py`, SEC-1): it walks
from the folder's root one directory at a time, each opened with O_NOFOLLOW from the one before,
so a folder swapped for a symlink after a check cannot redirect it; a write goes to a new file
(O_EXCL) in the verified folder and is renamed over the name; a read takes a regular file only (a
FIFO would block chatd). A path with a symlink on the way is refused, and the file tools say so.

## Attachments

An image goes to a model that reads images; a text file goes into the prompt, and its tokens show on
its chip as soon as it is added (the composer uploads it then). When a message is sent, each of its
attachments is saved in the chat's workspace under `uploads/` (`/workspace/uploads/<name>`, a free
name, cut to 255 bytes with its extension kept). When it cannot be saved there (`uploads/` is a
symlink, the disk is full), the message is sent all the same: the attachment keeps its stored copy
and the prompt gets one line saying it is not in the workspace and why. The message's text files
share a budget of a quarter of the window's free room, at most 32,000 tokens; a file past it is not
inlined: the prompt gets one line with its name, size, tokens and path, and the chip marks it "as a
file" before the send. A request that still does not fit names its cause: the attachment, its
message and its tokens, or the largest part.

## Web search

`web_search(query, max_results?)` is a built-in tool. It answers a numbered list of results: title, URL, snippet, and date and engine when the provider gives them. The model cites the URLs and reads a page with `web_fetch`. The search runs in chatd on the host, not in the sandbox.

| provider | request | setup |
|-|-|-|
| `duckduckgo` (default) | `POST https://html.duckduckgo.com/html/` with `q`, `kl` (region), `kp` (safe search), `df` (time) | none |
| `searxng` | `GET <base>/search?format=json` with `categories=general`, `language`, `safesearch`, `time_range` | an instance URL with `json` in `search.formats`; an optional auth header |
| `off` | the tool is not offered | |

- Egress: `chatd/egress.py` `request()` with a host list. DuckDuckGo reaches `html.duckduckgo.com` on public addresses only. SearXNG reaches the configured host and port only, a private address included. chatd refuses a redirect to another host.
- Errors are tool results: DuckDuckGo's rate limit or bot check (HTTP 202), SearXNG's 403 (JSON format off), 401, 429 (limiter), a timeout, an unreachable instance, an answer that is not JSON. No results is not an error.
- Settings (Chat settings, Web search): provider, `searxng_url`, `searxng_auth_header` and its value, `max_results` (1 to 20, default 8), `safe_search` (off, moderate, strict; default moderate), `language` (empty for any, or `de`, `de-DE`), `time_range` (empty, day, week, month, year), `timeout_s` (2 to 30, default 10). chatd seals the auth value with the secrets vault (`proxy/secrets.py`) and never answers it; answers carry `searxng_auth_set`.
- A chat's `web_search` setting overrides `provider`, `max_results`, `safe_search`, `language` and `time_range`; `null` follows the global settings.
- The tool's default policy is always: the user picks the provider in the settings. Set it to ask in the tools list for a prompt per search. The Now line reads `Searching the web for "<query>"`.
- A new chat gets `web_search` in its tools; an existing chat turns it on in the composer's tools menu.
- OpenAI's `web_search_options` field stays refused by the API (docs/server.md): the engine has no search backend of its own.

```text
GET  /v1/chat-app/web-search                 {settings, providers, safe_search, time_ranges, max_results_limit}
PUT  /v1/chat-app/web-search                 partial settings; searxng_auth_value null clears it
POST /v1/chat-app/web-search/test            {settings?, query?} -> {ok, provider, message, count, ms, results}
PATCH /v1/chat-app/conversations/{id}/settings  {web_search: {provider?, ...} | null}
```

## Permission modes

Each chat has a permission mode, and the chat settings hold the default that a new chat gets. A test run on a phone (2026-10-13) asked six times for bash commands such as `grep` and `cp`, which cost 20 s of tapping, and every one of them ran inside the chat's own sandbox. The three modes:

- Ask: every risky call asks first, as before 0.8.2.
- Auto, the default: a call that stays inside the chat's sandbox runs without asking. That covers bash, reading and writing files in `/workspace`, and `render_page`. A call that leaves the sandbox still asks: `web_fetch` to a domain this chat has not fetched before, an MCP tool that its server does not mark `readOnlyHint`, and an imported Open WebUI tool whose file imports a network or process module (`requests`, `urllib`, `socket`, `subprocess` and the like). `rm` or `rmdir` aimed at `/`, `/workspace` or a home directory asks too.
- YOLO: nothing asks, and everything runs.

Some limits hold in every mode. A deny rule refuses the call, and so does a tool set to "never". A tool set to "ask" asks. Plan mode offers only the read tools. The sandbox stays locked down: no LAN, no host, and the internet only through chatd's proxy.

A call that runs without a prompt still leaves a record. Its tool card and `tool.started` carry `approved: "auto"` or `"yolo"` (`"user"` after a prompt, `"rule"` for a rule or an always policy), and the chat's event log gets a `tool.auto_approved` event with the call's one-line summary. The run's mode is on `run.started`, on the assistant message (`permission_mode`) and, after a change during the run, on `run.mode`. A change applies from the next call. The routes:

```text
GET  /v1/chat-app/permissions/mode                       {default, modes}
PUT  /v1/chat-app/permissions/mode                       {default: ask|auto|yolo}
PATCH /v1/chat-app/defaults                              {permission_mode} -> the capabilities' defaults
GET  /v1/chat-app/conversations/{id}/permission-mode     {mode, effective, default}
PUT  /v1/chat-app/conversations/{id}/permission-mode     {mode: ask|auto|yolo}
```

A bash rule names a command's first word, or its first two for programs with subcommands (`git push`, `npm install`). A chained command (`&&`, `||`, `;`, `|`) runs without a prompt when every part has an allow rule, and is refused when any part hits a deny rule. "Always allow" on it saves a rule for each part not yet allowed. A command with `$(...)` or backticks matches no rule.

`PATCH /conversations/{id}/settings` takes `permission_mode` too. A chat from before 0.8.2 has none and runs in the default mode.

## Agents mode: a main agent with workers

A chat setting, `agents`, off by default (VIS-E15). With it on, the main agent talks with the user and plans, and gives every file write and every multi-step build or fix to a worker with the `task` tool. A long agent loop kept every file it wrote and read in its context: one chat on 2026-10-17 grew from 25k to 162k tokens in about 50 minutes, and its compactions read 131,505 tokens again each time. In Agents mode the file text lives in a worker's context and goes with it.

- The main agent has read_file (at most 200 lines or 12,000 characters a call), glob, grep, web_search, web_fetch, render_page, present, the library, todo_write, task and the chat's MCP tools; no bash, write_file or edit_file. Each of its results is cut at 12,000 characters, and a repeated read of an unchanged workspace answers with the earlier call's id.
- A task is `{title, brief, expect}`. The brief has four lines, Goal, Files, Constraints and Check, and no code: a code fence, more than 4 HTML tags, a run of more than 8 lines that look like code, a line of minified code or a "CODE:" block. A brief that breaks this is refused before any worker starts; its length is free. The brief describes the job and the worker writes the code: on 2026-10-17 a main agent wrote a whole clock page in its thinking and sent it in a 7,968-character brief, the worker copied it with a bug it had found, and the main context reached 64,240 tokens. A cap on the brief's length made the main agent cut requirements to pass, so there is none. If a brief still holds code, the worker treats it as a suggestion and fixes the bugs it finds.
- The worker's message holds the task's number and title, the brief, expect, the todo list and then the user's message that started the run, word for word (an attachment by its name and path), so the brief scopes the job and need not repeat the request.
- A worker is a full run on a hidden task conversation: a fresh context (the worker's instructions from `packages/system/workers/`, its tools, the brief, the todo list and the user's message), the chat's sandbox, `/workspace`, permission mode and rules, the parent's run slot, at most 40 model calls (`task_max_steps`), its own compaction and the verify phase. Its engine session is `chat-<chat>-task-<n>`; its memory notes are off. One worker at a time, at most 8 a run (`task_max_per_run`).
- The worker ends with a report in a fixed shape. chatd parses it, sends it back once when a field is missing, builds one from the record when it is missing twice, and at the step cap makes one report-only call whose tool calls never run. Below the report chatd adds its own check: the files the snapshot diff says changed, the calls that ran, and the checks the report claims with no call behind them. The main agent reads at most 8,000 characters of it; the whole is in `/workspace/.tandem/tasks/<n>/report.md`.
- Stop stops both. A message sent while a worker runs reaches the main agent after the report, and the worker as a note on its next tool result. An interrupted worker leaves its parent interrupted; Continue resumes the worker, then the parent, and never runs a cut-off write or command again.

```text
GET   /v1/chat-app/conversations/{id}/tasks              {tasks: Task[]}
GET   /v1/chat-app/conversations/{id}/tasks/{task_id}    {task, messages, seq}
PATCH /v1/chat-app/conversations/{id}/settings           {agents} -> the settings + reread_tokens
PATCH /v1/chat-app/defaults                              {agents} -> the capabilities' defaults
```

The events and types are in `dashboard/src/chat/CONTRACT.md`, section "Tasks (VIS-E15)". With Agents off, every request is the one the single loop sends (`tests/chatd_replay.py`).

## Chat tab (beta): known limitations

The Chat tab is in beta in this release, and a Beta pill beside it says so. We know of these limits and have not fixed them yet:

- The agent loop does not yet work the way Claude Code does (epic VIS-E10). Its prompts are not kept prefix-stable from step to step (SRV-375). Old tool results and old thinking stay in the context until a compaction (SRV-365). A compaction writes a general summary, not the task's state (SRV-373). Nothing notices when the model repeats the same failing approach with different calls (SRV-367). render_page passes on Chromium's console noise (SRV-369). Long tool output has no "see more" paging (SRV-371). Searches run in the main context, with no sub-agents (SRV-381).
- The engine reads a short Flash-Next chat again on its next turn when the answer ended inside a verified block (ENG-235).
- A request that arrives during a prewarm still waits 0.5 to 1 s (SRV-349).
- When memory reaches its ceiling, the engine drops the parked chat without writing it to the NVMe, so its next turn starts cold.
- Images only: there is no video input (VIS-97).
- The 5.05 Flash-Next pack does not load yet.
- Parallel decoding does not raise throughput over one request at a time: three requests together took 27.18 s against 27.03 s one after another, though the later ones start sooner.
- The phone layout's keyboard behaviour is not verified in Safari on an iPhone.

## Not yet

Importing Open WebUI tools and uploading a skill zip answer `not_available` (501). Uploads of
attachments are phase 1's.
