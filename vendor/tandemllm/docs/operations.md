# Operations

A DGX Spark has 128 GB of memory that the CPU and the GPU share, and a board that runs out of it does not recover by itself: the GPU driver locks, the box stops answering, and only a power cycle brings it back. Most rules on this page exist to keep that from happening on a box nobody can reach.

## One engine per board

Two engines loading side by side will wedge the board. The served engine reserves about 69 GB, and a second one of the same size does not fit. Every script here checks for a running engine first, and a loading engine counts even though it answers nothing yet. The server binds its port only after the weights are in, so for the minutes of a load neither `/health` nor the port shows it. `ops/engines.sh` finds engines by their process instead.

## Memory rules

Keep at least 15 GiB of `MemAvailable` at the worst case of anything you run. Before any change that can raise memory (a cache budget, the context length, more parallel sequences, a new resident copy), write down its worst-case bytes first. During a trial, run a guard that kills your test process when `MemAvailable` drops under 10 GiB.

What the served stack costs:

| item | size |
|-|-|
| NVFP4 weights and the FP8 head, on the device | about 16.4 GiB |
| KV buffer, allocated for the whole context at load | 104.25 KiB a token (106,752 bytes): 64 KiB for the target, 20 KiB for each of the two block drafters, which the served profile loads although StairCut drafts with one, and 256 B of rotary table (28.0 GB, 26.1 GiB, at 262,144 tokens) |
| the whole stack at load, 262,144-token context | 53.45 GB (measured with `tools/mem_audit.py`) |
| the served engine at load, 262,144-token context (2026-10-01, `memory.split`) | 57.0 GiB allocated (weights 16.35, KV pool 16.00, the drafters' context caches 10.00, GDN state 0.14, vision 0.86, the rest 13.7: drafter weights and buffers about 5.9, verify graphs 5.4, warm-up workspace 2.2), 58.2 GiB for the process, 62.7 GiB of `MemAvailable` |
| the same at 131,072 tokens | 44.0 GiB allocated, 45.2 GiB for the process: halving the window frees 13.0 GiB (KV pool 8, drafter caches 5) |
| the caches as they fill (2026-10-01, with that day's budgets: an 8 GiB state store, 4 GiB of anchors, a 2 GiB stash) | state store up to 4 GiB (its budget less the 4 GiB pin), resident prefix up to 10 GiB (anchors, stash and the 4 GiB pin); 75.0 GiB for the process after a short session |
| state store | 2 GiB budget (`CACHE_GB`), at most 1 GiB a snapshot (`STORE_ENTRY_GB`, about 7,000 tokens) |
| resident prefix | 2.5 GiB of anchors (`RESIDENT_GB`), a 1 GiB guest stash (`RESIDENT_STASH_GB`), a 4 GiB pin (`RESIDENT_PIN_GB`), 4 GiB for a parked session (`RESIDENT_PARK_GB`) |

The last two rows are the shipped `ops/serve.env`, which the box these figures come from also runs; its serve.env differs only in running two parallel sequences ([parallel-requests.md](parallel-requests.md#switching-it-on)). `tests/test_docs_defaults.py` checks both rows against the server's flag defaults and `ops/serve.env`.

Page cache counts as available memory but is not always given back in time. After a load, the engine never reads the weight files again, and `ops/start.sh` drops their page cache (`tools/drop_page_cache.py`, `posix_fadvise` with `DONTNEED`, no root needed). Without that drop, an 8,192-row prefill took free memory to 1 GB and the driver logged `NV_ERR_NO_MEMORY`. With it, the same request left 53 GB free.

The FP8 profile needs about 27.5 GiB of weights on the device instead of 16.4. The BF16 checkpoint needs about 54 GB: keep `--max-len` at 65,536 or less with it unless the byte math says otherwise.

## Running the service

`ops/serve.env` is the served configuration. It holds about 60 settings, from paths and the port to every `QWEN38_*` switch, each with the reason for its value in a comment. The scripts read it.

| script | what it does |
|-|-|
| `ops/make-secrets.sh` | creates `secrets.env` with the metrics token (mode 600), once; `--admin` adds the admin password (`QSE_ADMIN_TOKEN`), which turns the admin rule on; `--rotate` replaces what is there |
| `ops/start.sh [--setup [--host ADDR]]` | starts the engine on the port from `serve.env` and waits for `/health`; refuses if an engine is alive or the port is held. `--setup`, or the marker `run/setup-pending`, starts setup mode instead ([setup.md](setup.md)) |
| `ops/stop.sh [seconds]` | sends SIGTERM, waits for the drain (60 s by default), and kills only after that |
| `ops/watchdog.sh` | one health check, for cron every minute; restarts after 3 failures in a row; starts a dead chatd (at most 3 times an hour) |
| `ops/hold.sh <minutes> <command>` | runs a command with the board to itself, then restores the service |
| `ops/gate.sh <label>` | the release gate ([measurement.md](measurement.md)) |
| `ops/roles.sh load ROLE \| up \| down \| settle` | loads the roles beside the engine one at a time through the residency planner; `up` is what `start.sh` runs after `/health` |

`ops/start.sh` is safe to run at any time. If the engine is healthy it says so and exits. It refuses, with a non-zero exit, while any engine is alive or while something else holds the port without answering `/health`.

The watchdog restarts only what it is sure about. A healthy server is left alone, and so is a draining one, because killing a graceful stop halfway is worse than a slow stop. A loading engine is left alone for up to 900 s. Only a server that fails three checks in a row is restarted, so one slow check during a long verify does not bounce a working server.

chatd, the Chat tab's service, is its own process ([chat.md](chat.md)): `ops/start.sh` starts it once the engine answers, `ops/stop.sh` and the watchdog's restart leave it running, `tandem stop` stops it. When the engine answers and chatd does not, the watchdog starts chatd, at most 3 times an hour (one line each in `logs/watchdog.log`), and never restarts the engine for it.

At boot, two cron lines bring the service up:

    @reboot sleep 90 && bash $HOME/<checkout>/ops/start.sh >> $HOME/<checkout>/logs/boot.log 2>&1
    * * * * * bash $HOME/<checkout>/ops/watchdog.sh

On a box that `install.sh` set up, the scripts run from `run/ops`, beside the `serve.env` the installer rendered with this box's paths; the checkout's `src/ops/serve.env` is the release's template. The installer copies `start.sh`, `stop.sh`, `engines.sh`, `roles.sh`, `watchdog.sh`, `hold.sh`, `use-profile.sh`, `make-secrets.sh`, `engine_status.py` and `fetch_weights.py` there on every run; `chatd.sh` stays in `src/ops`. The lines are:

    @reboot sleep 90 && bash $HOME/TandemLLM/bin/tandem start >> $HOME/TandemLLM/src/logs/boot.log 2>&1
    * * * * * bash $HOME/TandemLLM/run/ops/watchdog.sh

Every line runs its script through `bash`, as the scripts call each other: a copy that lost its execute bit (a deploy that did not keep file modes) still runs. `tests/test_no_exec_bit.py` fails on a new direct call (OPS-167).

`bin/tandem` reads `run/ops/serve.env` under `set -e`. A missing `ops/active-profile` is the normal state of a plain install and loads the template's values; a command in the config that fails stops it with the file and the line. Boxes installed before 0.11.2 need one run of the installer to get both, and the scripts above (OPS-159, OPS-161).

## tandem doctor

`bash ~/TandemLLM/bin/tandem doctor` checks an install and repairs what is safe to repair; `--check` only reports, `--json` prints the report (OPS-171). It runs by itself before every `tandem start` and in the check step of every Setup Guide install, which keeps its report on the job. On a checkout it is `python -m server.setup.doctor`. It repairs file modes (against `ops/executables.txt`, the release's executable files), a missing copy in `run/ops`, `bin/tandem`'s mode, stale pid files of the engine and chatd, a hold's pause left behind, a setup marker with an address `start.sh` refuses, and a secrets file readable by others (600). It reports, and never changes: a `run/ops` copy that differs from `src/ops` (run the installer again), a `serve.env` that does not load (the failing line), an interpreter that does not run, a secrets file where the engine does not read it, a fresh hold, and too little disk for a download (`--profile P` or `--need-gb N`). Each run appends to `logs/doctor.log`; the exit status is 0 when nothing is left to look at.

`ops/qwen38-engine.service` is a systemd user unit for the same job. A user unit only survives logout with `loginctl enable-linger`, which needs root. Install one mechanism, never both: two supervisors for one port start the service twice.

## Setup mode and the setup guide

The dashboard's `#/setup` page installs a configuration from the browser ([setup.md](setup.md)). On a new box, `install.sh --setup` or `ops/start.sh --setup` starts setup mode: `server/app.py --setup` with the dashboard and Wren, the guide's small assistant, and no model of Qwen's. It binds 127.0.0.1 unless `--host` says otherwise, and `/health` answers `"mode": "setup"`, which the watchdog takes as healthy. While `run/setup-pending` exists, every start, the reboot's included, comes back in setup mode.

The guide's install writes `run/setup.env`, which `ops/serve.env` reads last (after the weight profile), and switches through `ops/use-profile.sh`. It never edits `ops/serve.env`. A box whose `serve.env` was copied before the setup guide needs the overlay's last lines from the repository's `ops/serve.env`, or the guide refuses to install (`no_overlay_hook`). Reset level 2 removes the overlay; the other reset levels remove caches, weights or history ([setup.md](setup.md), "Reset").

## Profiles

A profile is a `serve.env` file. `ops/serve-fp8.env` and `ops/serve-balanced.env` source `ops/serve.env` and change two lines: `NV` (the NVFP4 files) and `HEAD` (the FP8 head). The balanced profile takes its MLP overlay and head from the published NVFP4 set, the same files `ops/serve.env` loads. Everything else comes from `ops/serve.env`, StairCut included.

That includes StairCut's verify prices, `ops/stair-tables-nvfp4.json`, which were measured on the full-NVFP4 profile. The router corrects their level online, with its per-class block factor (`engine/lenrouter.py`), but the staircase's shape is not measured for either profile, and neither profile's speed is measured on this release.

- To gate a profile: `ops/gate.sh <label> --profile ops/serve-fp8.env`.
- To serve a profile on the service port: copy its two lines into `ops/serve.env` and restart the service through `ops/hold.sh`.
- For the engine tools, export the same values: `QWEN38_NVFP4=` and `QWEN38_FP8_HEAD=` (empty) for plain FP8.
- The FP8 profile also reads its own launch table (`QWEN38_FP8_TILES=ops/fp8-tiles.json`) and its router prices (`--price-table ops/prices.json:fp8-plain`).
- `--fp8-head build` builds the FP8 head at load, without the file.

## One model at a time: the 27B and Flash-Next

The box serves one model at a time; `ops/active-profile` says which. The 27B's profiles are `nvfp4` and `exl3-<bpw>`; Qwen3.8-Flash-Next's are `flashnext-2.05` to `flashnext-6.05`, one per turboderp EXL3 pack (`ops/profiles/_flashnext.env` and one small file each). The same scripts serve both:

- `ops/use-profile.sh flashnext-3.05` switches as for the 27B: it checks or fetches the pack, stops the engine under the box lock, starts the new profile and runs the smoke; a failed smoke starts the previous profile again, 27B or Flash-Next. The two never run at the same time. A Flash-Next profile names no 27B drafter, draft head or base snapshot (`QSE_ASSETS` is empty), and the MemAvailable a start waits for is the profile's own (`QSE_MEM_NEED_GIB`: 55 to 114 GiB by pack; 80 for the 27B).
- The pack comes from one branch of `turboderp/Qwen3.8-Flash-Next-exl3` (`QSE_WEIGHTS_REVISION`), pinned to its commit when the download starts. The branch has no checksum list, so `ops/fetch_weights.py` writes the hub's sha256 of every LFS file to `_hub_manifest.json` and checks against it. The 4.05, 5.05 and 6.05 packs share one n-gram table file (same sha256); a pack fetched next to one that has it hard-links the file instead of downloading 39 GB again.
- `ops/start.sh` passes the profile's `QSE_ENGINE_FLAGS` (`--model <pack> --drafter mtp`) and `--corpus` in place of the 27B's model and drafter flags; with no such key the 27B's command line is the one it was. The server chooses the engine from the checkpoint's config.
- The page-cache drop after a start (`DROP_PAGE_CACHE=1`) adds the pack (`QSE_PAGE_CACHE_DROP`) and leaves a warm n-gram table in the cache (`QSE_PAGE_CACHE_KEEP`): the engine keeps reading it.
- The watchdog and the `@reboot` line run `ops/start.sh` of the tree that serves, so a crash or a reboot brings back the active profile, whichever model it is.
- The status file and page name the profile that is loading (`profile`).
- A start refuses a `--max-len` above the profile's largest window in its registry entry (`max_window`), naming both numbers, before the minutes of loading.

Each profile sets its engine's ceiling and peak (`MEM_CEILING_GIB`, `MEM_PEAK_GIB`), with the arithmetic in the file: the weights, the runtime, slot 0's KV at 262,144 tokens and the caches, then the prefill workspace on top. The n-gram table is not in them. Warm, it is page cache, which every MemAvailable floor check counts as used (the warm table's resident bytes, read with `cachestat`); cold, it is read past the page cache through a row cache of `QWEN38_FN_TABLE_CACHE_GB` (2) in host memory.

## Holding the board

Measurements need the board to themselves, and so does anything that loads a second copy of the model. `ops/hold.sh` makes the safe sequence the easy one:

    flock ~/.qwen38-box.flock bash ops/hold.sh 30 python tools/verify_spec.py ...

It arms a pause file so the watchdog stands back, and keeps it fresh for the whole hold. Then it stops the service and runs the command. Afterwards it restarts the service and waits for `/health`. On every exit path, including a killed command, it removes the pause. The lock file makes two holds wait for each other. A pause file older than 2,400 s is ignored, so a crashed hold cannot leave the service unsupervised for long.

Taking the service down stops everyone who uses it. Hold the board when nobody needs the service, and keep holds short.

## While the engine is away (status page)

A stop, a hold or a start takes the engine away for minutes, and whatever sits in front of it used to answer every URL, the dashboard included, with its own fallback text. `ops/engine_status.py` (stdlib, 0.3.41-test, OPS-65) lets the service say what is going on:

- `ops/start.sh`, `ops/stop.sh` and `ops/hold.sh` write `logs/engine-status.json` (`QSE_STATUS_FILE`): the state (`stopping`, `held`, `starting`, `up`, `failed`, `down`), since when, the reason (a hold's: `HOLD_REASON`, or the first line of the file `HOLD_NOTE` names, else its command) and a hold's budget. A start records how long it took; the median of the last five is the next estimate (300 s before any).
- With `QSE_STATUS_PORT` in serve.env (8009 on the reference box), start.sh and hold.sh keep a small responder on 127.0.0.1 at that port (`engine_status.py ensure`). It answers only what the reverse proxy sends it when the engine does not answer: a page for `/dashboard/` (and any request that accepts HTML), 503 with `Retry-After`, that names the state, the reason, the time and about how long is left, checks `/health` every 5 s and opens the dashboard when the engine is back; and for everything else the OpenAI error shape (503, `Retry-After`, `type: provider_unavailable`, `code: engine_<state>`) with the same facts under `engine`. A file that says `up` while the engine does not answer reads "not answering". An open dashboard shows the same facts in its offline banner. `GET /v1/models` and `/v1/models/{id}` are the exception (0.3.43-test, OPS-67): they answer 200 with the engine's last model list, which `set up` keeps in the status file (a fixed TandemLLM entry when none was recorded), and the state under `tandem_engine`. Open WebUI refreshes its model list from `/v1/models` and dropped TandemLLM from it on a 503, so a chat during a restart said only "Model not found"; now the model stays and the chat gets the 503 with the engine's own sentence. A front proxy that merges several model lists should ask the responder when the engine's list is empty (the reference box's `ai-models-merge` on the Pi does).
- A responder belongs to one tree (serve.env's `REPO`) and one status file, and says which at `GET /_tandem/status` (OPS-173). `ensure`, which start.sh and hold.sh run, keeps a responder of the same tree, status file, release and address, and replaces any other responder: one of another tree, or of a release without the route, found by its listening socket and its command line. A program that is not a responder is reported and left alone. A responder of an old tree once answered an install's restart with that tree's status from two days before. A reason older than 6 h is not shown, and a start or stop older than twice the typical start plus 10 minutes reads "not answering".
- While a Setup Guide install restarts the engine, the responder carries the page (OPS-175): `GET /v1/setup/jobs/{id}` as the server sends it (SSE: the log, the job, each new event, done; or JSON) from the tree's `run/setup-jobs`, and `GET /v1/setup/status`: the engine's facts, the running or newest setup job with its step n of N, and the load's progress from `logs/engine.log` (weights, drafters, warm-up, verify graphs, the port). The server answers `/v1/setup/status` in the same shape while it runs, and `/health` names the responder's port (`status_port`). Its read-only routes answer any origin, so a page on the engine's port reaches it directly when no reverse proxy sends it there; it listens on `QSE_STATUS_HOST`, else the engine's `HOST`. install.sh sets `QSE_STATUS_PORT` to the API port + 9 on every install; `install.sh --uninstall` stops it.
- `python3 ops/engine_status.py show` prints what the responder would say; `stop --port N` stops this tree's responder.

The reverse proxy sends a failed upstream there and keeps a fixed answer for the case where the responder is down too (nginx):

```nginx
# in the server block that proxies the engine
recursive_error_pages on;
error_page 502 503 504 = @engine_away;
location @engine_away {
    proxy_pass http://127.0.0.1:8009;          # QSE_STATUS_PORT (through the same tunnel as :8000)
    proxy_set_header Accept $http_accept;
    error_page 502 503 504 = @engine_away_fixed;
}
location @engine_away_fixed {
    default_type application/json;
    add_header Retry-After 30 always;
    return 503 '{"error":{"message":"TandemLLM is not answering right now (restarting or stopped for maintenance); retry shortly","type":"provider_unavailable","code":"engine_unreachable"}}';
}
```

The engine's own 503s (a drain, a full queue) pass through unchanged: nginx only replaces an answer it did not get (`proxy_intercept_errors` stays off).

A browser that opened the dashboard before keeps working through a restart without the proxy's help (OPS-131): its service worker (`dashboard/pwa/sw.js`) answers a page load that gets 502, 503 or 504, or no answer, with the cached shell. The page then asks the worker (`./sw-away`), says the engine is restarting, polls `/health` every 3 s and reloads when it answers. API calls (`/v1/`) never go through the worker. A first visit has no worker yet and gets the proxy's answer.

## Roles beside the engine (residency)

The engine loads first and holds most of the board. Anything else that should run beside it (a decision model, a librarian, a reader) loads after it, one at a time, through the residency planner (`server/conductor/residency.py`). A role may load when `MemAvailable`, less what the engine may still take up to its declared peak (`MEM_PEAK_GIB`, 77, less what it holds now, from its `/health`; the whole peak when it does not answer), minus its claim stays at or above the floor plus a margin (15 + 6 GiB by default), never during a hold, and never while another load runs (another role, or the engine itself before `/health` answers). A refusal names its reason (`refused_hold`, `refused_load_in_progress`, `refused_memory`, `refused_unknown_role`) and is journaled in `<conductor root>/residency.jsonl`.

A claim is the role's peak, not its idle size, and it is what the role takes from `MemAvailable`, not what `nvidia-smi` shows: arbiter shows 5.1 GB there and took 8.2 GiB of `MemAvailable` (2026-10-01); Laya inside the engine claims 3.0 GiB (OPS-69, the shipped claim since 0.11.3). A role the roles document switches off (the narrator by default since 0.11.3, the reader) is left out of the plan. The floor, the margin, the settle time, the load order and the claims are the conductor's `residency` document, versioned like the others and edited in the dashboard (Conductor, Profiles, Residency). The System view shows the plan: the engine's own split (`memory.split` of `/v1/dashboard/system`, every part read off the object that holds it, and the load stages), `MemAvailable`, every role with its claim and whether it fits now, and the last refusal.

`ops/roles.sh` does the loading. `ROLES_AUTOLOAD` in `serve.env` names the roles `start.sh` brings up after the engine (none today); each role has `ROLE_<NAME>_START`, `ROLE_<NAME>_HEALTH`, `ROLE_<NAME>_RUNTIME` and `ROLE_<NAME>_WAIT`. A role served by vLLM runs BF16 weights with the KV cache unquantised, and its `--gpu-memory-utilization` is the planner's byte math (claim over the board's total, rounded down), never vLLM's default 0.9, which on this board would be the engine's memory too. The script refuses a command that sets another `--dtype`, `--quantization`, a quantised KV cache or its own memory fraction. Roles never load on the Mac. A hold stops the roles before the engine; after a stop, `hold.sh` and the watchdog wait the settle time (5 s by default) before the next 80 GiB load. Nothing unloads on its own when `MemAvailable` falls after a role loaded: the plan shows it.

## Open WebUI

Open WebUI sends no conversation id unless told to. Without one the conductor tells chats apart by their head and their first turn (docs/server.md, the router's session), which cannot separate a regenerated first message from a new chat that opens the same way. Give TandemLLM's connection the chat id as a header: Admin Settings, Connections, the TandemLLM connection's settings, Headers:

```json
{"X-OpenWebUI-Chat-Id": "{{CHAT_ID}}"}
```

The placeholder is filled per request (Open WebUI 0.11 or later, `parse_custom_headers`); the model list and the task requests get it too, which costs nothing (an empty value is ignored, a title request is a side request with no session). Prefer this to `ENABLE_FORWARD_USER_INFO_HEADERS=true`, which sends the same header but also four fields about the user (`name`, `id`, `email`, `role`) on every connection, external providers included. The engine reads `X-OpenWebUI-Chat-Id` either way; a decision row with `extra.session` = `id` shows it arrived.

## Settings

`engine/settings.py` is the one registry of every `QWEN38_*` switch the engine reads (110 of them), with its default and the module that reads it. `tests/test_settings.py` fails if any module reads a switch another way. To see what is in effect:

    python -c "from engine.settings import SETTINGS; import json; print(json.dumps(SETTINGS.describe(), indent=1))"

Server behaviour (ports, limits, caches, reasoning) is set by command-line flags. `python server/app.py --help` lists them, and `ops/start.sh` builds them from `serve.env`.

## Release checklist

Every release runs these on the box before its tag, and ships only when all pass:

1. `ops/gate.sh <label>`: the CPU and GPU tests, identity, agreement, the rows ([measurement.md](measurement.md)).
2. `tools/client_smoke.py` and `tools/api_text_check.py` against the candidate.
3. `bash tools/release_gate_setup.sh` (OPS-177), mandatory: a fresh scratch install of the candidate runs the whole Setup Guide flow beside the live engine without touching it. Run A installs with `install.sh --setup`, applies a profile through the Setup Guide's API while a phone-sized browser (WebKit at 390 px in the `mcr.microsoft.com/playwright:v1.63.0-noble` container) follows the job, checks that the restart card shows while the engine restarts and that the finished screen with Start chatting shows within 10 s of the engine answering, checks the job, `/health` and a chat, and removes a second profile's weights. Run B strips the execute bit from `start.sh`, `stop.sh` and `use-profile.sh` first and must still install. The scratch installs use spare ports (never 8000, 8009 or 8016), no chatd, the CPU fake engine and a few KB of stand-in weights, a HOME of their own, and an interpreter copy the live box's `ops/engines.sh` does not count; every process they start is stopped at the end. The report and the screenshots are in `results/gate-setup/<utc>/`.
4. A real agent session with its own tool calls.

## Deploying a new version

1. Copy the current serving directory to a dated backup (`<checkout>.bak-YYYYMMDD-HHMM`).
2. Copy the new files in, keeping the local `serve.env` values that are site-specific (paths, port).
3. Restart through `ops/hold.sh`, so the watchdog does not race the restart.
4. Check that `/health` says `ok` and that `VERSION` is the new one, then send a short request.

Run the gate and the client smoke set before step 1 ([measurement.md](measurement.md)). A deploy that fails step 4 goes back to the backup.

On a box that `install.sh` set up, the installer is the deploy: `bash ~/TandemLLM/src/install.sh --yes --ref <new tag>` checks out the tag, installs the new lock and rewrites `run/ops/serve.env` from the release's template; `run/local.env` stays. A box installed with `--setup` and then switched to a profile counts as configured, and `--setup` alone refuses it. Run `install.sh --setup --force-setup --yes --ref <new tag>`, then `ops/use-profile.sh <profile>` again; the profile's files are on disk already.

## Monitoring

`/health` returns the status, the queue counters (running, waiting, served, refused, errors, timeouts, abandoned), the cache report and the allocator's memory figures. On this board `nvidia-smi` reports no used memory, so `torch.cuda.memory_allocated` and `memory_reserved` are the numbers to watch. Allocated memory that climbs across requests is a leak. Reserved memory that climbs while allocated memory stays flat is fragmentation.

`/metrics` serves Prometheus metrics, listed in [server/METRICS.md](../server/METRICS.md). `ops/monitoring/` has a scrape config, alert rules with their tests, and a Grafana dashboard generated by `tools/grafana_dashboard.py`.

## When something is wrong

- An answer stops in the middle. Find the request's log line. `finish=length` hit the token limit, `timeout` the wall clock, `error` an exception (its traceback is in the log), and `abandoned` means the client hung up.
- An agent turn waits minutes before the first token. Check `cached_tokens` for that request in the usage ledger. On a growing conversation it should be close to the previous turn's prompt. If it is not, read the `resident` block of `/v1/cache/stats`.
- The server answers 503 or 429 while it looks idle. Read `inflight` in `/health`. A 503 means a full queue, and a 429 a request without a stream that waited longer than the queue timeout (a streamed chat request waits up to the request timeout and gets `: waiting` comments; docs/server.md). A high `waiting` count with nothing running is a bug.
- The port answers but nothing generates. The watchdog handles three failed checks. By hand: `bash ops/stop.sh 30 && bash ops/start.sh`.
