# The proxy: one dashboard in front of any OpenAI-compatible backend

The dashboard can sit in front of other engines and cloud APIs, not only this one. Clients send their OpenAI requests to one address; the proxy routes each request by its model id to the backend that serves it and copies the answer back byte for byte. On the way it records what a client could measure: tokens from `usage`, time to first token, decode tokens a second, cached and reasoning tokens when the provider reports them, and the cost at the model's price. One ledger and one dashboard cover every backend.

Backends in this release:

| kind | what it is | routes relayed |
|-|-|-|
| `tandemllm` | another TandemLLM server, by URL | chat, completions, embeddings, Responses |
| `generic` | vLLM, SGLang, llama.cpp, LM Studio, Ollama, any OpenAI-compatible server | chat, completions, embeddings, Responses |
| `openai` | OpenAI's API with your key | chat, completions, embeddings, Responses |
| `deepseek` | DeepSeek's API with your key | chat, completions |

The proxy only observes in this release. It never changes a request's meaning: it sets the model id the backend knows, and it asks a stream for its usage frame when the client did not (then it drops that frame before the client). The conductor's features (task class, effort, packages, the loop and tool checks) come to foreign backends in a later release. Their switches are already stored per model.

## Two ways to run it

### Inside the engine's server

`server/app.py --backends ~/.tandemllm/config.sqlite3` serves the engine's own model as before and relays every request whose model belongs to a backend. A relayed request runs on its own thread and never takes the engine's lock or a place in its queue, so a slow cloud answer cannot hold up the engine and the engine's queue cannot hold up a cloud answer. On the :8000 service, set `QSE_BACKENDS` in `ops/serve.env` and restart; empty means off, and the service is exactly what it was.

First live run, 2026-10-07 on the test board (0.4.1-test), with DeepSeek as the one backend:

    # once: the backend and its key into the config database (the key encrypted, AES-256-GCM;
    # the master key is secrets.key beside the database, mode 0600), from the Backends screen or
    # ConfigDB.create_backend({"id": "deepseek", "kind": "deepseek", "key": ...})
    # ops/serve.env:
    QSE_BACKENDS=${HOME}/.qwen38-spark-engine/backends/config.sqlite3
    TANDEM_BACKENDS=deepseek
    TANDEM_BACKEND_DEEPSEEK_KIND=deepseek
    TANDEM_BACKEND_DEEPSEEK_URL=https://api.deepseek.com/v1

Setting the kind and the URL in the environment locks them on the Backends screen. Without an admin password anyone who reaches the port can edit a backend, and a URL changed to another host would receive the key in the `Authorization` header; with the URL locked that edit answers 422. The key itself stays in the database, never in serve.env. The start log says `[proxy] on: ..., 1 backends (deepseek); TandemLLM stays the engine's`. Measured through the LAN reverse proxy: `/v1/models` listed `TandemLLM`, `deepseek-flash` and `deepseek-v4-pro`; streamed answers carried the reasoning in `reasoning_content` with `thinking` on and none with it off; time to first token 0.65 to 0.85 s; Open WebUI and opencode picked the DeepSeek models from the same connection as `TandemLLM`; a DeepSeek request sent while the engine was decoding a request of its own got its first token in 0.77 s, as when the engine was idle. Every row is in the ledger with its cost.

### On its own

Anyone without a GPU runs the same code. It needs Python 3.11 or newer, the standard library and `cryptography`, and never pulls torch. From a checkout:

    pip install .                       # or '.[host]': adds psutil and NVML for the System view
    tandem-dash                         # 127.0.0.1:8080, ~/.tandemllm/config.sqlite3
    open http://127.0.0.1:8080/dashboard/

`python -m proxy` in the checkout does the same without installing anything but `cryptography`. The page opens on Setup while no backend exists. The engine-only panels (caches, speculation, the conductor, memory) say that this server has no engine.

From a DeepSeek key to a row in the Usage view takes five commands:

    pip install '.[host]'
    printf 'TANDEM_BACKENDS=deepseek\nTANDEM_BACKEND_DEEPSEEK_KIND=deepseek\nTANDEM_BACKEND_DEEPSEEK_KEY=%s\n' "$DEEPSEEK_API_KEY" > dash.env && chmod 600 dash.env
    tandem-dash check --env-file dash.env
    tandem-dash serve --env-file dash.env &
    curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Say hi"}]}'

Then open `http://127.0.0.1:8080/dashboard/#/usage`.

### In docker

`ops/Dockerfile.dash` builds an image of about 70 MB on `python:3.12-slim`, with the built page inside and no compiler or torch in it:

    docker build -f ops/Dockerfile.dash -t tandemllm-dash .
    cp ops/dash.env.example dash.env    # fill in; no quotes around values
    docker run -d --name tandem-dash -p 127.0.0.1:8080:8080 -v tandem-dash:/data \
        --env-file dash.env tandemllm-dash

The config database, the usage ledger and the key file (when `TANDEM_SECRETS_KEY` is not set) live in the `/data` volume. Inside the container the server binds 0.0.0.0, so what the outside reaches is what `-p` publishes: `127.0.0.1:8080` for this machine, a LAN address for a local network. Turn the login on (`TANDEM_DASHBOARD_LOGIN=on` and `TANDEM_ADMIN_TOKEN`) before you publish it to anything but loopback. The container runs as an unprivileged user and answers its health check on `/health`.

### The `tandem-dash` command

| command | what it does |
|-|-|
| `tandem-dash` or `tandem-dash serve` | the server; every flag of `python -m proxy` (`--env-file`, `--config-db`, `--host`, `--port`, `--usage-ledger`, `--dashboard-login`) |
| `tandem-dash check` | reads the environment and a copy of the config database through the schema and prints what would run: the backends, their key hints and locked fields, the rules, the page, the listen address. It writes nothing and exits 1 on an error that names the variable |
| `tandem-dash secrets init` | makes the master key file beside the config database (0600) when no master key answers; `secrets status` says which source answers |
| `tandem-dash backends list` | the backends, one line each, from a copy of the database |
| `tandem-dash hostcollect` | the remote host collector (below) |
| `tandem-dash launch vllm\|sglang` | starts vLLM or SGLang with TandemLLM's plugin inside and registers it as a backend (below) |
| `tandem-dash version` | the installed version |

## Configuration: the GUI and the environment

Everything lives in one SQLite file, the config database: backends, their keys (encrypted), model lists, prices, routing rules, per-model switches, users. The Backends tab edits it: add a backend, test the connection (it shows the latency and the models before anything is saved), edit, delete. Every change is a row in its `audit` table with the time and who made it, and no key value.

The same objects can be declared as `TANDEM_*` variables, for a docker start or any headless box. The environment wins, field by field: at every start the variables are written over the database's rows (a backend is created when missing), and the GUI shows each field a variable sets as locked, with the variable's name on it. A save that changes such a field answers 422 and names the variable. When the variable goes away, the GUI can edit the field again, starting from its last value. A key from the environment stays in memory and is never written to the file.

    TANDEM_BACKENDS=deepseek,vllm1
    TANDEM_BACKEND_DEEPSEEK_KIND=deepseek
    TANDEM_BACKEND_DEEPSEEK_KEY=sk-...
    TANDEM_BACKEND_VLLM1_KIND=generic
    TANDEM_BACKEND_VLLM1_URL=http://gpu-box:8000/v1
    TANDEM_BACKEND_VLLM1_FALLBACK=deepseek
    TANDEM_ROUTE_1="deepseek-*=deepseek"
    TANDEM_PRICE_1="deepseek/deepseek-flash=0.30,0.006,1.2"
    TANDEM_PRICE_1_OFFPEAK="0.15,0.003,0.6"
    TANDEM_API_KEYS="laptop:tdm-...,ci:tdm-..."

`python -m proxy --env-file proxy.env` reads such a file; a variable already in the process environment keeps its value. The full list, with the fields each backend takes (`_KIND`, `_URL`, `_KEY`, `_LABEL`, `_MODELS`, `_MODELS_MODE`, `_FALLBACK`, `_MAX_INFLIGHT`, `_ENABLED`, `_INJECT_USAGE`, `_HARNESS`, `_HARNESS_PROFILE`, `_TIMEOUT_CONNECT`, `_TIMEOUT_FIRST_BYTE`, `_TIMEOUT_IDLE`, `_HOST`, `_HOST_URL`, `_HOST_DCGM_URL`), is at the top of `proxy/config.py`, and `ops/dash.env.example` is a file to start from. One schema (`proxy/config.schema.json`) checks both ways in: a wrong variable stops the start with a message that names it.

## Keys at rest

A backend's key is stored as AES-256-GCM ciphertext. The master key comes from the first of these that answers: `TANDEM_SECRETS_KEY` (base64 of 32 bytes), the OS keychain (`security` on macOS, `secret-tool` on Linux, read only), or `secrets.key` beside the config database, made once at mode 0600. The Backends tab names the source. No API answer, log line, ledger row or audit row holds a key: the GUI gets `key_set`, the last 4 characters and 6 hex digits of the key's sha256, and the key field is write-only.

Encryption at rest protects a copied or backed-up database. It does not protect against a process running as the same user, which can read the key file as easily as the proxy can, and a backup of the whole directory carries the key file too. Keep the master key in the environment or the keychain when backups leave the machine. A lost master key means the keys are typed in again, nothing else.

To keep it in the macOS keychain instead of the file:

    security add-generic-password -s tandemllm -a secrets-key -w "$(cat ~/.tandemllm/secrets.key)"
    rm ~/.tandemllm/secrets.key

## The machine behind a backend

The System view shows the machine each backend runs on, down to the process behind the backend's port. For a vLLM on the same machine that is its API server and its engine core, with their RSS, CPU, threads, uptime and GPU memory. A backend's host setting says where to read it:

| setting | reads | per-process figures |
|-|-|-|
| `auto` (the default) | this machine, when the backend's URL is loopback or one of this machine's addresses; nothing for a cloud API | yes |
| `same` | this machine, whatever the URL says | yes |
| `collector` | `tandem-dash hostcollect` running on the backend's machine, at `host.url` | yes |
| `exporter` | a node_exporter page at `host.url` and a DCGM exporter page at `host.dcgm_url` | no |
| `off` | nothing | |

The server's own process is always listed first, as `engine` in the engine's server and `proxy` in the standalone one. The Backends form sets the host per backend; `TANDEM_BACKEND_<ID>_HOST`, `_HOST_URL` and `_HOST_DCGM_URL` set it from the environment and lock it.

The collector reads psutil and NVML when they are installed (`pip install 'tandemllm[host]'`), else `/proc` and `nvidia-smi`, and macOS without psutil gives the machine's total memory and load only. The process behind a port is found by its listening socket: psutil, then `/proc/net/tcp` and the processes' file descriptors, then `lsof` or `ss`. A process that belongs to another user stays unreadable without root, and the view says so. Command lines show the home as `~`, and the collector masks the value after any flag whose name holds `key`, `token`, `secret`, `passw` or `auth`.

On a GB10 (the DGX Spark) the CPU and the GPU share one memory pool. The driver reports no memory for the GPU itself, so the view shows "unified" there and `MemAvailable` is the figure to watch. The GPU memory of a process is still reported, and it comes out of the same pool as its RSS: the two overlap.

Nothing is sampled while nobody looks. When the System view asks, the collector reads memory, CPU, GPUs, disks and network at most every 5 s and scans the processes at most every 15 s. One sample with a scan took 4.4 ms of CPU on a Mac with psutil, under 0.1 % of one core at that cadence, and each host's footer shows what its own last sample cost.

On another machine, run the collector next to the backend and give both sides the same token:

    TANDEM_COLLECTOR_TOKEN=$(openssl rand -hex 24) tandem-dash hostcollect --host 0.0.0.0 --port 9108

It answers `GET /host` as JSON and `GET /metrics` as Prometheus text (`tandem_host_*`), both behind the bearer token; it refuses to bind beyond loopback without one, because its answer names processes and their command lines. `--watch vllm:8000` reports a port on every answer; the proxy asks for its backends' ports itself. A machine that already runs node_exporter and NVIDIA's DCGM exporter needs nothing new: set the backend's host to `exporter` with both page URLs. That path has no per-process figures, and the panel says so. A remote read runs on its own thread, so a slow or dead host never holds the page; its panel shows the last answer, its age, or the error.

In docker the collector sees the container, not the machine. Run the container with `--pid=host` (and `--gpus all` for NVML) for the machine's view, or run `tandem-dash hostcollect` on the machine itself.

## vLLM and SGLang: their own metrics

The proxy relays a backend of kind `vllm` or `sglang` like any OpenAI-compatible server, and it also reads the engine's own `/metrics` page every 15 s. That page holds what a relay cannot see. It says how many requests run and wait inside the engine and how full the KV cache is. It gives the prefix-cache hit rate and the engine's own time split of each request. Speculation and preemptions are on it too. The Performance view shows one card per such backend, the System view shows the same cards, and a backend's page under Backends shows its engine.

The names come from the source of vLLM v0.30.0 and SGLang v0.5.21, the releases current on 2026-10-04. Each engine has a name table in `proxy/engines.py`. A family the page does not have maps to nothing, and its tile says "not reported by this backend". The proxy skips a family the table does not know. A new release can cost a tile, never a failed read.

| figure | vLLM | SGLang |
|-|-|-|
| running, waiting | `vllm:num_requests_running`, `vllm:num_requests_waiting` | `sglang:num_running_reqs`, `sglang:num_queue_reqs` |
| KV in use | `vllm:kv_cache_usage_perc` (a fraction, 1 is full) | `sglang:token_usage`, and tokens from `sglang:num_used_tokens` of `sglang:max_total_num_tokens` |
| decode rate | `vllm:generation_tokens_total` over the window | `sglang:generation_tokens_total` over the window, and the engine's own `sglang:gen_throughput` |
| prefix-cache hits | `vllm:prefix_cache_hits_total` over `vllm:prefix_cache_queries_total` | `sglang:cached_tokens_total` over `sglang:prompt_tokens_total`, and the engine's own `sglang:cache_hit_rate` |
| TTFT | `vllm:time_to_first_token_seconds` | `sglang:time_to_first_token_seconds` |
| queue, prefill, decode | `vllm:request_queue_time_seconds`, `..._prefill_time_seconds`, `..._decode_time_seconds` | `sglang:queue_time_seconds`; no prefill or decode histogram |
| preemptions | `vllm:num_preemptions_total` | `sglang:num_retracted_requests_total` |
| speculation | `vllm:spec_decode_num_drafts_total`, `..._num_draft_tokens_total`, `..._num_accepted_tokens_total`, `..._num_accepted_tokens_per_pos_total{position}`: acceptance and tokens per step over the window, and acceptance per draft position | `sglang:spec_accept_rate` and `sglang:spec_accept_length`, gauges over SGLang's own last window |

Window figures come from the first and the newest reading of the last 5 minutes. With one reading, or no traffic in the window, they are null. vLLM's families carry an `engine` label per data-parallel engine; the proxy adds them up and takes the mean of a fraction. SGLang reports its scheduler families from the first tensor-parallel rank only. With `--enable-metrics-for-all-schedulers` every rank reports the same numbers, so the proxy keeps rank 0 and drops the rest instead of counting them twice.

The proxy's own `/metrics` carries the mapped families as `qse_backend_*` with a `backend` label, `qse_backend_info{backend,kind,version}` and `qse_backend_scrape_age_seconds`. They never enter the engine's own families. The Performance view reads them from that page; the System view reads the same figures from the backend's `engine` block in the system body.

The flags each engine needs:

| | vLLM | SGLang |
|-|-|-|
| the metrics page | always on | `--enable-metrics`; without it `/metrics` answers 404 and the card says so |
| cached tokens per request | `--enable-prompt-tokens-details` | `--enable-cache-report` |
| reasoning tokens per request | `--reasoning-parser` (the stream's field is `delta.reasoning`) | `--reasoning-parser`; without it SGLang still sends `reasoning_tokens: 0`, which the proxy reads as null when the stream carried reasoning |
| the version | `GET /version` | `GET /server_info` (`/get_server_info` before it), which also gives the speculative settings, the cache report and the reasoning parser |

The proxy does not call either engine's `/health`. On SGLang it runs a one-token generation when the server is idle, and the health poll's `GET /v1/models` already says whether the server answers. After a failed read the proxy waits 30 s, then 60, then 120 s between tries; after a 404 it asks every 120 s.

Two more facts from the v0.30.0 source matter for later layers. vLLM removed `guided_choice` and the other `guided_*` fields: it still answers 200 but ignores them, so a constrained question must use `structured_outputs: {"choice": [...]}`. And vLLM takes a client's `X-Request-Id` as the request id (`chatcmpl-<id>`; the engine's internal id adds 8 random hex characters), while SGLang takes `rid` in the body instead.

## Inside vLLM and SGLang: the plugins and the launcher

The `/metrics` page gives totals. A single request's queue time, its prefill time and whether the scheduler preempted it are not on it. Two small packages in `plugins/` run inside the engine and send those numbers per request to the proxy (layer L2). Install one into the engine's own Python environment; neither has a dependency:

    pip install plugins/vllm        # tandemllm-vllm-plugin: vLLM loads it from vllm.stat_logger_plugins
    pip install plugins/sglang      # tandemllm-sglang-plugin: SGLang loads it from sglang.srt.plugins

The engine needs two variables, `TANDEM_COLLECT_URL` (the proxy's `/v1/collect/<backend id>`) and `TANDEM_COLLECT_TOKEN`. The plugin keeps the last 10,000 finished requests and 300 one-second scheduler samples in memory and posts what is new once a second, or an empty batch every 10 s when the engine is idle. `plugins/README.md` lists what each one reads, the versions it was checked against, and what happens when a release does not fit.

One command does all of it, from nothing to a backend with L2 data:

    tandem-dash launch vllm --model Qwen/Qwen3.5-4B --port 8100 -- --max-model-len 32768
    tandem-dash launch sglang --model Qwen/Qwen3.5-4B --port 30000

`launch` creates the backend in the config database (kind, `http://127.0.0.1:<port>/v1`, host `same`), hands the engine the collect URL and the token, adds the flags the dashboard reads (`--enable-prompt-tokens-details` on vLLM, `--enable-metrics` and `--enable-cache-report` on SGLang), checks that the plugin is installed in the engine's Python, and then replaces itself with the engine, so the pid it records is the engine's. Without the plugin installed it says how to install it and starts the engine at L1. `--dry-run` prints the command, `--collect-url` names another proxy, and everything after `--` goes to the engine unchanged. `launch` is optional: an engine started by hand with the plugin and the two variables gives the same data.

The proxy joins each item to the request's ledger row by an id it sends upstream. vLLM gets `X-Request-Id: px-<hex>` and reports `chatcmpl-px-<hex>`, the row's own id. SGLang gets `rid` in the body and reports it unchanged. A client that sets its own `X-Request-Id` or `rid` keeps it, and the join uses it. The row then holds the engine's own numbers:

| column | vLLM | SGLang |
|-|-|-|
| `queue_ms` | `queued_time` | the scheduler's queue duration |
| `prompt_ms` (prefill) | `prefill_time` | forward entry to the end of prefill |
| `decode_ms`, `decode_tps` | `decode_time`; (tokens - 1) over it | first token to the end; the same rate |
| `prefill_tps` | uncached prompt tokens over the prefill | the same |
| `cached_tokens` | `num_cached_tokens` | `cached_tokens` (device, host, storage) |
| `preemptions` | `num_preemptions` | retractions |
| `engine_finish`, `corrupted` | the engine's finish reason, the NaN check | the finish reason |
| `blocks`, `draft_tokens`, `draft_accepted` | null (vLLM counts speculation per step, not per request) | verify passes, proposed and accepted drafts |
| `l2_join` | `exact`, or `approximate` | the same |

TTFT stays the proxy's own: it is what the client saw. An item usually arrives a second after its row; the ledger then sets the columns on the row in place (`Ledger.amend`, schema version 4). An item that comes first waits for its row. Either waits 10 minutes for the other and is then dropped and counted. A request the proxy cannot tag (the Responses API, SGLang with `n` above 1) is matched by time and token counts: the same backend, the same prompt and completion counts, ends within 3 s of each other, and only when exactly one row fits. Such a row says `approximate`, and the log has one line for it.

The backend's card shows "L2 plugin" while batches arrive and "L1 only" after 60 s without one, with the plugin and engine versions it last saw. Under the tiles it says how many rows got the engine's split, and over the last 5 minutes the preemptions, the KV evictions and, on vLLM, the acceptance per draft position. The proxy's `/metrics` adds `qse_backend_plugin_age_seconds` and `qse_backend_plugin_joins{join}`. A vLLM that loads the second entry point, `tandem_stats_endpoint` (vLLM loads an endpoint plugin only when `VLLM_PLUGINS` names it), also serves `GET /tandem/stats`. When nothing was pushed for 25 s the proxy reads that route every 2 s instead. Items carry sequence numbers, so a batch read twice counts once.

The collect token is `TANDEM_COLLECT_TOKEN` in the proxy's environment, or a random one the proxy stores in its config database at the first need. Every batch needs it, from loopback too. It lets its holder post numbers for a backend that exists, nothing else. The plugins send counts and times only, never a prompt, an answer or a token id.

Not tested yet on real engines: the night pass (`tools/proxy_engines_check.sh --plugins`) installs both plugins into the release images, checks that every row joins and that the plugin's sums over 12 requests equal the engine's own `/metrics` within 1 %, and measures the plugin's cost on vLLM: decode tok/s with and without it, three starts (without, with, without), 20 requests each, gate 1 %.

## Models, names and routing

`GET /v1/models` lists each backend's real models under their own ids (`deepseek-flash`, `Qwen/Qwen3.5-4B`), with `owned_by` set to the backend's id and a `tandem` block (backend, kind, label, upstream id, the provider's name of the model and its largest `max_tokens`, each null when the adapter has none, whether the backend is a cloud API, context length, switches). In the engine's server, `TandemLLM` stays the engine's own. When two backends serve one id, both are listed as `<backend>/<model>` and the bare id is not; the prefixed form is always accepted. A model can get a public id of its own in the GUI, and an output limit: the operator's comes first in `max_output`, before the provider's list and the adapter's table, and a discovery never overwrites it (a relayed vLLM model has neither of the other two).

What a provider's own model list says goes into the card. DeepSeek's `GET /models` (read 2026-10-17) gives each model a `name` (`DeepSeek-V4.1-Flash`, shown as "DeepSeek V4.1 Flash"), `context_window` (1,048,576), `max_output_tokens` (393,216), `input_modalities` (`text`, `image` on deepseek-flash; `text` on deepseek-v4-pro) and `effort` (`supported_levels` low, high, max; `default_level` high). The discovery every 10 minutes stores them per model; the card then carries `capabilities`: `vision` when the input kinds name `image`, `effort_levels`, `effort_default` and `effort_off` (the word that turns thinking off: DeepSeek's `reasoning_effort: none`). The Chat tab offers the provider's levels in its effort chip, sends `reasoning_effort` with one of them, and sends images (uploads and `render_page` screenshots) only to a model that reads them.

A model id a later discovery finds for the first time is stored disabled: it is not listed until it is turned on under Backends. The log says `[proxy] deepseek: new model <id> in /models, disabled until it is turned on under Backends`, and one push goes to the ntfy topic of the monitor's stall watchdog (its `url`, `token_file` and `ca_file`) when one is set. An id is new once; when it leaves the list and comes back, nothing is sent again. On 2026-10-17 `deepseek-v4.1-flash-1017` was in DeepSeek's list for a while at about 01:59 UTC and then gone.

Every relayed answer's `system_fingerprint` (OpenAI's and DeepSeek's name for the backend configuration that wrote it) goes into its ledger row (schema 10) and the request detail on Performance. The model id stays `deepseek-flash` while the snapshot behind it changes; a new fingerprint in the rows shows the change.

A request goes to, in this order: an explicit `<backend>/` prefix, the first enabled routing rule whose glob matches the id, the one backend that serves it, else OpenAI's 404 `model_not_found` (which names the prefixed forms when two backends serve the id).

A backend or a single model may name fallbacks. On a connect error, a 429, 502 or 503, before any byte reached the client, the proxy tries the next backend at once, or the same one again after 1 s, then 2 s (or its `Retry-After`, up to 30 s). At most 2 tries again in all. After the first byte a failure is an `error` frame and `[DONE]`. The ledger row records the backend that answered, `fallback_from`, `retries` and the upstream status.

The proxy adds no limit of its own to how many requests a backend takes at once: a cloud API, vLLM or SGLang gets as many as clients send, and answers its own 429 or 503 when it is full (that answer goes to a fallback as any 429 or 503 does). A backend's `max_inflight` (a number from 1 to 4,096 on its form, or `TANDEM_BACKEND_<ID>_MAX_INFLIGHT`) is a cap you set yourself: past it a request goes to a fallback with room, else it gets 503 with `Retry-After: 1`. Empty, `null` or `none` is no limit, the default. Until 0.3.44-test every backend got 64 without asking; a stored 64 became none when the config file moved to schema 2. How many requests TandemLLM's own engine decodes at once is the engine's setting, not a backend's: `QSE_PARALLEL` in its serve.env ([parallel-requests.md](parallel-requests.md)), shown on the Backends screen under the engine's name. Timeouts per backend (local: connect 5 s, first byte 600 s, idle gap 120 s; cloud: 10, 120, 60 s) end a request with `finish_reason: timeout` and the kind in `error_type`. A client that leaves closes the upstream connection within a quarter second, so the backend stops generating. The standalone proxy (`python -m proxy`) closes a client's connection that sends nothing for 60 s while it reads a request, or sits idle that long between requests (`--idle-timeout`, `TANDEM_IDLE_TIMEOUT_S`; 0 for none). The timeout ends when the answer starts: a stream, the Chat tab's terminal included, is never cut by it.

## What the dashboard shows for a relayed request

Measured or null, never estimated:

| figure | from |
|-|-|
| prompt, completion tokens | the backend's `usage`; null when it sends none |
| cached prompt tokens | `prompt_tokens_details.cached_tokens` (OpenAI; vLLM with `--enable-prompt-tokens-details`; SGLang with `--enable-cache-report`), `prompt_cache_hit_tokens` (DeepSeek), `input_tokens_details.cached_tokens` (Responses) |
| reasoning tokens | `completion_tokens_details.reasoning_tokens` (vLLM with `--reasoning-parser`), SGLang's `reasoning_tokens`; a reported 0 beside streamed reasoning is null |
| TTFT | arrival at the proxy to the first frame with a token; the network and the backend's queue are inside it |
| decode tok/s | (completion tokens - 1) / (last token frame - first token frame); null without `usage`, because a frame is not a token |
| cost | the model's price when the row is written |
| proxy overhead | arrival to the upstream send, plus the first frame's delay through the proxy |

Queue, prefill time, blocks, draft acceptance and the cache source come off an engine's loop and are null for a relayed row; the views say "not reported by this backend". The Usage view filters by backend (`engine` is the engine's own rows) and by user, and shows the cost of the priced models. With a relayed backend in the ledger, Usage has a table of the engine and each relayed backend side by side (requests, tokens, decode and TTFT p50, cost), and Performance draws them apart: relayed requests are red dots of their own on a log axis in Recent requests, and a line of their own beside the engine's in the decode and TTFT history. The cache sources table and the engine's history count the engine's own requests only; the Live panel names a relayed request's backend and says its tokens are counted at its end.

## Every request at once: Performance, Parallel

With several backends, tens of streams run side by side. Performance has two pages: Engine (the engine's own requests, one at a time) and Parallel (`#/performance?tab=parallel`), which shows every request in flight on every backend, our engine included. It reads `GET /v1/dashboard/parallel` once a second (`server/parallel.py`, contract `parallel.schema.json`); the server builds the document at most every 0.4 s for all viewers, and a poll with `since` carries only the new series samples (about 25 KB a poll with 30 streams, under 2 ms to build).

The page has the aggregate decode rate now and its last five minutes stacked by backend, the requests by state (waiting, prefilling, thinking, tool call, writing), TTFT p50 and p90, the cache hit and the cost per minute over the requests finished in the last five minutes; one card per backend; and one lane per request: backend and model, client and session (`x-session-id`), state, tokens (thinking apart), decode tok/s now and on average, TTFT, prefill rate, cache, cost so far, elapsed. A lane keeps its row until a few seconds after it ends; a new request takes the first free row.

Live token counts, and where each comes from:

| backend | live count |
|-|-|
| the engine's own requests | the engine's count, prefill progress and rate, the state |
| a TandemLLM backend behind the proxy | the same: the relay sends `X-Tandem-Relay-Id`, the engine files the request under it, and the proxy reads the engine's `/v1/dashboard/parallel` once a second while it has requests there (it answers a trusted local caller or the admin token). The engine streams blocks of several tokens, so its frames are not a count |
| any other backend | its stream frames, only after they were checked: each finished streamed answer's frames are compared with its `usage.completion_tokens` (within 2 tokens and 1 %), answers with text only and answers with tool calls apart. Two passing answers of a kind and no failing one among its last 8 make those frames a count. DeepSeek's reasoning and content frames are one token each; its tool-call arguments arrive several tokens a frame, so a DeepSeek lane has no live count while it writes a call |

Without a count a lane shows its frames and "not reported"; its tokens and rate come with the usage at the end. A relayed request's prompt and cached tokens arrive with the usage too ("at the end"). Cost so far is the completion tokens so far at the output price, marked `+` until the usage adds the prompt's part.

`tools/parallel_load.py` drives the page: N agent-like sessions (a system prompt, two tools, up to three streamed turns each, tool results fed back) over the models you name, for example `--mix deepseek-flash=10,fake-a=18,TandemLLM=2 --duration 90 --budget-usd 0.5 --board`. It reports TTFT, decode rate, cache hit, throughput and cost per model, stops starting turns at the budget, and with `--board` puts the page's per-second series beside the frames its own clients received.

## Prices and cost

A streamed request with `X-Tandem-Cost: 1` (the Chat tab's chatd sends it) gets the request's cost as one SSE comment just before `data: [DONE]`: `: tandem-cost {"cost_micros": 6140, "backend": "deepseek"}`, every request the harness sent for it summed, the figure its ledger row gets; `cost_micros` is null without a price. SSE readers skip comments; without the header the stream is the backend's, byte for byte.

A price row per model holds USD per million tokens for input, cached input, cache write and output. The cost of a request is `(prompt - cached) x input + cached x cached input + cache write x its price + completion x output`, written into the row when the request ends, so a later price change never rewrites history. A model without a price has a null cost. DeepSeek charges half outside its peak hours (01:00-04:00 and 06:00-10:00 UTC on weekdays); a price row can carry an `offpeak` block, and the DeepSeek adapter picks by the request's UTC hour. Chinese public holidays count as peak here, so their cost reads high.

The proxy never fetches a price list. `proxy/prices.seed.json` holds the list prices of OpenAI and DeepSeek as read from their pages on 2026-10-04; the model rows offer them and apply them only on a click. Check the providers' pages before you bill anyone from these numbers.

## Users and their keys

Under Backends, Users, the admin makes a key per person or tool. The key is shown once and stored as its sha256. From the first key on, the OpenAI routes need one (`Authorization: Bearer tdm-...`) for every model, the engine's own included, from any machine other than this one: a request from loopback without proxy headers passes, so the engine's own helpers keep working. The key's label goes into the ledger's `user` column. `TANDEM_API_KEYS` adds keys from the environment.

## Who may change the configuration

Reading and writing `/v1/config` follow the admin rule ([server.md](server.md#the-admin-rule)). With no admin password (`TANDEM_ADMIN_TOKEN`) set, the default, anyone who reaches the port can change the configuration from the Backends and Setup screens, which ask for no password. With one set, a write needs it as a bearer header or a dashboard session, or comes from the machine itself. The default bind is 127.0.0.1; the page is meant for a trusted network. Never put it on the internet.

## Terms, in short

Using your own key through your own proxy for your own clients is ordinary API use. Putting one key in front of several people makes you the provider toward them: the provider's terms hold the key holder responsible for what those people send, and most forbid reselling access without an agreement. Per-user keys keep each person's usage visible in the ledger. This is not legal advice.

## The harness in front of a backend

The conductor runs in front of other backends too: the router's task class, the effort, the packages and skills as notes, the side requests' caps, the session's memory and its `/s/<token>` address, the library and Laya, the loop guards, E1 and the tool checks, the compaction section and the state lines. In the engine's server it is the engine's own conductor (`--backends` with `--conductor-config`); the standalone proxy keeps one of its own under `--conductor-config` (default `~/.tandemllm/conductor`, `off` for none), with its journal in the usage ledger's file as on the engine. A backend's `harness.mode` turns it on: `off` (the default) is the relay above, byte for byte; `shadow` journals what it would do and changes nothing; `on` acts. `harness.profile` picks the conductor profile for that backend's requests (empty: by client, as on the engine). Both are on the backend's form and in `TANDEM_BACKEND_<ID>_HARNESS` and `_HARNESS_PROFILE`.

Per model, each feature group can be `inherit`, `off`, `shadow` or `on`, and the model can override the mode. A group is never above the mode. What `inherit` means:

| group | what it does over HTTP | local engines | cloud APIs |
|-|-|-|-|
| class | the router's question asked of the backend itself, with its own constraint field (vLLM `structured_outputs.choice`, SGLang `regex`, OpenAI a JSON-schema enum, plain text otherwise) | on | on |
| effort | the backend's own switch: `chat_template_kwargs.enable_thinking`, `reasoning_effort` (OpenAI's o-series and gpt-5 only), `thinking.type` (DeepSeek); a budget is a word, never a cap | on | on |
| packages | the router's note at the end of the newest user message, replayed on later turns at the same message index | on | off |
| side | a client's title or summary call capped (64 or 256 tokens) with thinking off | on | on |
| loops | D1 and D2 on the streamed words; the tool-argument guard on the streamed arguments | on | on |
| e1 | an empty answer to a request with tools gets a second request with a hidden user turn | on | off |
| tools | C1 to C4 on the answer's calls; a regenerate is a second request | on | shadow |
| compaction | the TandemLLM section appended to a compaction request | on | on |
| states | the state lines in the stream | on | on |
| memory | the memory card and notes at the end of the user message, the model's blocks taken out of the stream and written; the background writer as one more request in the backend's idle time (`TANDEM_BACKEND_<ID>_WRITER`: auto is off on a paid model; a daily spend and run cap; docs/memory.md) | on | shadow |

The Backends screen shows, next to each switch, what it resolves to, one sentence on what it costs or loses on that backend, and what its extra requests cost over the last 24 hours (the class questions are ledger rows of the client `harness`; a second request's tokens and cost are in its journal row). The Compare view lists shadow precision per backend.

### What is different from the engine

The note is text at the end of the user's message, inside the `<runtime-note>` frame the router uses, not ids before the end of the turn, so a model can quote it. The replay keeps the bytes of the conversation's head the same from turn to turn, so a backend with a prefix cache (vLLM, SGLang, OpenAI's and DeepSeek's prompt caches) keeps its match: the hash covers each message's role, content, name and tool calls up to the noted one. A client that rewrites its history breaks the hash; the note is gone and the router adds its block again, as on the engine.

The class question is a request of its own, a few hundred prompt tokens each. A new user message can ask up to three (the class, the package among several candidates, a library skill); a tool turn and a repeated message ask none. On the engine the same questions cost nothing beside the turn. With the harness on, `proxy_ms` includes the time the plan took, the questions included.

E1, the regenerate and the reasoning close are second requests. The client gets one stream: the second answer goes out as the continuation of the first, with the first chunk's id, no second role delta and one `[DONE]`. To make that possible the proxy holds the finish frame, the usage frame and `[DONE]` until the turn is settled, and holds the tool-call frames while a check can act. When the client asked for usage, the usage frame it gets is the sum of every request, and so is the ledger row. A reasoning close needs `continue_final_message` (the `vllm` and `sglang` kinds); on any other backend D1 and D2 are journaled in shadow with `no_continue_final_message`. A request that is not streamed gets the plan (the note, the effort, the side cap, the compaction section) but none of the monitor's acts, which read the stream.

### The word-level loop detector

D1's three arms (a repeated pattern, the repeat share of the history, the novelty of the last window) run over words instead of tokens (`server/conductor/monitor/tier0_words.py`), with numbers measured for words, never the token ones copied. `tools/loop_words_fp.py` streams the opencode database's reasoning through it. On 2026-10-04: 2,674 finished turns whose reasoning compresses under 5 times, and 20 loops (turns the token limit cut inside the reasoning, or reasoning that compresses 8 times or more). The strictest of 32 settings fires on 6 of the finished turns and catches 11 of the 20 loops. The engine's own D1 acted on that traffic, so some of those 6 may be loops it closed; the text alone does not say which. The shipped `words` block in `ops/supervisor.yaml` therefore says `calibrated: false`, and every D1 verdict over the proxy stays in shadow until the Compare view's labels show its precision.

### Laya, the library and the session address

The standalone proxy serves `GET /v1/library[/...]`, and `POST /v1/laya/ask` when the conductor's `decider` role names a Jev endpoint over HTTP. Both also answer under `/s/<token>/`, the address the notes give the model, and are journaled to that session. The `classifier` role may name a backend (`backend:<id>/<model>`) to answer the class question instead of the request's own backend, and the `narrator` role the same for the state lines, inside its latency budget; without one the lines are the templates.

## License

The proxy, the collector and the package are part of TandemLLM and carry its licenses: AGPL-3.0-only, or the commercial license in `COMMERCIAL-LICENSE.md`. Under the AGPL, anyone who runs a changed copy of the dashboard as a service for others must offer those users the changed source. The one runtime dependency, `cryptography`, is Apache-2.0 or BSD at the user's choice; the `host` extra adds psutil (BSD-3-Clause) and nvidia-ml-py (BSD-3-Clause). `NOTICE` lists them. None of their code is copied into this repository. The two plugins in `plugins/` are part of TandemLLM too, under the same license, and have no dependencies; vLLM and SGLang (Apache-2.0) are the engines they run in, not part of them.

## Not in this release

An Anthropic adapter, and a release on PyPI (the wheel builds; publishing it is a separate step). The cloud adapters and the harness's second requests are tested against the providers' documented shapes and a fake backend, not against a paid call or a real vLLM or SGLang; `tools/proxy_harness_check.py` runs the harness in front of a live TandemLLM engine. The per-process GPU memory on a discrete GPU is tested against a recorded NVML answer, not yet on such a machine. The vLLM and SGLang tables are tested on pages built from each release's own metric definitions, and the plugins on stand-ins built from each release's own classes; `tools/proxy_engines_check.sh --plugins` checks both against running servers.
