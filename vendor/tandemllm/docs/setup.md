# The setup guide

The dashboard's `#/setup` page walks through an install in eight steps, from picking a model and its weights to the tools beside it, the parallel requests and the memory they take. The last two steps show what will change and then install it. A small assistant called Wren answers questions on the side and can change the page when you ask. The server half lives in `server/setup/` and `server/setup_api.py` and answers under `/v1/setup/*`. None of it is on the serving path: a chat request never reaches it, and the engine imports it only when a `/v1/setup` route is called.

## Setup mode

A new box has no configured engine yet. Two commands install TandemLLM and start setup mode, with no model loaded:

```bash
curl -fsSLO https://raw.githubusercontent.com/0xBakeer/TandemLLM/v0.13.0/install.sh
bash install.sh --setup --yes
```

`install.sh --setup` installs the code and the Python environment, downloads only Wren's model (Qwen3.5-2B, 4.3 GB), writes the config with no weights, and starts setup mode. Then open the Setup Guide in a browser and pick a model:

- On the box itself: `http://127.0.0.1:8000/dashboard/#/setup`.
- From another computer: setup mode listens on 127.0.0.1 only, so open an SSH tunnel first, `ssh -L 8000:127.0.0.1:8000 <user>@<box>`, and then open `http://127.0.0.1:8000/dashboard/#/setup` on that computer.
- On your network: install with `--host 0.0.0.0` (`bash install.sh --setup --yes --host 0.0.0.0`) and open `http://<box>:8000/dashboard/#/setup` from any computer that reaches the box. Until an admin password is set, anyone who reaches the port can run the setup.

The installer prints the addresses when it finishes, and the dashboard opens on the Setup Guide by itself while the box is in setup mode.

On a checkout, the same mode starts with `ops/start.sh --setup` (127.0.0.1 unless `--host` names another address). Setup mode runs `server/app.py --setup`: no model of Qwen's, no caches, no state store, no drafters, no conductor and no ledger. It answers the dashboard, `/v1/setup/*`, `/health` (`"mode": "setup"`) and an empty `/v1/models`, loads Wren at the start, and gives every other route a 503 with the code `setup_mode`. The watchdog sees its `/health` answer 200 and leaves it alone.

The installer writes the marker `run/setup-pending` (with the host to bind). While it exists, a plain `ops/start.sh` (the reboot's, the watchdog's) starts setup mode again. The install job removes it before it starts the configured engine, and puts it back if that engine does not come up. Setup mode does not start chatd; the configured engine's start does, so the Chat tab works once the guide is done.

Until an admin password is set, anyone who reaches the port can run the setup, apply it and reset the box. The review step says so. `ops/make-secrets.sh --admin` sets one; the setup page's "generate" option does the same on a box that has no `secrets.env` yet.

## Advanced: from the command line

Since 0.8.1 a new DGX Spark needs no files from anyone and no Hugging Face token. Every file a profile reads is public on Hugging Face. The Setup Guide runs `ops/use-profile.sh` for you; to pick the weights from a shell instead, install with the two commands above, then:

```bash
bash ~/TandemLLM/run/ops/use-profile.sh --list                 # every profile, the active one marked
bash ~/TandemLLM/run/ops/use-profile.sh <profile> --dry-run    # what it would download: each URL and size
bash ~/TandemLLM/run/ops/use-profile.sh <profile>              # download, check, leave setup mode, start
```

Run the copy in `~/TandemLLM/run/ops/`, which reads the `serve.env` the installer wrote. The switch downloads what the profile needs, checks it, and starts the engine on port 8000. Two examples:

| profile | downloads | size |
|-|-|-|
| `flashnext-3.05` | the pack's branch of `turboderp/Qwen3.8-Flash-Next-exl3` | 85.1 GB |
| `exl3-4.5` | `0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-4.5`, the K3 drafters and the K4 draft head (`...-EXL3-drafters`, into `~/.qwen38-spark-engine/weights`), the BF16 drafters (`...-DFlash2-b8` and `-b16`, into `~/TandemLLM/src/train`) and the tensors of the base checkpoint `Qwen/Qwen3.8-27B-FP8` an EXL3 load reads (`base-plain`, into `~/.qwen38-spark-engine/weights/base-plain`; none when the full checkpoint is in the Hugging Face cache) | 18.4 + 1.9 + 7.7 + 3.54 GB |

The other EXL3 sizes download 12.0 to 23.3 GB of weights and share the rest. Before a download starts, the script checks that the disk keeps 60 GB free after it. It checks every weight file against the sha256 the hub lists, and a folder appears only when its files are all there. If a download stops, run the same command again: it continues and keeps the files that finished. The engine needs about 80 GiB free for a 27B profile and 70 GiB for `flashnext-3.05`. The switch stops setup mode first.

On a box in setup mode the switch removes `run/setup-pending`, so the engine starts with the profile. If the profile does not come up, the marker comes back and setup mode starts again. The setup page does the same through its install job.

The `curl` line fetches the installer from the tag `v0.13.0`, and the installer checks out the same tag from the repository `--repo` names, `https://github.com/0xBakeer/TandemLLM.git` by default. Where the tag is missing, `curl` fails with a 404, and `install.sh` stops at its clone step with "no tag, branch or commit 'v0.13.0'"; `--ref` picks another release. The manual ([`tandemllm-manual.pdf`](../manual/tandemllm-manual.pdf), chapter "Installing TandemLLM") has the details.

## The model registry

Each model the guide offers is a folder under `models/` with a `model.yaml` ([models/README.md](../models/README.md) lists the fields). The 27B's file, `models/qwen3.8-27b/model.yaml`, is the first; every number in it says where it comes from:

- The weights and the drafters per profile are the `[memory] load stages` lines of the 0.6.2 runs of 2026-10-10, GiB allocated, one run per EXL3 size. NVFP4's are from the 0.5.0 load of 2026-10-09.
- The tools' memory is the dashboard's live memory split of the same evening (the reference box on `exl3-6.0` with 4 slots, the narrator, Laya and the shadow reader loaded). The registry's parts add up to that split's 59.34 GiB in use within 0.1 GiB.
- The measured table is the 0.6.2 run itself; `models/qwen3.8-27b/measurements/0.6.2-summary.json` is its summary file, and a test holds the two equal. The NVFP4 row is from build 0.4.8 and is marked not comparable.

Memory is in GiB (2^30 bytes) and disk in GB (10^9 bytes) everywhere in the registry and the API. `python -m server.setup.registry --check` checks a file.

Every weight profile names who made it (`made_by`): TandemLLM's own packs (0xBakeer), the vendor's release (NVIDIA, LiquidAI), or a community author with a GitHub login (turboderp for Qwen3.8-Flash-Next). Each weight card shows the pack's Hugging Face repository as owner/name, linked, and its maker.

Packs that other people published sit beside our profiles as community packs, one JSON file each under `models/<model>/community/`. The weights step lists them after our own, as profiles named `community/<id>`. A pack that one of the model's own profiles serves names that profile (`profile`): turboderp's Flash-Next branches are community cards whose Use selects `flashnext-<bpw>`, and they come with the release in `models/community-builtin.json`, not from GitHub. The page fetches the list from GitHub when it opens and when you press Refresh, never on a timer, and an unchanged list costs one 304 answer. Without the network it shows the last list it fetched, or the one that came with the release. A 404 from GitHub means the public repository has published no list yet: the section shows the note "no published list yet" and the release's list, with no error. A real failure is one short line, such as "GitHub could not be reached", and the server's log has the URL. On the command line, `ops/use-profile.sh community/<id>` runs a pack as it runs our profiles, with `--dry-run` and the same disk, memory and smoke checks. [docs/community-weights.md](community-weights.md) says how to add a pack.

### The KV bytes a token

One token of context window costs 106,752 bytes, which is 104.25 KiB or 106.75 kB: 64 KiB of the model's attention KV (16 attention layers, 4 KV heads of 256, K and V in BF16), 20 KiB for each of the two drafters' context KV (5 layers, 8 KV heads of 128) and 256 B of rotary table. At 262,144 tokens that is 28.0 GB, or 26.1 GiB. The docs said "106.5 KiB" until 0.6.2: that was the kB figure with the KiB unit. With the KV pool on (`QSE_KV_POOL=on`) a slot reserves its window as addresses and commits pages only for the rows its conversation writes.

### The CUDA graphs and the side slots

The 6.63 GiB of CUDA graphs belong to slot 0. A side slot captures its own verify graphs, about 7.5 GiB each, only with `QSE_SIDE_GRAPHS=on`; the guide always writes it off, as the reference box runs, and a side slot then costs 1.19 GiB beyond its KV (the load stage `side_slots` took 3.58 GiB for three on 2026-10-10).

### Flash-Next, the second model

`models/qwen3.8-flash-next/model.yaml` gives the models step its second tile. Its five profiles are the EXL3 packs (2.05 to 6.05 bits a weight). Fields the 27B's entry does not use, all optional:

- `hf_revision`: the branch of the repo the pack is on.
- `max_window`: the largest window the pack holds on the device preset; null when it fits none (6.05). The overview line and the window buttons follow the selected profile (buttons above it are disabled, with the reason), the estimate refuses a longer window (422 `out_of_range`), and the engine refuses one at start.
- `table`: the pack's n-gram table, its size and policy. Warm adds the table as a fixed part ("n-gram table (page cache, kept warm)"); cold adds its row cache instead.
- `components_gib`: a tool's memory under this profile, where it differs from the tool's own figure.
- `memory.per_window_parts`: parts that grow with slot 0's window, each with an optional `max_gib` where it stops. Flash-Next's prefill workspace grows 76,309 bytes a context token up to 10.5 GiB (10.46 GiB measured at a 147,149-token prefill on 3.05, 10.47 at 261,549); its warm-up buffers, graphs and draft-layer cache take 32,230 bytes a token of window. With these and the engine's part calibrated on the SPD-99 runs, the estimate predicts the lowest MemAvailable of the 2.05, 3.05 and 4.05 runs at 262,144 tokens within 0.7 GiB.

Rows of the measured table and every `max_window` come from runs on the box only.

## The memory estimate

`server/setup/estimate.py` holds the one formula; `models/estimate-cases.json` pins it with 58 cases (47 for the 27B, 11 for Flash-Next), and the dashboard's TypeScript copy must give the same answers.

- The parts: the OS and other programs (the device's `reserved_gib`), the weights and the drafters of the profile, the CUDA graphs, the engine's warm-up workspace, the allocator's reserve, 1.19 GiB per side slot, and every tool that is on.
- The conversations (KV): with the pool, each slot at 32,768 tokens or its window if smaller (`typical_fill_tokens`); without it, every window in full, since it is allocated at load.
- `budget_gb` is the device's memory less the floor (`QSE_PARALLEL_FLOOR_GB`, 15). `fits` is the total at or under the budget, compared to the cent. `fits_worst` asks the same with every slot at its full window; when it is false the engine makes a new request wait for memory instead of running out.
- `context_tokens_fit` is what the room left after the fixed parts holds, in tokens, over all slots.

`exl3-4.5` with the recommended tools at 4 slots of 262,144: 67.35 GiB of a 106.63 GiB budget, 13.03 of it conversations at 32,768 tokens a slot. Every slot full would take 104.25 GiB of KV, so `fits_worst` is false.

## Wren

Wren is Qwen3.5-2B (Qwen3.5-0.8B when the 2B is not on disk) on the engine's own forward, in the plainest setup the engine has: one sequence, a window of 8,192 rows, thinking off, greedy, no caches. Its persona and rules are `server/setup/wren_system.md`.

- Each turn it reads the page's state (fields named like a secret arrive as `<set>`), the box's free memory and disk and the profile it runs, and the four best sections of a BM25 search over `docs/*.md`, the manual's chapters, `models/*/model.yaml` and the packages that describe TandemLLM. It also reads the current step: the ids a tool call may name there, and on the three steps that show the measured table, the table itself with the fastest profile for each task.
- It can ask the page to `set_field`, `toggle`, `select_profile`, `go_to_step`, `next`, `previous` and `explain`. `explain` takes only what the page can highlight, and a step's name is not one of them. The browser runs every call; the server checks each one against the step schema first. An invalid change never reaches the browser: Wren sees why and tries once more. If that fails too, Wren answers in words with no tools in the prompt, and the page shows one notice that nothing changed. A highlight it cannot make goes nowhere.
- A turn always ends in words. After a call the page sends the result back and asks again, at most 4 rounds. A change the page refuses gets one repair, and the last round asks for words only (`"tools": false` in the chat body), so the server gives the model no tools and drops any call it still writes.
- On a phone the sheet stays open while Wren works. A change shows in the sheet as a toast with Show me, which closes the sheet, scrolls to the field and highlights it. A refused call leaves the sheet alone.
- It cannot finish, apply, reset, load or unload anything.
- On a serving engine it loads only when asked (`POST /v1/setup/assistant/load`), through the residency planner. If the narrator is loaded with the same model, Wren borrows the narrator's copy and costs no memory. It is the first thing given back when the KV pool needs memory, and it unloads itself after 5 minutes without a poll or a chat.
- It generates in slices of 150 ms while no request runs or waits, so a request that arrives waits at most one slice.
- The chat's text is never logged.

On the reference box (2026-10-11, `tests/gpu/setup_wren_gpu.py`) Wren's own copy loads in 22 s and holds 3.62 GiB on the device; MemAvailable drops 5.0 to 5.2 GiB at the first load, of which 1.1 to 1.5 GiB is the process's CUDA context, which stays. Later loads and unloads return within the box's own noise (0.2 to 0.4 GiB), with the process's RSS and the allocator flat. Borrowing the narrator costs 0.03 to 0.06 GiB, and the narrator's next line is the same to the token. A turn takes 1 to 3 s, the first words after 0.4 to 0.7 s. On ten plain requests and three questions, the 2B made the right call or answered 10 times of 13; Qwen3.5-4B (`QSE_WREN_MODEL=qwen3.5-4b`) 12 of 13, at about 2.5 times the time a turn. The 2B sometimes mixes the numbers of two profiles in an answer, so the page's own numbers stay the reference.

0.7.1 on a phone (2026-10-11 10:13): after an answer on the Model step, the 2B called `explain("overview")`, the name of the step. The server took it as valid, the page refused it, and the page asked again four times. Each round brought two more refused calls. The question about coding tasks got eight notices and the 4-round stop instead of an answer, and the sheet dropped to a peek once per call. Measured again on the box with 0.7.2 (`tests/gpu/setup_wren_gpu.py cases`, the 2B at the narrator's 4,096 rows): 15 of 15 right on ten requests for a change and five questions, the two of the report among them. No call was invalid, Wren said what it changed after every change, and a turn took 0.7 to 1.6 s. The question about coding tasks now gets `exl3-2.5` at 124.17 tokens a second, in one round.

## The install job

`POST /v1/setup/plan` returns the review step's facts: what is wrong, what changes against `ops/serve.env` (each key whose effective value differs, old and new; a key the current `run/setup.env` sets and the new one drops is `removed`), the commands that will run and the estimate. `POST /v1/setup/apply` checks the selection again and starts a job; `GET /v1/setup/jobs/{id}` streams it (SSE): the log lines on disk, the job as it is now, then each event as it comes; `Last-Event-ID` resumes, `?format=json` gives the job.

The job runs a fixed list of steps from `server/setup/apply.py`, and nothing a client sends becomes a command:

| step | what it runs |
|-|-|
| check | `tandem doctor` first (file modes, the `run/ops` copies, stale pid files, the marker; its report on the job as `doctor`, OPS-171), then the selection, the memory, the disk, the files the profile needs, the HF token (only a private repository needs one; every published profile is public), the box lock |
| unload_assistant | Wren is unloaded before the job starts |
| fetch | `bash ops/use-profile.sh <profile> --fetch-only`: the profile's weights and the files it needs beside them (the `assets_required` with a `fetch` list in the registry), and the models of the tools the selection turns on (`QSE_FETCH_ASSETS`: the shadow reader's, the narrator's, Laya's; each once); skipped when all of them are on disk |
| preflight | before anything stops (OPS-169): the scripts the switch runs are there and parse (`bash -n`; the Python ones compile), `serve.env` with the profile and the overlay the job will write sources under `set -u` as `start.sh` sources it, the weights are complete (the folder's files and the hub manifest's sizes), memory (MemAvailable now and what the running engine gives back cover the profile's need), the new port is free, no box lock and no hold, the engine's and chatd's interpreters run; a failure ends the job here, in plain words, and the running setup keeps running |
| write_config | first `waiting` until no request runs or waits on the engine (10 minutes at most, else nothing changes); then `run/setup.env`, every value quoted, the old one kept as `run/setup.env.bak-<time>`; with a port change, `bash ops/stop.sh 120` first, and a failure after that stop goes through the rollback; with "generate", `bash ops/make-secrets.sh --admin` |
| switch | `bash ops/use-profile.sh <profile>` with `QSE_NO_ROLLBACK=1`: stop, start, smoke; a failure is this job's rollback, not the script's (which would start the old profile under the new overlay); in setup mode `run/setup-pending` goes first, a copy kept in the job's folder |
| smoke | `/health` names the profile and one chat answers (512 tokens at most, as use-profile.sh asks); then the tools with a model of their own, from `/health` `roles` and the decision models, polled up to 90 s: one that failed to load is a warning on the job (`warnings`, with its error), one waiting for memory a note (`notes`); the job still ends `done` |
| rollback | on a failed switch or smoke: with a port change the engine on the new port is stopped first, then the old `run/setup.env` comes back and the previous profile starts with its own served name in the smoke (in setup mode, the marker comes back unchanged, its HOST too, and setup mode starts again; a runner that crashes puts the marker back as well) |

`ops/serve.env` is never edited. It reads `run/setup.env` after the weight profile, so the overlay never sets the weights. The overlay leaves out a key that serve.env leaves unset when its value is what the engine runs with anyway, so the engine's environment does not change for nothing: the disk tier's fingerprint counts every `QWEN38_*` variable that is set, and a new one sweeps that profile's saved heads and parked chats at the next start. A box whose `serve.env` predates the setup guide must get those last lines first; the plan says so.

The job runs as a process of its own, because the switch replaces the server that started it. Its state is in `run/setup-jobs/<id>/`, and the status responder (`ops/engine_status.py`) answers `GET /v1/setup/jobs/{id}` (the same stream) and `GET /v1/setup/status` while the engine restarts: the page shows one restart card with the job's step and the load's phase, and the job's true end when the engine is back (OPS-175, VIS-209, VIS-211). Apply is refused while requests run (409 `busy`), while another job runs (409 `job_running`), when the selection does not fit (409 `over_budget`), and when the box lacks something (412 `missing_assets`, `missing_token`, `disk_full`, `no_overlay_hook`).

A failed step says what broke in plain words, from the runner's reason and the command's last lines: a script that lost its execute bit (tandem doctor repairs the mode at once and the step says so), a hold, too little memory, an engine that did not answer in time with its log's last line, a download that stopped (VIS-213). The failed and rolled-back screens list what tandem doctor repaired, offer Retry and Copy diagnostics. `POST /v1/setup/jobs/{id}/retry` starts a new job from the failed one's selection (`retry_of`): the check and tandem doctor run again and a finished download is not repeated, because the plan finds the weights. `GET /v1/setup/jobs/{id}/diagnostics` is the job as text for a bug report: the versions, the steps, tandem doctor's reports, the selection and the last 200 log lines, passed through the secret filter; where the browser does not let a page copy (plain http on a phone), the page shows it to select.

On a new box the fetch step downloads the 27B's K3 drafters, K4 draft head, BF16 drafters and base with the weights, and the plan's `download_gb` counts them (OPS-95). The NVFP4 profile's base is the FP8 checkpoint (30.4 GB). An EXL3 profile's is `base-plain` (OPS-133): the checkpoint's files fetched by range requests without the F8_E4M3 projections, their scales and `lm_head.weight`, which the EXL3 pack replaces, 3.54 GB; a box with the full checkpoint uses that. The review lists each download with its size (`downloads`), the tools' models too. Only a file nothing downloads fails the check with `missing_assets`, before anything changes; for a tool's model the error names the tool, and the fix is to turn it off in the Tools step.

## Removing a model

The weights step lists the weights on the box (`GET /v1/setup/weights`): each profile's folder with its size, the running one marked, a folder two profiles share as one row; the drafters, the draft head and the base checkpoint, which the models share, are listed apart and never removed here (SRV-560, VIS-215). Remove asks first, with the GB it frees and the line that downloads the weights again, then `POST /v1/setup/remove {profiles, confirm: true}` deletes the folder. It refuses the running profile (switch first, or tick "Remove ... after the switch" in the review step: `remove_after` of the apply, done once the new setup answers), a folder a kept profile shares, and any folder that is not a real folder under the home folder or the Hugging Face cache, holds the installation, or holds what something else reads; in the Hugging Face cache the repository's whole folder goes, only when nothing else reads from it. Each removal appends a line to `run/weights-removed.txt`: the time, the profile, its repository at its revision, the folder, the GB and `bash ops/use-profile.sh <profile> --fetch-only`. A download that does not fit is refused before it starts, with the GB too few and the weights that could be removed.

## Benchmarks and the overlay

`tools/exl3_fastrun.py --env-file ops/serve.env` reads serve.env without `run/setup.env` (it sources it with `QSE_SETUP_ENV` pointing nowhere): a benchmark row is the file's knobs, not what the box was last set up with. Its report says `setup_overlay: <path> (ignored)` when an overlay was there. Pass a knob with `--env KEY=VALUE` to measure an overlay's value.

## Where the wizard starts

A model the browser has no progress for starts from what the box runs now (`GET /v1/setup/models/{id}/current`): the active profile and every registry knob as `ops/serve.env` resolves it, `run/setup.env` included, so host, port, served name, the backends and every hand-tuned value are the box's own, and Review lists only what the user changed. A key serve.env leaves unset counts as what the engine then runs with, when that is one of the tool's two values: the lookup store's default folder from `engine/settings.py` (on), or the `${KEY:-default}` of `ops/start.sh` (`VISION` on, `RESPONSE_CACHE` off, `SESSION_SPILL` on). A box whose serve.env predates these keys starts with the tools it runs. A value the registry has no option for (a HOST that is neither address) stays at the recommended one and is listed in `unmatched`. Review warns when HOST, PORT, SERVED_MODEL or QSE_BACKENDS change, since every client notices. On an install.sh box the profiles and `active-profile` are read under serve.env's REPO, as `ops/use-profile.sh` reads them.

## Reset

`GET /v1/setup/reset` lists the levels with what each removes and how big it is; `POST /v1/setup/reset {level, confirm, dry_run}` removes one.

| level | what goes | confirm | restart |
|-|-|-|-|
| 1 | the page's progress and Wren's chat (the browser's own storage) | the page's | no |
| 2 | `run/setup.env` and its backups; the recommended profile starts | `true` | yes (a job, as apply) |
| 3 | the disk caches of every profile, the parked chats, the RAM caches, the lookup store (a running one is emptied by the engine, its folder stays), the shadow reader's disk | `delete caches` | no |
| 4 | the weights of every profile but the active one, and their HF cache copies; never what an engine on any port reads (its `--nvfp4`, `--fp8-head`, `QWEN38_EXL3`, `QWEN38_DRAFT_*`), nor `~/nvfp4-bf16` while `~/tandem-staircut` is there; nothing at all while the active profile is unknown (409 `active_unknown`) | `delete weights` | no |
| 5 | the usage ledger file and its `-wal`, `-shm` and `-journal` files (not its folder), the conductor's settings and journal, the memory notes | `delete history` | yes |

Levels 2 to 4 never touch the active profile's weights, the K3 drafters, the K4 draft head, the base checkpoint, the small models, the HF token, the secrets file the engine reads (`QSE_SECRETS`, else `$QSE_STATE_DIR/secrets.env`, else `~/.qwen38-spark-engine/secrets.env`; an install's `DIR/state/secrets.env`) and its folder, the history (`QSE_USAGE_LEDGER`) or the conductor's data. Level 5 never touches the secrets file either. A cache the running engine holds is cleared by the engine's own function; the shadow reader's disk is kept while it runs. A symbolic link goes as a link and its target stays. Every reset is refused while requests run or a job runs, and every call writes one line to `run/setup-audit.log`.

## The routes

All of them are admin routes: open while no admin password is set, and the admin token, a dashboard session or a tool on the box itself once one is. Every POST must also be same-origin JSON: `Content-Type: application/json` (else 415 `json_only`), no `Sec-Fetch-Site: cross-site`, and an `Origin`, when sent, that names the server's `Host` or `X-Forwarded-Host` (else 403 `cross_site` / `cross_origin`). A page on another site cannot run apply or reset through a visitor's browser, even on an open box or through a tunnel's loopback. The shapes are `dashboard/src/api/types.ts`, contract 1.0.

```text
GET  /v1/setup/models                  a tile per model
GET  /v1/setup/models/{id}             model.yaml as JSON, with wizard_fields
GET  /v1/setup/models/{id}/steps       the step schema and Wren's tools
GET  /v1/setup/models/{id}/current     what the box runs now as a selection: the wizard's start
GET  /v1/setup/device                  memory, GPU, disk, profiles on disk, Wren's state
GET  /v1/setup/community               every model's community packs and where the list came from
POST /v1/setup/community/sync          fetch the list from GitHub now (the page opened, Refresh)
POST /v1/setup/estimate                {selection} -> the estimate
POST /v1/setup/plan                    {selection} -> the review step's plan
POST /v1/setup/apply                   {selection, dry_run?} -> 202 {job}
GET  /v1/setup/jobs/{id}               SSE of the job (?format=json: the job)
GET  /v1/setup/reset                   the reset levels
POST /v1/setup/reset                   {level, confirm, dry_run?}
GET  /v1/setup/assistant/status        Wren's state; a poll keeps it loaded
POST /v1/setup/assistant/load          load Wren
POST /v1/setup/assistant/unload        unload Wren
POST /v1/setup/assistant/chat          {messages, page, tools?} -> SSE (/v1/setup/chat is the same)
```
