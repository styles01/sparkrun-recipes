# Memory notes

The model keeps notes about the user and the work, as files, and reads them back in later turns and sessions. One note is one markdown file with a frontmatter: one fact, a title of at most 120 characters, a body of at most 400 words. The files are the truth; the dashboard's Memory screen (Notes tab) reads and edits them.

## Where the files are

```text
~/.qwen38-spark-engine/memory/notes/<user>/
  INDEX.md                               generated each night: one line per live note
  user/<kind>-<slug>--<id>.md            about the person
  projects/<name>-<hash8>/<kind>-...md   about one working directory
  sessions/<session>/<kind>-...md        true for one conversation only
  revisions/<id>/<n>.md                  every earlier text of a note
  archive/<yyyy-mm>/...                  archived notes, deleted after 30 days
  usage.json                             how often each note went into a request, by day
```

`<user>` is `owner` on the engine. A request the proxy relayed carries the proxy user's name (the key's label in process, `X-Tandem-User` from a loopback address or one listed in `QWEN38_MEMORY_PROXIES`); a relayed request without one reads and writes nothing. Files are `0600`, and the store refuses a root inside a git work tree, as the transcripts do.

TandemLLM serves one user. Every client that reaches the engine (opencode, Open WebUI, a script with curl) is that user, so a "remember that ..." from any of them is a note in the same store and is read back in the next conversation; that is the intended behaviour, not a leak. A write is never silent: the answer carries the state line `› remembered: <title>` (or `forgot`, `proposed`) in its content, after the answer text, in every client (opencode shows it as the title of a reasoning part, its state-line channel), and the Notes screen lists each note with its writer and the request that wrote it, where it is archived or deleted.

A request that must not touch the notes says so with the header `X-Tandem-Memory`: `off` reads nothing, writes nothing and gets no memory card (the journal row has `extra.memory = {mode: off, why: header}`); `read` gets the reads as the profile gives them, no card, and a block the model writes anyway is hidden from the answer and dropped (journaled `read_only`). Tests, benchmarks and smoke checks against a live engine send `X-Tandem-Memory: off`. Any other value, or no header, leaves the profile's `memory` switch in charge (SRV-173, 2026-10-06: a self-test's curl "remember that my dog is called Biscuit" was recalled in the next conversation).

The eight kinds: `rule` (how the user wants things done), `ruling` (a decision), `gotcha` (what went wrong once), `runbook`, `program_state` (with `verify_by`; stale after that date and never injected), `reference`, `registry`, `archive`.

## How the model writes

It writes a block in its answer, usually after its text:

```text
<tandem_memory op="add" scope="project" kind="gotcha">
title: the build needs node 20
tags: build

npm run build fails on node 18 with "Unexpected token 'with'".
</tandem_memory>
```

`op` is `add`, `update` (with `id:`) or `archive` (with `id:` and `reason:`). The server holds the block back from the client the way it holds a `<tool_call>` (`server/memblock.py`), writes the note, and the client sees one state line, `› remembered: the build needs node 20`, when the profile shows state lines. The line goes into the content, after the answer, whatever the profile's state-line channel and the request's `reasoning_format` (0.3.43-test, SRV-185: in `tags` a line after `</think>` had no place and was dropped, and a `reasoning_content` line after the answer went unseen, so Open WebUI and the OpenAI SDK never showed a write); opencode's `inline` channel keeps it as a reasoning part's title. The server takes the line out of the history a client sends back, so the model never reads it. With `X-Tandem-Memory: off` or `read` nothing is written and no line is shown. A closed block never reaches the client, whether it was written or not (SRV-277, 0.4.5-test): one over 2,000 characters, the fifth of an answer, or one the store refused is hidden like a written one, and its reason goes to the journal (`memory_notes`/`write`, the row's `extra.memory.failed`) and the log line `[memory] <request>: 1 block(s) not written (cap), hidden from the answer`; its tokens are hidden spans, so the next turn still resumes at the end anchor. Before, a refused block went out as the markup it was: an Open WebUI chat showed `<tandem_memory scope="project" ...>` to the user. Only text that is not a closed block goes out as written: a block that never closed, one still open after 4,000 characters, `<tandem_memoryx`. A block inside the reasoning is left alone.

A conversation without a working directory (every Open WebUI chat, a script with curl) has no project. A `scope="project"` add there is written at user scope (`rescoped: project` in the result and the journal), since every client is the one user and a note is meant to be read back in the next conversation (SRV-173); a session note would be gone in the next chat. Before 0.4.5-test it was refused (`no_project`).

The rules are code, not prompt text (`server/conductor/notes.py`):

- A write over a cap is refused, never cut: title 120 characters, body 400 words, 20 notes a session, 500 live notes a user.
- Every title and body passes the secret-pattern refusal of the transcripts; a body that is mostly a secret is refused.
- A `rule` or `ruling` about the user needs the user's own message in the turn. A tool turn or a script cannot write one.
- The model writes `provisional` notes. Ratify, pin and delete are the dashboard's. A model revision of a ratified note makes it provisional again, and so does the model's archive of one: the note stays, with the model's reason in its changelog, until the user archives it or ratifies it again on the Notes tab. The client sees no `forgot` line for it.
- Before an `add` lands, the same scope is searched: the same normalised title, or one of BM25's top 5 whose word overlap is at or above `dedupe_threshold` (0.6), turns the add into a revision of that note.

The model's scripts have the same write as a route, under the session's address only: `POST {tandem_url}/v1/memory`, `GET {tandem_url}/v1/memory/search?q=`, `GET {tandem_url}/v1/memory/<id>` (`server/memory_api.py`). Without a known session token the routes answer 403.

## How the model reads

A profile's `memory` switch: `off` (the default; the request is the release's), `shadow` (the card goes in and blocks are written, nothing is read back) or `on`. With `on`, three tail notes, each replayed at its place on later turns so each costs its prefill once:

- the memory card, about 150 tokens: the block's shape and when to write one;
- the index, on the first turn of a session and again after a compaction: the pinned ratified rules, then one line per note the request may see (`settings.memory.notes.index_tokens`, 1,500);
- the matching notes, on a new user message (never a tool turn): BM25 over the message, at most 3 notes that share at least 2 of its words, within 600 tokens. On a session's first turn they go in after the index. Their note has its own frame: the router's wrapper says "do not mention it", these are the user's notes, so the frame and its head tell the model to answer from a note that answers the message, to say it comes from its notes (a ratified note is confirmed by the user), and to tell the user both when the files disagree.

Laya's question "does this message ask to remember, forget or recall" attaches the long `memory` skill (`settings.memory.notes.intent`, shadow by default: journaled only).

## The background writer

Besides the blocks in its answers, a session's facts are written by a hidden turn of Qwen itself, thinking off, on the conversation the engine already holds (`server/conductor/memwriter.py`, `server/app.py` `memory_once`). `settings.memory.notes.writer`: `mode` off (the default), shadow (the turn runs, its blocks are counted, nothing is stored) or on.

- The triggers: the session's end (no request for `idle_s`, 180 s, and `min_new_tokens` new since its last pass), a GIVE_UP, and a compaction. With `after_compaction: idle` (the default) the compaction's turn waits for the next idle moment like the others; with `turn` it runs right after the summary, before the engine goes to the next request (at most `turn_deadline_s`). A compaction served as a guest (a short prompt) has no record in place; its mark waits for the idle look.
- The record: the session's last generation (its end anchor), else the last prompt in the rows up to its last message's end.
- The gate: Laya's "do these messages contain something to remember" on the session's last user messages; a no at 0.6 or more skips the turn. Not asked after a compaction in the turn (the decider asks only while no request holds the engine).
- The turn's prompt carries the notes that already match the session, so it updates instead of adding. Its output is parsed as the inline blocks are; a `tandem_memory` tool call (Qwen writes the call form when the session's tools are in the record) counts as its block. Notes land as `writer: qwen-writer`, provisional.
- Any other tool call ends the turn at its function's name (after a compaction Qwen answered an empty or a `bash` call). The run then says what NONE says: `none: true`, with `answer` (`none`, `empty`, `tool_call`) and `cut` (the function's name, or `empty`); the log line reads `wrote: NONE (tool_call: bash, cut)`.
- A request that arrives stops the turn at its next chunk; the conversation's rows and anchors stay as they were.
- The writer's last look is in its stats (`GET /v1/conductor/memory/notes/writes`, `writer.last_look`), and the log says `[memory] writer look: hold` (or `gpu`, `nothing`, `moved`) when a due session cannot be written, and `writer look: no session` once when the resident conversation never came with a memory session (no session id, raw, or a profile with memory off). Inside a hold the box lock is held and every look is `hold`: a test engine in a hold needs its own `BOX_LOCK`.
- `QWEN38_MEMORY_WRITER_TEXT`: a file a test engine appends each writer turn's text to. Never set it on the service; the text is the session's.
- Its own NONE (0.3.35-test): `ban_tool_call` (on by default) adds a logit bias of -100 to the `<tool_call>` token for the writer's turn, so a call cannot start; the cut above stays as the backstop. `prefill` (empty by default) is text the answer starts with; its tokens follow the generation prompt. The run records `ban` (the token ids, or `unavailable`) and `prefill` (its token count).
- Shadow (0.3.35-test): what a shadow run would write, each block after the secret-pattern refusal, is appended to `<user>/shadow/writer-<yyyy-mm-dd>.md` (`0600`). Nothing reads it back; the store reads only `user/`, `projects/` and `sessions/`, and the user's delete removes it with the rest.

### Behind the proxy

For a backend behind the proxy (vLLM, SGLang, any OpenAI-compatible server, a cloud API) the same turn is one more request to the backend the session used (`proxy/memwriter.py`): the messages of the session's last request as they were sent (the notes in place, so the backend's prefix cache matches), its answer's text, and the writer's user turn. Thinking is off, the session's tools stay in the body with `tool_choice: "none"` (`ban_tool_call`), and `prefill` is an assistant start where the backend can continue one (vLLM, SGLang). The answer goes through the engine writer's parser, store and journal; the run says `via: proxy`, `backend`, `model`, `cost_micros`, and the request is a ledger row of the client `harness:writer` (the Backends screen counts it under `memory`).

- When: the session's end (`idle_s`, `min_new_tokens`) and the client's compaction request in the same session (the session's last turn before it is the record, kept apart from the turns after it). One writer request at a time, only while the backend has no client request in flight. On a local backend a client request that arrives stops the writer's stream (the connection is closed, the session stays due); a cloud request runs to its end.
- The switches, per backend (`harness` document, `TANDEM_BACKEND_<ID>_WRITER*`): `writer` auto (on, but off on a paid cloud model: a cloud backend whose model has no price or a price above zero), off, shadow, on; `writer_usd_day` (0.25 USD) the spend a day on a paid model, checked against an estimate before each run (the prompt at the input price, `max_tokens` at the output price), `null` for none; `writer_runs_day` (48); `writer_prompt_tokens` (32,000 estimated; the oldest messages after the system prompt are dropped first). The effective mode is the lower of the backend's and `settings.memory.notes.writer.mode`. A paid model without a price is refused (`no_price`). Refusals are journaled once per record (`cap_runs`, `cap_cost`, `no_price`, `too_long`, `http_<status>`); the day's runs and spend come back from the ledger after a restart. The Backends API shows the mode per model (`writer.mode`, `writer.why`), the day's figures, the settings document's mode (`writer.settings_mode`) and where a shadow run's blocks go (`writer.shadow_log`). The Backends screen sets the four switches on a backend's form (a PATCH of `harness.writer*`, only the fields that changed; a field a variable sets is locked) and shows each model's resolved mode on the backend's page.

## What is journaled

The Notes tab's counts (`GET /v1/conductor/memory/notes`, `memory`): `index_notes` counts the notes each injected index listed, `body_notes` the note bodies that went in. Before 0.3.39-test a first turn counted both as the index, so a session whose body went in read `body_notes 0`.

Kind `memory_notes`: each write (`write`, `api_write`: op, id, scope, kind, why it was refused, redaction counts), each injection (`injected`: the note ids), the dashboard's edits (`human`), the nightly job. Never a title or a body.
