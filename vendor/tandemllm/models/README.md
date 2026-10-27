# The model registry

One folder per base model. The dashboard's setup guide (`#/setup`, [docs/setup.md](../docs/setup.md)) shows one tile per `models/<id>/model.yaml` and builds its steps from the file. `server/setup/registry.py` reads it with YAML's safe loader and checks it; `python -m server.setup.registry --check` prints what is wrong with a file and where.

```text
models/<id>/model.yaml           the entry, schema 1
models/<id>/measurements/*.json  the raw data behind the measured table (optional)
models/<id>/README.md            more about the model (optional)
models/schema/model.schema.json  the schema as a JSON Schema file (python -m server.setup.registry --schema)
models/<id>/community/<pack>.json  a community pack: weights someone else published (optional)
models/schema/community.schema.json  its schema (python -m server.setup.registry --community-schema)
models/community-index.json      every community pack in one file (python -m server.setup.community --index)
models/community-builtin.json    the community packs one of the model's own profiles serves (`profile`), same command
```

A community pack is one JSON file that a pull request adds: a Hugging Face repository at a pinned commit, its format and size, its author and what a maintainer measured. [docs/community-weights.md](../docs/community-weights.md) lists every field and what is not accepted.

What a `model.yaml` holds, in the order of the file:

- `schema: 1`, `id` (the folder's name), `name`, `family`, `params`, `pitch` (one line for the tile), `architecture`, `license`, `links`, `icon`, `recommended`, and optionally `order`, the tile's place on the models step (lower first, 0 when absent; equal places sort by folder name).
- `devices`: the boards the numbers were measured on. `total_gib` is MemTotal, `reserved_gib` what the OS and other programs hold outside the engine.
- `assets`: files a profile or a tool needs besides the profile's weights, with `fetchable` (something here downloads it) and `protected` (reset level 4 never deletes it).
- `profiles`: one entry per `ops/profiles/<id>.env`, with the weights and drafters in GiB from the load log, the download size in GB, the quality against BF16 with its date and source, the assets it needs and exactly one `recommended: true`. `made_by` says who made the weights: `kind` is `ours` (TandemLLM's own packs, `github: 0xBakeer`), `vendor` (the model's maker, e.g. `name: NVIDIA`) or `community` (someone else, with their GitHub login in `github`); `repo` names the weights' Hugging Face repository when it is not `hf_repo`. The setup guide shows the repository and the maker on every weight card.
- `measurements`: the measured table of one build, a cell per mode and task as `[tok/s, min, ms a round, tokens a round, n(, TTFT s)]`. A row from another build is `comparable: false` and says why in `note`.
- `memory`: the engine's own parts beside the weights (CUDA graphs, warm-up workspace, allocator reserve), with sources.
- `components`: the tools beside the model. `knobs` maps a serve.env key to its value with the tool on and off; quote `"on"` and `"off"`, which YAML otherwise reads as booleans.
- `parallelism`: slots, windows, the KV pool, the memory floor, the KV bytes a token and the keys they write.
- `methods` and `advanced`: the fields of the methods step, each with its type, range, default, recommended value and serve.env key.

Memory is in GiB (2^30 bytes) everywhere, disk in GB (10^9 bytes). `tests/test_setup_registry.py` holds the files to the repository: every serve.env key must be one the engine or the scripts read, every profile in `ops/profiles/` must have an entry, and the measured table must equal its raw file.
