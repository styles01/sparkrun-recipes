# Community weights

A community pack is a set of weights someone else published on Hugging Face that TandemLLM can load: a smaller quantisation, or a fine-tune of a model we serve. The setup guide lists each one under its model, after our own profiles, with the author's GitHub avatar, the Hugging Face repository as owner/name, the format, the size and the speed a maintainer measured.

Some community quants are the packs one of the model's own profiles serves: turboderp's EXL3 packs of Qwen3.8-Flash-Next are the `flashnext-<bpw>` profiles. Their entries (`models/qwen3.8-flash-next/community/turboderp-flashnext-<bpw>.json`) name that profile in `profile`, so the card says who made the pack and its Use button selects the profile. They ship with the release in `models/community-builtin.json`, beside the profiles they describe; a box never fetches them.

Listing a pack takes one JSON file. You send a pull request that adds `models/<model>/community/<id>.json`, and nothing else.

## What happens to your file

1. You open a pull request with the file. The `community` check on the pull request runs `python -m server.setup.registry --check` and `python -m server.setup.community --index --check`.
2. A maintainer reads it, then downloads the pack at your commit and runs it on a DGX Spark. If it loads and answers, the maintainer fills `verified` and sets `status` to `verified`. If it does not, the entry stays `submitted`, or becomes `broken` with a note.
3. After the merge, every TandemLLM box sees the pack the next time someone opens the weights step of the setup guide. The box asks GitHub for `models/community-index.json` on the main branch. When GitHub does not answer, the box uses its last copy, then the copy that came with its release.

To have a maintainer write the file for you, open an issue with the "Add a community weight pack" form (`.github/ISSUE_TEMPLATE/community-quant.yml`). The setup guide's "Add yours" button opens that form with the model already filled in.

## The file, field by field

The file name is the id: `models/qwen3.8-27b/community/thinkingcap-27b-kearuga-exl3-4.00.json` holds the entry with `"id": "thinkingcap-27b-kearuga-exl3-4.00"`. That file is the first entry and a good one to copy. The JSON Schema is `models/schema/community.schema.json`.

| Field | Required | What it holds |
|---|---|---|
| `schema` | yes | `1` |
| `id` | yes | The file name without `.json`. Lowercase letters and digits, with single dots or dashes between them, at most 80 characters. |
| `name` | yes | The name the card shows and the model calls itself when asked, at most 120 characters. |
| `base_model` | yes | The TandemLLM model it runs on, which is the folder it sits in: `qwen3.8-27b` or `nemotron-3.5-super-vl-120b-a12b` ([Bases](#bases)), or any model when `profile` names one of its own profiles. |
| `finetune_of` | no | The Hugging Face id of the fine-tune, or `null` for a quantisation of the stock model. |
| `author.github` | yes | Your GitHub user name. The card shows `https://github.com/<you>.png` and links to your profile. |
| `author.name` | no | The name you go by. |
| `author.social` | no | Any of `x`, `huggingface` and `bluesky` as handles (`0xWhiteMage`, `name.bsky.social`), and `mastodon` and `website` as https URLs. No other keys. |
| `hf.repo` | yes | `owner/repo` on Hugging Face. It must be public: a box downloads it without a token. |
| `hf.revision` | yes | The full 40-character commit. A branch name is refused, because a branch can change after the review. |
| `format` | yes | `exl3`. The schema also knows `nvfp4`, `fp8` and `gguf`, and the check refuses them until TandemLLM can load them. |
| `bpw` | yes | Bits per weight, between 1 and 16. |
| `size_gb` | yes | The download in GB (10^9 bytes): the sum of the repository's file sizes at that commit. |
| `load` | yes | `standalone` when the pack is the whole model, or `overlay` when it holds EXL3 projections that go over the base checkpoint. |
| `modalities` | yes | `text`, `image`, `audio` and `video`, each `true` or `false`: what works in TandemLLM today. |
| `modalities_in_pack` | no | What the pack contains, even when TandemLLM cannot serve it yet. The card shows those in grey. |
| `drafters` | yes | `tandem-exl3` for our EXL3 draft models (K3 drafters and K4 draft head), `dflash2` for our BF16 draft models, `in-pack` for a draft layer inside the pack (only with `profile`), or `none`. On Nemotron 3.5 Super VL only `none`: the pack brings no drafter of its own and drafts with the model's own ([Bases](#bases)). |
| `license` | yes | The license as its SPDX names, e.g. `Apache-2.0`. |
| `license_url` | no | An https link to the license file, best at your commit. |
| `description` | no | One or two plain sentences, at most 500 characters. The card shows it as text: no HTML, no Markdown. |
| `tags` | no | Up to 12 lowercase words, such as `fine-tune` or `reasoning`. |
| `added` | yes | The date of the pull request, `YYYY-MM-DD`. |
| `status` | yes | `submitted`. A maintainer changes it. |
| `verified` | no | Filled by the maintainer who ran it: `build`, `device`, `date`, tok/s per kind of text (`prose`, `code`, `chat`, `plain`), tokens per round, and a note. |
| `links` | no | https links, such as `pr` for a pull request that made the pack load. |
| `profile` | no | The model's own profile that serves this pack, for a pack TandemLLM already runs as one of its profiles. The entry must have that profile's repository, format and bits a weight. Set by a maintainer. |

The check refuses a modality the load path cannot serve. A standalone pack is text only for now (ENG-271), even when it carries a vision tower: write `"image": false` and list `image` in `modalities_in_pack`. An overlay keeps the base checkpoint's vision tower, so `"image": true` is allowed there. No path serves audio or video yet.

## Bases

| Base | Loader | Load paths | Serves | Drafters |
|---|---|---|---|---|
| `qwen3.8-27b` | the 27B family's (`engine/loader.py`) | `standalone`, `overlay` | text; images on `overlay` | `tandem-exl3`, `dflash2` or `none` |
| `nemotron-3.5-super-vl-120b-a12b` | NemotronH (`engine/models/nemotron_h`, [nemotron-h.md](nemotron-h.md)) | `standalone` | text | `none`: LiLiCorr and the lookup, from the model's own profile |

A Nemotron 3.5 Super VL pack (ENG-377) is served with the model's own EXL3 profile nearest in size (`nemotron-super-vl-exl3-4.0` for a 3.5 bpw pack): its memory limits, its sampler, NVIDIA's LiLiCorr drafter and the lookup, with the pack in place of that profile's weights. Its files must be a NemotronH checkpoint: `config.json` flat (`NemotronHForCausalLM`) or the Omni wrapper's `llm_config`, 88 layers, an EXL3 `quantization_config`, every shard the index names. A pack that lists `image` in `modalities_in_pack` holds the RADIO tower too: `vision_model.radio_model.*` and `vision_projector.*` (or `mlp1.*`) tensors, `preprocessor_config.json` and the config's `vision_config`. `python -m server.setup.community check-pack community/<id> [--dir <folder>]` checks the files and names what differs; the loader's audit checks every tensor's shape and rate at load.

## What is not accepted

- A pack that is private or gated, or that needs a token to download.
- A branch or tag in `hf.revision` instead of a commit.
- A format other than EXL3, until a loader for it is in TandemLLM.
- A base model outside [Bases](#bases), unless the entry names one of that model's own profiles in `profile`.
- A claim in `modalities` that the load path does not serve.
- Links that are not https, and HTML or Markdown in any text field.
- A license that does not let others download and run the weights.
- A pull request that changes anything besides the one JSON file. A maintainer regenerates the index `models/community-index.json` (and `models/community-builtin.json` for the entries with `profile`), or you run `python -m server.setup.community --index`.

## Running a pack

On the box, `ops/use-profile.sh community/<id>` writes the pack's profile from its entry, downloads the pack at the pinned commit into `~/.tandemllm/community/<id>`, checks every weight file against the sha256 Hugging Face publishes, and fetches our draft models if they are missing. Then it stops the engine, starts it on the pack and sends one test chat. It refuses a download that would leave less than 60 GB of disk free, and it rolls back to the previous profile when the start or the test chat fails. `--dry-run` prints every URL and size without downloading. The setup guide's "Use" button runs the same steps after its review page.

The engine estimates a pack's memory from our own profile of the same format and the nearest size (`exl3-4.0` for a 4.0 bpw EXL3 pack). The number is only an estimate until a maintainer has loaded the pack.
