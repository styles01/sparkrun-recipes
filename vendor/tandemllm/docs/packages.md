# Packages

A package is what the conductor attaches to a turn by class: instructions the model follows, references it can read, and an eval set that says whether it helped. The shipped packages live under `packages/` ([server.md](server.md#the-package-library-r1-2026-10-03) says how the router injects them and how the library is searched). This page lists the first packages of OPS-39, their shape, and how each one is measured. Nothing below is measured yet: every line says so.

## The shape of a package folder

```
packages/system/<name>/        a system package: PACKAGE.md (the text that goes in, whole), manifest.json, eval/, references/
packages/library/<name>/       a first package: SKILL.md (body before `## Reference` goes in; the reference on request),
                               manifest.json, references/*.md, exemplars/*, eval/prompts.jsonl, eval/rubric.md
```

- `manifest.json`: name, version, kind, description, `max_tokens` (the cap of the text that goes in), `general`, `library`, `requires_tools`, `fallback`, the classes, the files, the eval set (its prompts, rubric, count, checks, human sample), `calibration` (a box-dependent threshold such as `fps_min`, written once by the calibration run), `measured` (null until a run exists), `signature` (the system set is signed at install, OPS-31).
- `eval/prompts.jsonl`: one prompt a line, `{id, class, prompt, checks, terms?, ...}`. Prompts are original (adapted, never copied), free of owner names, hosts and paths; `tools/packages_first.py` checks that.
- `eval/rubric.md`: the automatic checks and the judge items in the format of the `eval-judge` package (`packages/system/eval-judge/references/rubric-format.md`).
- `deliverable` (in `packages/seed.json`, SRV-615): what the package's result is for when a tool can make it, `send`, `schedule` or `post`. write-message and formal-letter carry `send`; event-plan, weekly-schedule and checklist-builder `schedule`; social-post `post`. On a request that offers a client tool whose declaration makes that result (a recipient and a message for `send`, a time and a subject for `schedule`; a post counts as a send), such a package gets one more side question before it goes in: does the user want it done with a tool, or the text itself? Without such a tool nothing is asked and the package goes in. On `tool` it stays out; the journal row's `package_choice.kept_out` names it with the tools whose declaration makes the deliverable (`signals.tool_effects`). The router's `tool_fit.deliverable` is `shadow` as shipped (journals under `shadow.tools.kept_out`, changes nothing), `on` or `off`.
- `packages/seed.json` registers every package; `packages/library/INDEX.md` names a library skill's classes. The router (`server/conductor/schema.py`, `SHIPPED_ROUTER`) lists a package on the classes whose package question may pick it, or on every turn of a class when it is `general`.

The texts are written for a small model: numbered steps, one instruction a line, concrete defaults, a Do/Don't pair per key rule, a "Done when" list the model can run. The injected part stays short (a library skill's body within the house 350 tokens); the detail is in the `## Reference` (served by `GET /v1/library/<name>`) and in `references/` and `exemplars/` (read through `package_read` once the tool server of SRV-105 exists).

## The first packages

| package | kind | goes in | cap | needs | what it teaches | eval | measured |
|-|-|-|-|-|-|-|-|
| `tandem-core` | system, general on every class but summarize | whole | 300 | nothing | the runtime's conventions: typed tool arguments, no think tags in text, one answer a turn, size limits, the stop rule for long files, a plain final answer | 8 prompts, automatic checks only | not measured |
| `render-tools` | system, general on the classes that build | whole | 750 | `render` | when to render, what to fix first (errors, motion, fps, sound, picture), three renders an answer, the CDN allowlist, the tool names of SRV-105 | 6 prompts | not measured; the tool server does not exist, so no chat offers `render` and the tool fit leaves it out |
| `eval-judge` | system, routed by no class | whole | 900 | nothing | the judge prompt: five items 1 to 5 from three frames and the probes, one JSON object, the same scores for the same input | 6 described cases | not measured |
| `creative-coding` | library skill (drawing, three_d, code_edit, design_ui) | body | 350 | nothing | one-file visuals with sound: a three-line plan, the hard rules (one file, seed, keys, WebAudio on a gesture), defaults; reference: skeleton, sound, easing, palettes; files: particles and boids, WebAudio patterns, pixel sprites, a Three.js skeleton, a 120-line exemplar | 24 prompts | not measured |
| `charts` | library skill (drawing, design_ui, code_edit, research) | body | 350 | nothing | an SVG chart from the prompt's data in the house style, every value as text, counted back; reference: the SVG grammar per form; file: the chart grammar | 20 prompts with data; `prompt_terms` reads every value back | not measured |
| `voxel-worlds` | library skill (three_d, code_edit) | body | 350 | nothing | the voxel build order with a render after each feature, the frame-rate rule, a to-do file, a seed; reference: rng, voxels, meshes, day-night, a state machine, audio, presets; files: six curated modules and two exemplars (prompt, plan, diff) | 20 prompts, one a six-feature sequence scored on the sixth's frame rate | not measured |
| `explainer-video` | library skill (drawing, code_edit) | body | 350 | `render_video` | a scene script and a cue list for the `render_video` tool, the return's checks, the honesty rules (every number names its source, schematics say so); reference: the fields; files: a 30-second example, the tool's interface spec | 12 prompts | not measured; without the tool its fallback `explainer-video-script` goes in |

Tokens are counted on the engine's tokenizer with `python tools/packages_first.py --tokenizer <dir>`; the table's caps are the manifests'.

## How a package is measured

The method is [the eval of VIS-E6](server.md): for a package P and its eval set, two sides on the same build in the same hour, one request at a time, the lookup store scrubbed of the eval outputs, the order alternating off/on per prompt:

- off: `tandem-core` only, greedy, thinking off, `max_tokens` 24,000, seed 42;
- on: the same request with P; when P needs the render tool the on side is an agent turn with at most 3 render calls;
- a third row, "off with tools", separates the tool's effect from the instructions'.

Reported per package: pass rate (every automatic check passed), judge score (`eval-judge`, the served model through its image input, temperature 0), the human score on the ten-prompt sample (blind, the reviewer named), the three deltas, tokens and tok/s per side, renders per prompt. A run records the build, the profile digest, the package version, the date and hour, the judge version and the raw JSON's hash. The table goes into the package's note, append-only, and the dashboard tile shows `pass 17/20 · judge 3.8/5 · +0.9 vs off · <date>`; without a run it shows "not measured". A surprising delta is "needs verification" until a second run in another hour agrees.

The `fps_min` threshold is calibrated on the box once (the exemplar page at 640x360 on the box's software WebGL, minus 20 %) and written into the manifest's `calibration`; a package without a calibration record fails that check.

Until the tool server (SRV-105) and the harness (OPS-37) exist, the sets and the rubrics are the deliverable, and every gain a package is expected to bring is a hypothesis.
