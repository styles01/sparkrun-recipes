# The render tool server: interface (SRV-105)

This is the contract the render-tools package is written against. The server does not exist yet;
when it is built, its tools must match this page, and the package's eval runs against it.

Offered as an MCP server and as an OpenAPI tool server beside the engine, so every client
(an agent CLI, a chat UI, the dashboard's Playground, the eval harness) calls the same thing. The
engine never executes tools; the client does.

## render

```json
{"name": "render",
 "parameters": {"type": "object", "required": ["html"],
   "properties": {"html": {"type": "string", "description": "the whole page"},
                  "frames_at": {"type": "array", "items": {"type": "number"}, "default": [0.5, 2, 5]},
                  "size": {"type": "string", "enum": ["640x360", "1280x720"], "default": "640x360"}}}}
```

Returns (one tool result with image parts and one JSON text part):

```json
{"frames": [{"at": 0.5, "image": "data:image/png;base64,..."}, {"at": 2, "image": "..."}, {"at": 5, "image": "..."}],
 "console": {"errors": ["ReferenceError: ctx is not defined at index.html:41"], "warnings": [], "error_count": 1, "warning_count": 0, "unhandled_rejections": 0},
 "fps": {"median": 57.0, "min": 31.0, "window_s": [2, 5]},
 "stalled": false,
 "bytes": 18422,
 "external_urls": ["https://unpkg.com/three@0.170.0/build/three.module.js"],
 "allowlist_ok": true}
```

Rules of the server: a sandboxed headless browser, no network except the allowlist, 1280x720 or
640x360 viewport, device pixel ratio 1, 10 s wall clock, software WebGL. A page that never paints
returns `stalled: true` and no frames after the first. A frame is sent as a `data:` URL because the
engine's https fence refuses private hosts and `data:` needs no network.

Cost: every frame is prompt rows for the vision tower (a 640x360 frame about 900 rows, a 1280x720
frame about 3,600 rows). That is why the default is the small size and three frames.

## audio_probe

```json
{"name": "audio_probe",
 "parameters": {"type": "object", "required": ["html"],
   "properties": {"html": {"type": "string"}, "seconds": {"type": "number", "default": 10}}}}
```

Returns `{"has_audio_context": true, "rms_dbfs": -24.1, "peak_dbfs": -6.3, "single_tone": false,
"seconds": 10, "started_by": "click"}`. The probe dispatches a click and a key press on the page
before it listens, because a context created outside a gesture stays suspended. The check
`audio_nonsilent` passes at RMS above -40 dBFS, peak below -1 dBFS and `single_tone` false.

## package_read, package_search

```json
{"name": "package_read", "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}}
{"name": "package_search", "parameters": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}}
```

`path` is relative to the active package (`references/webaudio-patterns.md`,
`exemplars/bouncing-ball.html`). `package_search` is BM25 over the active package's `references/`
and `exemplars/` split into passages of about 40 lines; it answers the five best as
`[{"path", "start_line", "text", "score"}]`. The model chooses the query; no keyword rule on the
request picks a file.

## render_video

```json
{"name": "render_video",
 "parameters": {"type": "object", "required": ["scenes", "cues"],
   "properties": {"scenes": {"type": "array", "items": {"type": "object"}},
                  "cues": {"type": "object"},
                  "out": {"type": "string", "description": "file name without extension, default explainer"}}}}
```

Returns `{"path": "explainer.mp4", "seconds": 31.2, "lufs": -18.1, "true_peak_dbtp": -1.6,
"black_max_s": 0.0, "scenes": 7, "frames_checked": [0.5, 15.6, 30.7]}`. The scene and cue formats
are in the explainer-video package's reference. The heavy dependencies (Manim or an HTML scene
capture, numpy/scipy for the bed, ffmpeg `loudnorm`) run on a remote runner; the tool server
forwards the call and waits (a time limit of 10 minutes, then an error the model reports).

## Tool capabilities for the router

The router reads what a chat's tools can do off their schemas (SRV-243). A tool with a string
`html` argument gives the capability `render`; a tool with `scenes` and `cues` gives
`render_video`. The packages name what they need in `requires_tools`; a chat without the tool never
sees render-tools or explainer-video.
