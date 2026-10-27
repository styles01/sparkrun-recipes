# render_video: the tool's contract (interface spec)

Read with `package_read("references/render-video-interface.md")`. The tool does not exist yet;
this page is what it must do. The package is attached only to a chat that offers a tool with
`scenes` and `cues` arguments (the router's `render_video` capability), so the text above is
never shown to a model that cannot call it.

## Schema

```json
{"name": "render_video",
 "description": "Render an explainer video from a scene script and a cue list; returns the mp4 and its measurements.",
 "parameters": {"type": "object", "required": ["scenes", "cues"],
   "properties": {
     "scenes": {"type": "array", "items": {"type": "object"}, "description": "the scene script, see the explainer-video package"},
     "cues": {"type": "object", "description": "the cue list: bed, events, narration"},
     "asked_seconds": {"type": "number", "description": "the length the user asked for; the tool checks the total against it"},
     "out": {"type": "string", "description": "file name without extension, default explainer"},
     "size": {"type": "string", "enum": ["1280x720", "1920x1080"], "default": "1280x720"},
     "fps": {"type": "integer", "default": 30}}}}
```

## Return

```json
{"path": "explainer.mp4", "seconds": 31.2, "lufs": -18.1, "true_peak_dbtp": -1.6, "lra": 4.2,
 "black_max_s": 0.0, "scenes": 7, "frames_checked": [0.5, 15.6, 30.7],
 "frames": [{"at": 0.5, "image": "data:image/png;base64,..."}, {"at": 15.6, "image": "..."}, {"at": 30.7, "image": "..."}]}
```
or `{"error": "s4: number without source"}` (the rules table in scene-and-cue-format.md).

## What the tool does

1. Validates the lists (the rules table). An error returns at once; nothing is rendered.
2. Lays the scenes on a timeline: each scene's `seconds`, a 0.45 s cross-fade between scenes;
   narration lengthens a scene to the spoken sentence plus 0.5 s.
3. Draws every scene. Two renderers give the same picture: Manim (CPU, Cairo) for the schematic
   kind, or an HTML template per kind captured frame by frame with a headless browser. Title and
   number cards are HTML; schematics are Manim when it is installed, HTML boxes and arrows when
   not. The house palette: ground `#161d25`, ink `#eef2f7`, cobalt `#3987e5`, orange `#f7a24f`,
   monospace digits with tabular numerals.
4. Synthesises the sound with numpy/scipy, no samples: the bed (detuned triangle and sine
   partials, four chords, a slow low-pass sweep, a quiet filtered-noise pulse), the events
   (whoosh = band-passed noise sweep; chime = two-partial sine; tock and click = short filtered
   bursts), narration through the system voice (`say` on a Mac, `espeak-ng` on Linux) at the
   scene's start, the bed ducked 3 dB under it with 2 s ramps.
5. Mixes and normalises: two-pass ffmpeg `loudnorm` to -18 LUFS integrated, true peak -1.5 dBTP;
   muxes with the video (`-c:v libx264 -crf 18 -pix_fmt yuv420p`, AAC 192k).
6. Measures the output: duration (`ffprobe`), integrated LUFS and true peak (`ebur128`), the
   longest black run (`blackdetect`, threshold 0.1, minimum 0.2 s), three frames at 0.5 s, the
   middle and 0.5 s before the end, sent back as `data:` images so the model can look.

## Where it runs

The heavy dependencies (Manim, LaTeX for `MathTex`, ffmpeg, a browser) run on a remote runner, not
on the box beside the engine. The tool server forwards the call over the LAN, waits up to 10
minutes, and streams nothing: one call, one result. The runner writes the mp4 into the project
folder the client names (`out`), and the tool returns the path the client sees.

## Checks the eval harness runs on the file

`file_exists`, `length_range` (asked ±5 s for 30 s; ±10 % above 60 s), `audio_nonsilent` (RMS
above -40 dBFS over the whole file), `loudness_range` (-19 to -17 LUFS), `true_peak` (at or below
-1.5 dBTP), `no_black_second` (`black_max_s` at most 1.0), `sources_named` (every `number` and
`compare` scene has a `source`, read from the scene list the model sent), `schematic_labelled`.
The judge sees the three returned frames with the package's rubric; a human listens to the sample.
